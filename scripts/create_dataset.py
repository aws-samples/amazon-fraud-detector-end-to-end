import os
import time
import s3fs
import boto3
import json
import argparse
import pandas as pd
import numpy as np
import pathlib
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument('--signups-feature-group-name', type=str)
parser.add_argument('--outcomes-feature-group-name', type=str)
parser.add_argument('--region', type=str)
parser.add_argument('--bucket-name', type=str)
parser.add_argument('--bucket-prefix', type=str)
args = parser.parse_args()

region = args.region
signups_fg_name = args.signups_feature_group_name
outcomes_fg_name = args.outcomes_feature_group_name

#Initialize Boto3 session
boto3.setup_default_session(region_name=region)
boto_session = boto3.Session(region_name=region)

#initialize S3 client
s3_client = boto3.client('s3')

#initialize Sagemaker client and roles
sagemaker_boto_client = boto_session.client('sagemaker')
sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_boto_client)
sagemaker_role = sagemaker.get_execution_role()

#initalize Athena client
athena = boto3.client('athena', region_name=region)

#-----Declare global variables
sg_features=[]
oc_features=[]
sg_db = ''
sg_table = ''
oc_db = ''
oc_table = ''
afd_meta_labels = ['EVENT_TIMESTAMP', 'EVENT_LABEL']
ignore_col = 'EventTime'


#------Lookup feature store details-----------
# Initialize feature store runtime and session
def get_feature_store():
    featurestore_runtime = boto_session.client(
        service_name='sagemaker-featurestore-runtime', 
        region_name=region
    )

    feature_store_session = sagemaker.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_boto_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime
    )

    signups_feature_group = FeatureGroup(
        name=signups_fg_name, 
        sagemaker_session=feature_store_session)

    outcomes_feature_group = FeatureGroup(
        name=outcomes_fg_name, 
        sagemaker_session=feature_store_session)
    
    try:
        signups_fg_metadata = signups_feature_group.describe()
        outecomes_fg_metadata = outcomes_feature_group.describe()
        
        s_fg_status = signups_fg_metadata['OfflineStoreStatus']['Status']
        o_fg_status = outecomes_fg_metadata['OfflineStoreStatus']['Status']
        
        #Wait for feature stores to become Active
        stime = time.time()
        while True:
            if s_fg_status == 'Active' and o_fg_status == 'Active':
                print(f"Feature Store Offline Stores are Active")
                break            
            elif s_fg_status in ['CreateFailed', 'Deleting', 'DeleteFailed'] or o_fg_status in ['CreateFailed', 'Deleting', 'DeleteFailed']:
                print(f'Feature Group data ingestion problem: {signups_fg_name}:{s_fg_status}, {outcomes_fg_name}:{o_fg_status}')
                os._exit(1)
            else:
                print(f"Current progress: {(time.time() - stime)/60:{3}.{3}} minutes")
                print(f"Waiting for Feature Store Offline Stores to become Active")
                sleep(30)
                signups_fg_metadata = signups_feature_group.describe()
                outecomes_fg_metadata = outcomes_feature_group.describe()
                s_fg_status = signups_fg_metadata['OfflineStoreStatus']['Status']
                o_fg_status = outecomes_fg_metadata['OfflineStoreStatus']['Status']
                
    except Exception as e:
        print(e)
        os._exit(1)
        
    return signups_fg_metadata, outecomes_fg_metadata

#------Generate Athena Query based on features in the Feature store----
# this function by default finds the common columns in the two feature groups
# and sets the where clause using the common columns on the two tables. Modify as appropriate
def gen_query():    
    signup_features=[]
    outcomes_features=[]
    separator = ", "
    and_clause = " AND "
    
    for i in sg_features:
        if i['FeatureName'] != ignore_col: signup_features.append(i['FeatureName'])

    for i in oc_features:
        if i['FeatureName'] != ignore_col: outcomes_features.append(i['FeatureName'])
            
    signup_features = np.array(signup_features)
    outcomes_features = np.array(outcomes_features)
    
    # Common columns
    common = list(np.intersect1d(signup_features, outcomes_features))
    common_cols = [f'"{sg_table}".{i} as {i}' for i in common]
    
    diff_cols_signups = list(set(common).symmetric_difference(signup_features))
    diff_cols_outcomes = list(set(common).symmetric_difference(outcomes_features))
    
    join_string = [f'"{sg_table}".{i} = "{oc_table}".{i}' for i in common]
    join_clause = and_clause.join(join_string)
    
    select_stmt = f"""
        SELECT DISTINCT {separator.join(common_cols)},{separator.join(diff_cols_signups)},{separator.join(diff_cols_outcomes)} 
        FROM "{sg_table}" LEFT JOIN "{oc_table}" ON
        {join_clause}
    """
    
    #--Data schema metadata
    all_cols = np.unique(np.concatenate([signup_features,outcomes_features]))
    schema = {
        'modelVariables': list(set(afd_meta_labels).symmetric_difference(all_cols)),        
    }
    print(f'Variables: {schema}')
    
    return select_stmt, schema

#----Run Query on offline Feature Store datastore and generate training dataset
def gen_training_data(query, schema):
    try:        
        query_execution = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={
                'Database': sg_db
            },
            ResultConfiguration={
                'OutputLocation': f's3://{args.bucket_name}/{args.bucket_prefix}/afd-pipeline/query_results/'
            }
        )
        
        query_execution_id = query_execution.get('QueryExecutionId')
        query_details = athena.get_query_execution(QueryExecutionId=query_execution_id)        
        query_status = query_details['QueryExecution']['Status']['State']
        
        #--Wait for query to finish executing
        print(f'Query ID: {query_execution_id}')
        while query_status in ['QUEUED', 'RUNNING']:
            print(f'Query status: {query_status}')
            time.sleep(30)
            query_details = athena.get_query_execution(QueryExecutionId=query_execution_id)        
            query_status = query_details['QueryExecution']['Status']['State']
        print(f'Query status: {query_status}')
        
        query_result_s3_uri = query_details['QueryExecution']['ResultConfiguration']['OutputLocation']
        
        df_train = pd.read_csv(query_result_s3_uri)
        train_output_path = pathlib.Path('/opt/ml/processing/output/train')
        
        #--Write the final training dataset CSV file--
        df_train.to_csv(train_output_path / 'afd_training_data.csv', index=False)
        
        #--Generate Training data schema
        train_schema_path = pathlib.Path('/opt/ml/processing/output/schema')
        trainingDataSchema = {
            'modelVariables': schema['modelVariables'],
            'labelSchema':{
                'labelMapper': {
                    'FRAUD': [df_train["EVENT_LABEL"].value_counts().idxmin()],
                    'LEGIT': [df_train["EVENT_LABEL"].value_counts().idxmax()]
                }
            }
        }
        
        with open(train_schema_path / 'schema.json', 'w') as outfile:
            json.dump(trainingDataSchema, outfile)
        
        print(f'Training Dataset and Training Data Schema Generated: {trainingDataSchema}')
    except Exception as e:
        print(e)
        os._exit(1)
        

def gen_train_data():
    select_query, schema = gen_query()
    print(f'Athena Query: {select_query}')
    gen_training_data(select_query,schema)
            
    
signups_fg_metadata, outecomes_fg_metadata = get_feature_store()

if signups_fg_metadata['OfflineStoreStatus']['Status'] == 'Active' and outecomes_fg_metadata['OfflineStoreStatus']['Status'] == 'Active':
    print('Offline Data Store is active active')
    sg_features = signups_fg_metadata['FeatureDefinitions']
    oc_features = outecomes_fg_metadata['FeatureDefinitions']
    sg_db = signups_fg_metadata['OfflineStoreConfig']['DataCatalogConfig']['Database']
    sg_table = signups_fg_metadata['OfflineStoreConfig']['DataCatalogConfig']['TableName']
    oc_db = outecomes_fg_metadata['OfflineStoreConfig']['DataCatalogConfig']['Database']
    oc_table = outecomes_fg_metadata['OfflineStoreConfig']['DataCatalogConfig']['TableName']
    gen_train_data()
else:
    print('Offline Data Store is Inactive')
    os._exit(1)