import os
import time
import json
import boto3
import argparse
import pathlib

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str)
parser.add_argument('--data-access-role', type=str)
parser.add_argument('--model-name', type=str)
parser.add_argument('--s3-file-loc', type=str)
args = parser.parse_args()

region = args.region

#Initialize Boto3 session
boto3.setup_default_session(region_name=region)
boto_session = boto3.Session(region_name=region)

#initialize the AFD client 
client = boto3.client('frauddetector')

def initiate_training():
    try:
        #Get training data schema file
        print(f'Attempting to load training Schema file')
        try:
            train_schema_path = pathlib.Path('/opt/ml/processing/schema')
            with open(train_schema_path/'schema.json') as f:
                trainingDataSchema = json.load(f)
            print(f'Loaded schema file : {trainingDataSchema}')
        except Exception as e:
            print(f'Unable to load schema file: {e}')
            os._exit(1)

        print(f'Attempting to train AFD Model: {args.model_name}')

        #Initiate AFD Model Training
        response = client.create_model_version(modelId     = args.model_name,
                                               modelType   = 'ONLINE_FRAUD_INSIGHTS',
                                               trainingDataSource = 'EXTERNAL_EVENTS',
                                               trainingDataSchema = trainingDataSchema,
                                               externalEventsDetail = {
                                                   'dataLocation'     : args.s3_file_loc,
                                                   'dataAccessRoleArn': args.data_access_role
                                               }
                                              )

        model_version = response['modelVersionNumber']

        print("Wait for model training to complete...")
        stime = time.time()
        while True:
            response = client.get_model_version(modelId = args.model_name, modelType = "ONLINE_FRAUD_INSIGHTS", modelVersionNumber = model_version)
            if response['status'] == 'TRAINING_IN_PROGRESS':
                print(f"Current progress: {(time.time() - stime)/60:{3}.{3}} minutes")
                time.sleep(60)  # -- sleep for 60 seconds 
            if response['status'] != 'TRAINING_IN_PROGRESS':
                print(f"Model status : {response['status']}")
                break

        etime = time.time()
        print("Model training complete. Elapsed time : %s" % (etime - stime) + " seconds \n"  )

        if response['status'] == 'TRAINING_COMPLETE':
            train_response_path = pathlib.Path('/opt/ml/processing/output')
            with open(train_response_path / 'train_response.json', 'w') as outfile:
                json.dump(response, outfile)

            # we will grab the model's AUC so that we can make a decision in the pipeline to make a decision to 
            # activate the model or not
            auc = client.describe_model_versions(
                        modelId= args.model_name,
                        modelVersionNumber=model_version,
                        modelType='ONLINE_FRAUD_INSIGHTS',
                        maxResults=1
                    )['modelVersionDetails'][0]['trainingResult']['trainingMetrics']['auc']

            auc_metric = { 'auc_metric': auc }

            train_auc_path = pathlib.Path('/opt/ml/processing/auc')

            with open(train_auc_path / 'train_auc.json', 'w') as outfile:
                json.dump(auc_metric, outfile)
        else:
            print(f'AFD model {args.model_name}..,Please check AFD logs.')
            os._exit(1)

    except Exception as e:
        print(e)
        os._exit(1)

print(f"Initializing AFD Model Training for model: {args.model_name}")
initiate_training()


