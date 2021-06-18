import os
import time
import json
import boto3
import botocore
import argparse
import pathlib
import pandas as pd

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str)
parser.add_argument('--detector-name', type=str)
args = parser.parse_args()

region = args.region

#Initialize Boto3 session
boto3.setup_default_session(region_name=region)
boto_session = boto3.Session(region_name=region)

#initialize the AFD client 
client = boto3.client('frauddetector')

outcome_list = [
    {
        "name": 'verify_customer',
        "desc": 'this outcome initiates a verification workflow'
    }, 
    {
        "name": 'review',
        "desc": 'this outcome sidelines event for human or automated review'
    }, 
    {
        "name": 'approve',
        "desc": 'this outcome approves the event'
    }
]

def get_rule_ver(rule_id, expr, outcome):
    try:
        resp = client.update_rule_version(expression = expr,
                                          language = 'DETECTORPL',
                                          outcomes = [outcome],
                                          rule = {
                                                 'detectorId': args.detector_name,
                                                 'ruleId': rule_id,
                                                 'ruleVersion': "9999"
                                              })
    except botocore.exceptions.ClientError as error:
        if "whereas the most recent version is" in error.response['message']:
            rule_ver = error.response['message'].split('.')[0][-1]
            print(f'Rule version for {rule_id} is {rule_ver}')
            return rule_ver
        else:
            print(error)
            os._exit(1)
            
#--- Generate and create/update rules ---
def gen_create_rules(df_model, model_name):
    model_stat = df_model.round(decimals=2)  
    
    m = model_stat.loc[model_stat.groupby(["fpr"])["threshold"].idxmax()] 

    def make_rule(x):
        rule = ""
        if x['fpr'] <= 0.05: 
            rule = f"${model_name}_insightscore > {x['threshold']}"
        if x['fpr'] == 0.06:
            rule = f"${model_name}_insightscore <= {x['threshold_prev']}"
        return rule

    m["threshold_prev"] = m['threshold'].shift(1)
    m['rule'] = m.apply(lambda x: make_rule(x), axis=1)

    m['outcome'] = "approve"
    m.loc[m['fpr'] <= 0.03, "outcome"] = "review"
    m.loc[(m['fpr'] > 0.03) & (m['fpr'] <= 0.05), "outcome"] = "verify_customer"
    
    rule_set = m[(m["fpr"] > 0.0) & (m["fpr"] <= 0.06)][["outcome", "rule"]].to_dict('records')
    rule_list = []
    
    for i, rule in enumerate(rule_set):
        ruleId = f"rule{i}_{model_name}"
        
        print(f"Creating rule: {ruleId}: IF {rule['rule']} THEN {rule['outcome']}")
        try:
            response = client.create_rule(
                ruleId = ruleId,
                detectorId = args.detector_name,
                expression = rule['rule'],
                language = 'DETECTORPL',
                outcomes = [rule['outcome']]
                )
            rule_list.append({"ruleId": ruleId, 
                              "ruleVersion" : '1',
                              "detectorId"  : args.detector_name})
        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Message'] == "Failed to save rule since it already exists.":                
                print(f"Rule {ruleId} already exists in this detector...Updating")
                try:
                    rule_version = get_rule_ver(ruleId, rule['rule'], rule['outcome'])
                    
                    resp = client.update_rule_version(expression = rule['rule'],
                                                      language = 'DETECTORPL',
                                                      outcomes = [rule['outcome']],
                                                      rule = {
                                                         'detectorId': args.detector_name,
                                                         'ruleId': ruleId,
                                                         'ruleVersion': rule_version
                                                      })

                    rule_list.append({"ruleId": resp['rule']['ruleId'], 
                                      "ruleVersion" : resp['rule']['ruleVersion'],
                                      "detectorId"  : args.detector_name})
                except Exception as e:
                    print(f'Unable to update Rule {ruleId} : {e}')
                    os._exit(1)
            else:
                err = error.response['Error']['Message']
                print(f'Unable to update Rule {ruleId} : {err}')
                os._exit(1)
                
    return rule_list


try:
    #Get training data schema file
    activation_response_path = pathlib.Path('/opt/ml/processing/input')
    with open(activation_response_path/'activation_response.json') as f:
        activation_response = json.load(f)
    
    model_id      = activation_response['modelId']
    model_type    = activation_response['modelType']
    model_version = activation_response['modelVersionNumber']
    model_status  = activation_response['status']
    
    df_model = pd.DataFrame(client.describe_model_versions(
                    modelId= model_id,
                    modelVersionNumber=model_version,
                    modelType=model_type,
                    maxResults=10
                )['modelVersionDetails'][0]['trainingResult']['trainingMetrics']['metricDataPoints'])
    
    # Generate outcomes
    for outcome in outcome_list:
        outcome_name = outcome['name']
        try:
            client.get_outcomes(name = outcome_name)
            print(f"Outcome {outcome_name} already exists ...")
        except Exception as e:
            print(f"Creating outcome: {outcome_name} ...")
            client.put_outcome(name = outcome['name'],
                               description = outcome['desc'])
    
    #generate, create/update rules
    rule_list = gen_create_rules(df_model, model_id)
    
    response = client.create_detector_version(detectorId = args.detector_name,
                                              rules = rule_list,
                                              modelVersions = [
                                                  {
                                                      "modelId":model_id, 
                                                      "modelType" : model_type,
                                                      "modelVersionNumber" : model_version
                                                  }
                                              ],
                                              ruleExecutionMode = 'FIRST_MATCHED'
                                             )
    print(response)
    
except Exception as e:
    print(e)
    os._exit(1)



