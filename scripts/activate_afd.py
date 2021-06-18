import os
import time
import boto3
import json
import argparse
import pathlib

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str)
args = parser.parse_args()

region = args.region

#Initialize Boto3 session
boto3.setup_default_session(region_name=region)
boto_session = boto3.Session(region_name=region)

#initialize the AFD client 
client = boto3.client('frauddetector')

try:
    #Get training data schema file
    train_response_path = pathlib.Path('/opt/ml/processing/input')
    with open(train_response_path/'train_response.json') as f:
        train_response = json.load(f)
        
    model_id      = train_response['modelId']
    model_type    = train_response['modelType']
    model_version = train_response['modelVersionNumber']
    model_status  = train_response['status']
    
    if model_status == 'TRAINING_COMPLETE':
        #Activate AFD Model
        response = client.update_model_version_status (
                                modelId = model_id,
                                modelType = model_type,
                                modelVersionNumber = model_version,
                                status = 'ACTIVE'
                            )
    
        print("Activating model...")

        #-- wait until model is active 
        stime = time.time()
        while True:
            response = client.get_model_version(modelId=model_id, modelType = model_type, modelVersionNumber = model_version)
            if response['status'] != 'ACTIVE':
                print(f"Current progress: {(time.time() - stime)/60:{3}.{3}} minutes")
                time.sleep(60)  # wait for 1 minute 
            if response['status'] == 'ACTIVE':
                model_status = response['status']
                print(f"Model status : {model_status}")
                break

        etime = time.time()
        print("Elapsed time : %s" % (etime - stime) + " seconds \n"  )

        if response['status'] == 'ACTIVE':
            activation_response_path = pathlib.Path('/opt/ml/processing/output')
            with open(activation_response_path / 'activation_response.json', 'w') as outfile:
                json.dump(response, outfile)
        else:
            model_status = response['status']
            print(f'Unable to activate AFD model {args.model_name}...status= {model_status} Please check AFD logs.')
            os._exit(1)
            
    elif model_status == 'ACTIVE':        
        print(f"Model {model_id} version {model_version} already {model_status}")        
    else:
        print(f"Unable to activate. Model {model_id} with version {model_version}, status is {model_status}")
        os._exit(1)
            
except Exception as e:
    print(e)
    os._exit(1)



