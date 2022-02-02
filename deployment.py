import subprocess

from flask import Flask, session, jsonify, request
import os
import json
from utils import check_dir


####################function for deployment
def store_model_into_pickle(dataset_csv_path, prod_deployment_path):
    move(os.path.join(dataset_csv_path, "trainedmodel.pkl"),
         prod_deployment_path)
    move(os.path.join(dataset_csv_path, "latestscore.txt"),
         prod_deployment_path)


def move(file_a, to_b):
    check_dir(to_b)
    subprocess.run(f"cp {file_a} {to_b}/", shell=True)

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_path = os.path.join(config['output_model_path'])
    prod_deployment_path = os.path.join("", config['prod_deployment_path'])
    store_model_into_pickle(model_path, prod_deployment_path)

    move(os.path.join(config['output_folder_path'], "ingestedfiles.txt"),
         prod_deployment_path)




        
        
        

