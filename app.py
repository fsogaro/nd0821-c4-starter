import io
import subprocess
import pprint

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import model_predictions, dataframe_summary, \
    execution_time, outdated_packages_list



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/", methods=['GET'])
def welcome():
    return "welcome - valid urls are \n /prediction \n /scoring \n /stats "

@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    # take a dataset's file location as its input
    # call the prediction
    # function you created in Step 3
    folder_name = request.args.get('foldername')
    y = model_predictions(folder_name)
    return str(y)  # add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    subprocess.check_output(["python", "scoring.py"])
    with open(
            os.path.join(config['output_model_path'], 'latestscore.txt')) as f:
        lines = f.readlines()
    outscore = '<br>'.join([l for l in lines])
    out_str = f"model in {config['output_model_path']} scored: <br>" + outscore
    return out_str

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    data_path = os.path.join(
        config['output_folder_path'],
        "finaldata.csv")
    df_out = dataframe_summary(data_path)

    return df_out.to_html()

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    files = ["ingestion.py", "training.py"]
    timing = execution_time(files_to_time=files)
    broken = outdated_packages_list()

    string_out = f"""
    Running simple checks: <br> 
    broken dependencies: {broken} <br>
    times to run {files} = {timing}
    """
    return string_out

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
