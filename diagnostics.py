import pickle
import subprocess

import pandas as pd
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
import sklearn.metrics

with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
deploy_folder = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(data_path):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(deploy_folder, "trainedmodel.pkl"), 'rb') as file:
        clf = pickle.load(file)

    df = pd.read_csv(data_path)
    x = df[config["model_features_cols"]]
    y = clf.predict(x)

    return y.tolist()

def model_performance(data_path, predictions):
    df = pd.read_csv(data_path)
    y_true = df[config["target_col"]]
    return sklearn.metrics.f1_score(y_true, predictions)





##################Function to get summary statistics
def dataframe_summary(data_path):
    df = pd.read_csv(data_path)
    df_num = df.select_dtypes(include=np.number)
    df_stats = df_num.agg(["mean", "std", "median", "count"]).T
    df_stats["nulls_pctg"] = (len(df) - df_stats["count"])/len(df)
    return df_stats

##################Function to get timings
def execution_time(files_to_time=None):
    # calculate timing of training.py and ingestion.py
    if files_to_time is None:
        files_to_time = ["ingestion.py", "training.py"]
    timing = []
    for file in files_to_time:
        starttime = timeit.default_timer()
        subprocess.check_output(["python", file])
        timing.append(timeit.default_timer() - starttime)

    return timing

##################Function to check dependencies
def outdated_packages_list():
    broken = subprocess.check_output(["pip", "check"])
    installed = subprocess.check_output(['pip', 'list'])

    with open('broken.txt', 'wb') as f:
        f.write(broken)
    with open('installed.txt', 'wb') as f:
        f.write(installed)
    requirements = subprocess.check_output(['pip', 'freeze'])
    with open('requirements.txt', 'wb') as f:
        f.write(requirements)

    return broken


if __name__ == '__main__':
    data_path = os.path.join(test_data_path, "testdata.csv")
    model_predictions(data_path)
    dataframe_summary(data_path)
    execution_time()
    outdated_packages_list()

