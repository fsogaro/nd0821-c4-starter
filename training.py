import logging
import timeit

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from utils import check_dir

logFormatter = logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

fileHandler = logging.FileHandler("./model_dev.txt")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)



#################Function for training the model
def train_model(train_data, features,target, savepath):
    
    #use this logistic regression for training
    clf = LogisticRegression(C=1.0)
    
    #fit the logistic regression to your data
    clf.fit(
        train_data[features].values.reshape(-1, len(features)),
        train_data[target].values
    )
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(clf, open(savepath, 'wb'))
    return clf

if __name__ == '__main__':
    ###################Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])

    train_data = pd.read_csv(
        os.path.join(dataset_csv_path, "finaldata.csv")
    )
    check_dir(model_path)
    save_path = os.path.join(model_path, "trainedmodel.pkl")
    features = config["model_features_cols"]
    target = config["target_col"]

    starttime = timeit.default_timer()
    train_model(train_data, features, target, save_path)
    timing = timeit.default_timer() - starttime
    logger.info(f"----process time: {timing}----")

