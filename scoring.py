from datetime import datetime

from flask import Flask, session, jsonify, request
import pandas as pd
import pickle
import os
from sklearn import metrics
import json


#################Function for model scoring
def score_model(model_path, test_data_path, features, target,
                output_scores_folder):

    with open(model_path, 'rb') as file:
        clf = pickle.load(file)
    test_data = pd.read_csv(test_data_path)

    y_pred = clf.predict(
        test_data[features].values.reshape(-1, len(features)),

    )

    f1_score = metrics.f1_score(test_data[target].values, y_pred)

    with open(os.path.join(output_scores_folder, 'latestscore.txt'), 'w') as f:
        f.write(f"DATE: {datetime.now().strftime('%d-%b-%Y (%H:%M:%S.%f)')}, ")
        f.write(f"test score: {f1_score}")
    return f1_score

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    test_data_path = os.path.join(config['test_data_path'], "testdata.csv")
    model_path = os.path.join(config['output_model_path'], "trainedmodel.pkl")
    output_scores_folder = config['output_model_path']
    features = config["model_features_cols"]
    target = config["target_col"]

    f1_score = score_model(
        model_path, test_data_path, features, target, output_scores_folder)
    print("score:")
    print(f1_score)










