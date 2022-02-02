import pickle
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
deploy_folder = os.path.join(config['prod_deployment_path'])


#  ############# Function for reporting
def score_model():
    with open(os.path.join(deploy_folder, "trainedmodel.pkl"), 'rb') as file:
        clf = pickle.load(file)

    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    x = df[config["model_features_cols"]]
    y_true = df[config["target_col"]]
    plot_confusion_matrix(clf, x, y_true)
    fig = plt.gcf()
    fig.savefig(os.path.join(dataset_csv_path, "confusion_matrix.png"))
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace


if __name__ == '__main__':
    score_model()
