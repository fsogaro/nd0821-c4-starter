import json
import os
import subprocess

from diagnostics import model_predictions, model_performance


def new_data_check(new_data_path):

    new_files = [f for f in os.listdir(new_data_path) if ".csv" in f]
    try:
        with open('production_deployment/ingestedfiles.txt') as f:
            lines = f.readlines()
            for existing in lines:
                if existing.strip("\n") in new_files:
                    new_files.remove(existing.strip("\n"))
    except FileNotFoundError:
        pass
    return new_files


def model_dirft_check(new_data_path):
    try:
        with open('production_deployment/latestscore.txt') as f:
            lines = f.readlines()
            latest_score = float(lines[-1].split("test score: ")[-1])
    except FileNotFoundError:
        return True  # train from scratch

    drift_detected = False
    for file in new_files:
        datapath = os.path.join(new_data_path, file)
        y_pred = model_predictions(datapath)
        perf = model_performance(datapath, y_pred)
        # raw comparison test
        if perf < latest_score:
            drift_detected = True
    return drift_detected


if __name__ == '__main__':
    ##################Check and read new data
    #first, read ingestedfiles.txt
    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    with open('config.json', 'r') as f:
        config = json.load(f)
    new_data_path = config["input_folder_path"]

    new_files = new_data_check(new_data_path)

    ##################Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process here
    if len(new_files) == 0:
        exit()

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    drift_detected = model_dirft_check(new_data_path)

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if not drift_detected:
        exit()

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    subprocess.check_output(["python", "ingestion.py"])
    subprocess.check_output(["python", "training.py"])
    subprocess.check_output(["python", "scoring.py"])
    subprocess.check_output(["python", "deployment.py"])
    subprocess.check_output(["python", "diagnostics.py"])
    subprocess.check_output(["python", "reporting.py"])
