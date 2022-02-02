import logging
import timeit
import pandas as pd
import os
import json
from datetime import datetime
from utils import check_dir

logFormatter = logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

fileHandler = logging.FileHandler("./logs.txt")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


#############Function for data ingestion
def merge_multiple_dataframe(input_folder,
                             output_folder, out_name="finaldata.csv"):

        path_in = os.path.join(os.getcwd(), input_folder)
        path_out = os.path.join(os.getcwd(), output_folder)
        check_dir(path_out)
        file_out = os.path.join(path_out, out_name)

        filenames = [f for f in os.listdir(path_in) if ".csv" in f]
        with open(os.path.join(path_out, 'ingestedfiles.txt'), 'a') as f:
            for filename in filenames:
                f.write(f"{filename}")
                f.write(f"\n")

        logger.info(f"found {len(filenames)} .csv files in {path_in}")
        df_list = pd.DataFrame()
        for file in filenames:

            df1 = pd.read_csv(os.path.join(path_in, file))
            df_list = df_list.append(df1)
            logging.info(f"name: {file}, "
                         f"rows: {df1.shape[0]}, "
                         f"columns: {df1.shape[1]}")

        logger.info(f"combined data with  rows: {df_list.shape[0]}, columns: "
                     f"{df_list.shape[1]}")
        result = df_list.drop_duplicates()
        result.to_csv(file_out, index=False)
        logger.info(f"final data with  rows: {result.shape[0]}, columns: "
                     f"{result.shape[1]}")


    #check for datasets, compile them together, and write to an output file



if __name__ == '__main__':
    #############Load config.json and get input and output paths
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger.info(f"---INGESTION DATE: "
                f"{datetime.now().strftime('%d-%b-%Y (%H:%M:%S.%f)')}")
    logger.info(f"---ingesting data from {config['input_folder_path']}")
    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    starttime = timeit.default_timer()
    merge_multiple_dataframe(input_folder_path, output_folder_path)
    timing = timeit.default_timer() - starttime
    logger.info(f"----process time: {timing}----")
