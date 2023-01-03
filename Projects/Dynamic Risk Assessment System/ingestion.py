import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    list_files = []
    columns = ['corporation', 'lastmonth_activity',
               'lastyear_activity', 'number_of_employees', 'exited']

    df = pd.DataFrame(
        columns=columns
    )
    
    root_path = os.getcwd()
    saved_path = os.path.join(root_path, output_folder_path)

    if not os.path.isdir(saved_path):
        os.mkdir(saved_path)

    for file in glob.glob(f'{root_path}/{input_folder_path}/*.csv'):
        list_files.append(os.path.basename(file))
        df_tmp = pd.read_csv(file)
        df = pd.concat([df, df_tmp])
    
    clean_df = df.drop_duplicates()
    clean_df.to_csv(os.path.join(saved_path, "finaldata.csv"), index=False)

    with open(f"{saved_path}/ingestedfiles.txt", "w") as f:
        for filename in list_files:
            f.write(f'{filename}\n')


if __name__ == '__main__':
    merge_multiple_dataframe()
