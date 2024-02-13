import os
import glob
import tqdm
import argparse
import pandas as pd
from datetime import datetime

# Attempt to convert it to a datetime object
def is_valid_timestamp(timestamp_string):
    try:
        # Define the format based on the given string
        format_string = "%Y_%m_%d-%H_%M_%S"
        converted_time = datetime.strptime(timestamp_string, format_string)
        return True, converted_time
    except ValueError:
        return False, None

def verify_folder(folder_path):
    # Check if folder exists
    if not os.path.isdir(folder_path):
        return False

    folder_name = os.path.basename(folder_path)
    # Check if folder name is a valid timestamp
    if not is_valid_timestamp(folder_name)[0]:
        return False
    
    # Check if folder contains ECG and summary files
    file_path = glob.glob(os.path.join(folder_path, '*_ECG.csv'))
    if len(file_path) != 1:
        print("Skipping folder: {} (Doesn't have _ECG.csv)".format(folder_name))
        return False
    
    file_path = glob.glob(os.path.join(folder_path, '*_SummaryEnhanced.csv'))
    if len(file_path) != 1:
        print("Skipping folder: {} (Doesn't have _SummaryEnhanced.csv)".format(folder_name))
        return False

    return True

# combine ECG files
def ECG_read_and_combine(valid_folders):
    ecg_combined_df = pd.DataFrame()
    for folder_path in tqdm.tqdm(valid_folders):
        file_path = glob.glob(os.path.join(folder_path, '*_ECG.csv'))[0]
        df = pd.read_csv(file_path)        
        ecg_combined_df = pd.concat([ecg_combined_df, df])

    return ecg_combined_df

# combine summary files
def summary_read_and_combine(valid_folders):
    summary_combined_df = pd.DataFrame()

    for folder_path in tqdm.tqdm(valid_folders):
        file_path = glob.glob(os.path.join(folder_path, '*_SummaryEnhanced.csv'))[0]
        df = pd.read_csv(file_path)
        summary_combined_df = pd.concat([summary_combined_df, df])
    
    return summary_combined_df

def process_glucose(glucose_path):
    glucose_df = pd.read_csv(glucose_path,delimiter=',')
    glucose_df = glucose_df[glucose_df['Event Type'] == 'EGV'].reset_index(drop = True) 
    glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].str.replace('Low','40')
    glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].str.replace('High','400')

    glucose_df = glucose_df[['Timestamp (YYYY-MM-DDThh:mm:ss)','Glucose Value (mg/dL)']]
    glucose_df.columns = ['Timestamp','glucose']

    try:
        glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'],format = '%Y-%m-%dT%H:%M:%S')
    except:
        glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'],format = '%Y-%m-%d %H:%M:%S')
    glucose_df['glucose'] = glucose_df['glucose'].astype(float)
    glucose_df = glucose_df.sort_values('Timestamp').reset_index(drop = True)

    return glucose_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine ECG and summary files')
    parser.add_argument('--folder_path', type=str, help='path to folder containing ECG and summary files')
    parser.add_argument('--glucose_path', type=str, help='path to glucose file')
    parser.add_argument('--out_folder', default=None, type=str, help='path to folder to store combined files')
    args = parser.parse_args()

    if args.out_folder is None:
        args.out_folder = args.folder_path

    print("===========================================")
    valid_folders = []
    # Iterate over only directories in the folder
    folders = glob.glob(os.path.join(args.folder_path, '*'))
    for folder in folders:
        if verify_folder(folder):
            valid_folders.append(folder)
    valid_folders.sort()
    print("Found {} valid folders".format(len(valid_folders)))

    print("===========================================")
    print("Combine ECG files")
    ecg_combined_df = ECG_read_and_combine(valid_folders)
    print("Saving csv", end="...")
    # save csv
    ecg_combined_df.to_csv('{}/ECG.csv'.format(args.out_folder), index=False)
    print("OK")
    # save pkl
    print("Saving pkl", end="...")
    ecg_combined_df.to_pickle('{}/ECG.pkl'.format(args.out_folder))
    print("OK")

    print("===========================================")
    print("Combine summary files")
    summary_combined_df = summary_read_and_combine(valid_folders)
    # save csv
    print("Saving csv", end="...")
    summary_combined_df.to_csv('{}/summary.csv'.format(args.out_folder), index=False)
    print("OK")
    # save pkl
    print("Saving pkl", end="...")
    summary_combined_df.to_pickle('{}/summary.pkl'.format(args.out_folder))
    print("OK")
    print("===========================================")
    print("Processing glucose file")
    glucose_df = process_glucose(args.glucose_path)
    print("Saving csv", end="...")
    glucose_df.to_csv('{}/glucose.csv'.format(args.out_folder), index=False)
    print("OK")
    # save pkl
    print("Saving pkl", end="...")
    glucose_df.to_pickle('{}/glucose.pkl'.format(args.out_folder))
    print("OK")
    print("===========================================")
    print("Done")
