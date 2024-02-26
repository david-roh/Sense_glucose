import os
import glob
import tqdm
import argparse
import pandas as pd
from datetime import datetime
import flirt.reader.empatica

def verify_folder(folder_path):
    # Check if folder exists
    if not os.path.isdir(folder_path):
        return False

    folder_name = os.path.basename(folder_path)
    
    # Check if folder contains ACC, BVP, EDA, HR, IBI, TEMP files
    file_types = ['ACC', 'BVP', 'EDA', 'HR', 'IBI', 'TEMP']
    for file_type in file_types:
        file_path = glob.glob(os.path.join(folder_path, '{}.csv'.format(file_type)))
        if len(file_path) != 1:
            print("Skipping folder: {} (Doesn't have {}.csv)".format(folder_name, file_type))
            return False

    return True

def load_and_merge_data(data_folder):
    try:
        bvp_data = flirt.reader.empatica.read_bvp_file_into_df(data_folder + '/BVP.csv')
    except ValueError as e:
        print(f"Error occurred in folder: {data_folder}, no BVP.csv file found.")
        raise e
    
    try:
        acc_data = flirt.reader.empatica.read_acc_file_into_df(data_folder + '/ACC.csv')
    except ValueError as e:
        print(f"Error occurred in folder: {data_folder}. Creating an empty DataFrame for ACC data.")
        acc_data = pd.DataFrame()

    bvp_data = flirt.reader.empatica.read_bvp_file_into_df(data_folder + '/BVP.csv')  # We need this at least

    try:
        eda_data = flirt.reader.empatica.read_eda_file_into_df(data_folder + '/EDA.csv')
    except ValueError as e:
        print(f"Error occurred in folder: {data_folder}. Creating an empty DataFrame for EDA data.")
        eda_data = pd.DataFrame()

    try:
        hr_data = flirt.reader.empatica.read_hr_file_into_df(data_folder + '/HR.csv')
    except ValueError as e:
        print(f"Error occurred in folder: {data_folder}. Creating an empty DataFrame for HR data.")
        hr_data = pd.DataFrame()

    try:
        ib_data = flirt.reader.empatica.read_ibi_file_into_df(data_folder + '/IBI.csv')
    except ValueError as e:
        print(f"Error occurred in folder: {data_folder}. Creating an empty DataFrame for IBI data.")
        ib_data = pd.DataFrame()

    try:
        temp_data = flirt.reader.empatica.read_temp_file_into_df(data_folder + '/TEMP.csv')
    except ValueError as e:
        print(f"Error occurred in folder: {data_folder}. Creating an empty DataFrame for TEMP data.")
        temp_data = pd.DataFrame()
        
    # Combine the sliced DataFrames
    data = pd.concat([bvp_data, acc_data, eda_data, hr_data, ib_data, temp_data], axis=1)
    
    # opted to not to forward fill acceleration data
    # forward fill the data for HR, EDA, TEMP as they have lower frequencies (1Hz, 4Hz, 4Hz respectively) than BVP/IBI (64Hz)
    # for col in ['eda', 'hr', 'temp']:
    #     data[col] = data[col].ffill()

    dtype_dict = {
        'bvp': 'float16',
        'acc_x': 'Int8',
        'acc_y': 'Int8',
        'acc_z': 'Int8',
        'eda': 'float32',
        'hr': 'float16',
        'temp': 'float16'
    }
    if 'ibi' in data.columns:
        dtype_dict['ibi'] = 'float16'

    data = data.astype(dtype_dict)

    # data['IBI_Presence'] = data['ibi'] > 0

    # for i in data.index[data['IBI_Presence']]:
    #     current_datetime = i
    #     ibi_datetime = current_datetime - pd.Timedelta(milliseconds=data.loc[i, 'ibi'])
    #     closest_datetime = data.index[data.index.get_indexer([ibi_datetime], method='nearest')[0]]
    #     data.at[closest_datetime, 'IBI_Presence'] = True
    return data

def combine_all_data(valid_folders):
    combined_df = pd.DataFrame()

    for folder_path in tqdm.tqdm(valid_folders):
        df = load_and_merge_data(folder_path)
        combined_df = pd.concat([combined_df, df])

    # Convert the datetime index to a timezone-naive datetime (remove timezone)
    combined_df.index = combined_df.index.tz_localize(None)
    
    return combined_df

def process_glucose(glucose_path):
    glucose_df = pd.read_csv(glucose_path, delimiter='\t')
    
    glucose_df = glucose_df[glucose_df['Event Type'] == 'EGV'].reset_index(drop = True)
    glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].str.replace('Low','40')
    glucose_df['Glucose Value (mg/dL)'] = glucose_df['Glucose Value (mg/dL)'].str.replace('High','400')

    glucose_df = glucose_df[['Timestamp (YYYY-MM-DDThh:mm:ss)','Glucose Value (mg/dL)']]
    glucose_df.columns = ['Timestamp','glucose']

    try:
        glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'],format = '%Y-%m-%dT%H:%M:%S')
    except:
        glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'],format = '%Y-%m-%d %H:%M:%S')
    glucose_df['glucose'] = glucose_df['glucose'].astype('int16')
    glucose_df = glucose_df.sort_values('Timestamp').reset_index(drop = True)

    return glucose_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine e4 files')
    parser.add_argument('--folder_path', type=str, help='path to folder containing e4 files')
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
    print("Combining e4 files")
    combined_df = combine_all_data(valid_folders)
    # save csv
    print("Saving csv", end="...")
    combined_df.to_csv('{}/combined_e4.csv'.format(args.out_folder), index=True)
    print("OK")
    # save pkl
    print("Saving pkl", end="...")
    combined_df.to_pickle('{}/combined_e4.pkl'.format(args.out_folder))
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
