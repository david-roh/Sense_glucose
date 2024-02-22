import os
import tqdm
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libs.helper import align_ppg

def show_aligned_demo(row_data, row_data_aligned, sampling_rate=250):
    ecg_data = np.array(row_data[:sampling_rate])
    aligned_ecg_data = np.array(row_data_aligned[:sampling_rate])

    plt.figure(figsize=(12, 6))  # Adjust figure size
    plt.plot(np.arange(ecg_data.shape[0]), ecg_data, label="ECG Original", color='grey')
    plt.plot(np.arange(aligned_ecg_data.shape[0]), aligned_ecg_data, label="ECG Aligned", color='cyan')

    valp = int(row_data['p'])
    valq = int(row_data['q'])
    valr = int(row_data['r'])
    valt = int(row_data['t'])

    valp_aligned = int(row_data_aligned['p'])
    valq_aligned = int(row_data_aligned['q'])
    valr_aligned = int(row_data_aligned['r'])
    valt_aligned = int(row_data_aligned['t'])

    # Scatter plots with different markers and colors
    if valp >= 0 and valp < len(ecg_data):
        plt.scatter(valp, ecg_data[valp], color='green', marker='o', label="P Original")
    if valq >= 0 and valq < len(ecg_data):
        plt.scatter(valq, ecg_data[valq], color='blue', marker='X', label="Q Original")
    if valr >= 0 and valr < len(ecg_data):
        plt.scatter(valr, ecg_data[valr], color='red', marker='v', label="R Original")
    if valt >= 0 and valt < len(ecg_data):
        plt.scatter(valt, ecg_data[valt], color='purple', marker='*', label="T Original")

    # Similar for aligned data with different marker styles
    if valp_aligned >= 0 and valp_aligned < len(aligned_ecg_data):
        plt.scatter(valp_aligned, aligned_ecg_data[valp_aligned], color='green', marker='o', label="P Aligned")
    if valq_aligned >= 0 and valq_aligned < len(aligned_ecg_data):
        plt.scatter(valq_aligned, aligned_ecg_data[valq_aligned], color='blue', marker='X', label="Q Aligned")
    if valr_aligned >= 0 and valr_aligned < len(aligned_ecg_data):
        plt.scatter(valr_aligned, aligned_ecg_data[valr_aligned], color='red', marker='v', label="R Aligned")   
    if valt_aligned >= 0 and valt_aligned < len(aligned_ecg_data):
        plt.scatter(valt_aligned, aligned_ecg_data[valt_aligned], color='purple', marker='*', label="T Aligned")


    # Mark the R peak vertical line
    plt.axvline(x=row_data['r'], color='r', linestyle='--', label="R Peak Original")
    plt.axvline(x=row_data_aligned['r'], color='b', linestyle='--', label="R Peak Aligned")

    plt.legend(loc='upper right')  # Adjust legend position
    plt.title("ECG Peak: Original({}) vs Aligned({})".format(row_data['r'], row_data_aligned['r']))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("./demo/alignment.png", dpi=300)  # Save with high DPI    

def analyze_average_r_peak_position(processed_ecg_folder):
    # get the average R peak position
    print("Analyzing target r peak position...")

    folders = glob.glob(os.path.join(processed_ecg_folder, "*"))
    r_peak_positions = []

    for folder in tqdm.tqdm(folders):
        basename = os.path.basename(folder)
        df = pd.read_pickle(os.path.join(folder, "{}.pkl".format(basename)))

        _r_peak_positions = df['r'].to_list()
        r_peak_positions += _r_peak_positions

    r_peak_positions = np.array(r_peak_positions)
    print("R peak's position: ", r_peak_positions)
    print('R shape: ', r_peak_positions.shape)
    print("R peak's avg position: ", r_peak_positions.mean())
    print("R peak's std position: ", r_peak_positions.std())
    print("====================================")

    return int(r_peak_positions.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser("This is the script for aligning the R peaks for each ECG signal")
    parser.add_argument('--ecg', type=str, help='path to the processed ecg data')
    parser.add_argument('--all_ecg', default="/mnt/data2/mtseng/dataset/SeNSE/TCH_processed", type=str, help='path to the folder that stores all the processed ecg data')
    parser.add_argument('--r_peak_pos', default=None, type=int, help='target position of the R peak, if not provided, will analyze the average R peak position from all the processed ecg data')
    parser.add_argument('--out_folder', default="/mnt/data2/mtseng/dataset/SeNSE/TCH_aligned", type=str, help='path to the output aligned R peak folder')
    args = parser.parse_args()
    
    aligned_r_peak_pos = args.r_peak_pos
    if aligned_r_peak_pos is None:
        aligned_r_peak_pos = analyze_average_r_peak_position(args.all_ecg)

    print("Reading data from {} ~".format(args.ecg))
    print("Target R peak position: {}".format(aligned_r_peak_pos))
    df = pd.read_pickle(args.ecg)
    # I think its already sorted, but just in case
    df.sort_values('Time', inplace=True) 

    # remove NaN
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column].fillna(pd.Timestamp('1900-01-01'), inplace=True)  # Or pd.NaT
        elif pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(0, inplace=True)
        else:
            df[column].fillna('Missing', inplace=True)


    filename = os.path.basename(args.ecg).split('.')[0]

    if not os.path.exists(args.out_folder): 
        os.mkdir(args.out_folder)
    out_dir = os.path.join(args.out_folder, filename)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create an empty dataframe to store the aligned R peaks, only keep the "glucose", "time" columns that are needed
    sampling_rate = 250
    columns = list(np.arange(sampling_rate))
    columns.extend(['p', 'q', 'r', 't', 'Time', 'glucose', 'flag', 'hypo_label'])

    aligned_rows = []
    # iterate through each row in df
    for i in tqdm.tqdm(range(len(df))):
        row_data = df.iloc[i]
        aligned_row = align_ppg(row_data, aligned_r_peak_pos)
        aligned_rows.append(aligned_row)
    df_aligned = pd.DataFrame(aligned_rows, columns=columns)

    # save the aligned dataframe
    df_aligned.to_pickle(os.path.join(out_dir, "{}.pkl".format(filename)))

    # pick a random row to show the alignment
    random_row = np.random.randint(0, len(df_aligned))
    show_aligned_demo(df.iloc[random_row], df_aligned.iloc[random_row])

