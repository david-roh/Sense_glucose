import os
from sympy import use
import tqdm
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libs.helper import align_ppg
import matplotlib.cm as cm

def show_aligned_demo(row_data, row_data_aligned, sampling_rate=64):
    ppg_data = np.array(row_data[:sampling_rate])
    aligned_ppg_data = np.array(row_data_aligned[:sampling_rate])

    plt.figure(figsize=(12, 6))  # Adjust figure size
    plt.plot(np.arange(ppg_data.shape[0]), ppg_data, label="PPG Original", color='grey')
    plt.plot(np.arange(aligned_ppg_data.shape[0]), aligned_ppg_data, label="PPG Aligned", color='cyan')

    valr = int(row_data['r'])
    valr_aligned = int(row_data_aligned['r'])

    if valr >= 0 and valr < len(ppg_data):
        plt.scatter(valr, ppg_data[valr], color='red', marker='v', label="R Original")

    if valr_aligned >= 0 and valr_aligned < len(aligned_ppg_data):
        plt.scatter(valr_aligned, aligned_ppg_data[valr_aligned], color='red', marker='v', label="R Aligned")   

    # Mark the R peak vertical line
    plt.axvline(x=row_data['r'], color='r', linestyle='--', label="R Peak Original")
    plt.axvline(x=row_data_aligned['r'], color='b', linestyle='--', label="R Peak Aligned")

    plt.legend(loc='upper right')  # Adjust legend position
    plt.title("PPG Peak: Original({}) vs Aligned({})".format(row_data['r'], row_data_aligned['r']))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("./demo/alignment.png", dpi=300)  # Save with high DPI

def plot_random_aligned_beats(df_aligned, num_beats=5, sampling_rate=64):
    colors = cm.rainbow(np.linspace(0, 1, num_beats))  # Generate as many colors as needed
    plt.figure(figsize=(12, 6))  # Adjust figure size

    for i in range(num_beats):
        random_row = np.random.randint(0, len(df_aligned))
        row_data_aligned = df_aligned.iloc[random_row]
        aligned_ppg_data = np.array(row_data_aligned[:sampling_rate])
        valr_aligned = int(row_data_aligned['r'])

        plt.plot(np.arange(aligned_ppg_data.shape[0]), aligned_ppg_data, label="PPG Aligned {}".format(i+1), color=colors[i])

        if valr_aligned >= 0 and valr_aligned < len(aligned_ppg_data):
            plt.scatter(valr_aligned, aligned_ppg_data[valr_aligned], color=colors[i], marker='v', label="R Aligned {}".format(i+1))   

    # Mark the R peak vertical line
    plt.axvline(x=valr_aligned, color='black', linestyle='--', label="R Peak Aligned")

    plt.legend(loc='upper right')  # Adjust legend position
    plt.title("Overlay of {} Aligned PPG Beats".format(num_beats))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("./demo/overlay_alignment.png", dpi=300)  # Save with high DPI

def analyze_average_r_peak_position(processed_ppg_folder):
    # get the average R peak position
    print("Analyzing target r peak position...")

    folders = glob.glob(os.path.join(processed_ppg_folder, "*"))
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
    parser = argparse.ArgumentParser("This is the script for aligning the R peaks for each PPG signal")
    parser.add_argument('--ppg', type=str, help='path to the processed ppg data')
    parser.add_argument('--all_ppg', default="/mnt/data2/david/data/TCH_processed", type=str, help='path to the folder that stores all the processed ppg data')
    parser.add_argument('--r_peak_pos', default=None, type=int, help='target position of the R peak, if not provided, will analyze the average R peak position from all the processed ppg data')
    parser.add_argument('--out_folder', default="/mnt/data2/david/data/TCH_aligned", type=str, help='path to the output aligned R peak folder')
    args = parser.parse_args()
    
    # print("Do you want to work with Neurokit systolic peak detection (no filtering) or Empatica's Diastolic point detection (filtered)? (n/e)")
    # ans = input()
    # if ans == 'n':
    #     print("Using Neurokit systolic peak detection ...")
    #     use_neurokit = True
    # elif ans == 'e':
    #     print("Using Empatica's Diastolic point detection ...")
    #     use_neurokit = False
    # else:
    #     raise ValueError("Invalid choice. Please enter 'n' or 'e'.")
    
    aligned_r_peak_pos = args.r_peak_pos
    

    print("Reading data from {} ~".format(args.ppg))
    df = pd.read_pickle(args.ppg)
    # I think its already sorted, but just in case
    df.sort_values('Time', inplace=True) 
    
    #automatically determine if we are using neurokit or not by seeing if there is an 'r' column or a 'd' column
    use_neurokit = True if 'r' in df.columns else False

    if use_neurokit:
        if aligned_r_peak_pos is None:
            aligned_r_peak_pos = analyze_average_r_peak_position(args.all_ppg)
        print("Target R peak position: {}".format(aligned_r_peak_pos))
    else:
        aligned_r_peak_pos = 0
        
    # remove NaN
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column].fillna(pd.Timestamp('1900-01-01'), inplace=True)  # Or pd.NaT
        elif pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(0, inplace=True)
        else:
            df[column].fillna('Missing', inplace=True)


    filename = os.path.basename(args.ppg).split('.')[0]

    if not os.path.exists(args.out_folder): 
        os.mkdir(args.out_folder)
    out_dir = os.path.join(args.out_folder, filename)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create an empty dataframe to store the aligned R peaks, only keep the "glucose", "time" columns that are needed
    
    # print("Choose window generation method:")
    # print("1. 3-second windows")
    # print("2. By beat")
    # choice = input("Enter your choice (1 or 2): ")
    
    # if choice == '1':
    #     print("Generating 3-second windows...")
    #     sampling_rate = 192
        
    # elif choice == '2':
    #     sampling_rate = 64
    # else:
    #     raise ValueError("Invalid choice. Please enter 1 or 2.")
    
    # instead of above, automatically determine the sampling rate by checking the length of the first row, if it is over 190, then it is 192, otherwise it is 64
    sampling_rate = 192 if len(df.columns) > 190 else 64
    
    columns = list(np.arange(sampling_rate))
    if use_neurokit:
        columns.extend(['r', 'Time', 'glucose', 'flag', 'hypo_label'])
    else:
        columns.extend(['d', 'Time', 'glucose', 'flag', 'hypo_label'])

    aligned_rows = []
    # print("example of one of the rows in df before alignment: ", df.iloc[0])
    print("the shape of df: ", df.shape)
    print('type of df: ', type(df))
    print('is df empty? ', df.empty)
    print("the column names of df: ", df.columns.tolist())
    
    # iterate through each row in df
    if use_neurokit:
        for i in tqdm.tqdm(range(len(df))):
            row_data = df.iloc[i]
            aligned_row = align_ppg(row_data, aligned_r_peak_pos, sampling_rate=sampling_rate, use_neurokit=use_neurokit)
            aligned_rows.append(aligned_row)
    else:
        aligned_rows = df[columns].values.tolist()
    df_aligned = pd.DataFrame(aligned_rows, columns=columns)
    
    # Add 'hr' column to df_aligned
    if use_neurokit:
        df_aligned['hr'] = 60 / df['rr']
    else:
        df_aligned['hr'] = 60 / df['dd']
    
    # filter beats where the previous
    
    # remove zero entries from cgm
    df_aligned = df_aligned[df_aligned['glucose'] != 0]
    

    # save the aligned dataframe
    df_aligned.to_pickle(os.path.join(out_dir, "{}.pkl".format(filename)))

    # # pick a random row to show the alignment
    # random_row = np.random.randint(0, len(df_aligned))
    # show_aligned_demo(df.iloc[random_row], df_aligned.iloc[random_row])
    # plot_random_aligned_beats(df_aligned)
