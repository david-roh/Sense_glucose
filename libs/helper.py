import os
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
# import seaborn as sns

def align_ppg(row_data, target_r_peak, sampling_rate=64):
    r_peak = row_data['r']
    displacement = target_r_peak - r_peak

    ppg_data = np.array(row_data[:sampling_rate])
    aligned_ppg_data = np.roll(ppg_data, displacement)
    if displacement > 0:
        aligned_ppg_data[:displacement] = 0
    elif displacement < 0:
        aligned_ppg_data[displacement:] = 0

    aligned_r = int(row_data['r'] + displacement)
    aligned_time = row_data['Time']
    aligned_glucose = row_data['glucose']
    aligned_flag = row_data['flag']
    aligned_hypo_label = row_data['hypo_label']

    # append the aligned data to the dataframe
    aligned_row = aligned_ppg_data.tolist()
    aligned_row.extend([aligned_r, aligned_time, aligned_glucose, aligned_flag, aligned_hypo_label])

    return aligned_row

#     return bvp_cleaned, bvp_features
def load_checkfile(combined_df, out_path, regenerate=False):
    combined_df = combined_df.dropna(subset=['bvp'])
    bvp_data = combined_df['bvp']
    bvp_data = np.asarray(bvp_data)
    idx =  np.arange(len(bvp_data))
    bvp_cleaned = nk.ppg_clean(bvp_data, sampling_rate=64)

    if regenerate:
        print("Generating r-peaks ~")
        start_time = time.time()
        _, info = nk.ppg_peaks(bvp_cleaned, sampling_rate=64, correct_artifacts=True)
        
        ppg_peaks = np.unique(info['PPG_Peaks'])
        print(f"Number of unique PPG_Peaks: {len(ppg_peaks)}")
        arr_time = np.array(combined_df.index)
        print("--- {:.3f} seconds ---".format(time.time() - start_time))
        
        time_r = []
        ppg_r = []
        idx_r = []
        
        for i in range(len(ppg_peaks)):
            time_r.append(arr_time[ppg_peaks[i]])
            ppg_r.append(bvp_cleaned[ppg_peaks[i]])
            idx_r.append(idx[ppg_peaks[i]])
        bvp_data = {'r_peaks':ppg_r,
        'time_r':time_r,
        'idx_r':idx_r}
        bvp_features = pd.DataFrame(bvp_data)
        bvp_features.to_pickle(out_path)

    bvp_features = pd.read_pickle(out_path)

    # calculate rr interval
    bvp_features['rr'] = bvp_features['time_r'].shift(-1) - bvp_features['time_r']
    bvp_features['rr'] = bvp_features['rr'].dt.total_seconds()

    bvp_features = bvp_features.dropna()
    
    percentage_in_range = ((bvp_features['rr'] > 0.3) & (bvp_features['rr'] < 1.8)).mean()
    print(f"Percentage of 'rr' intervals in range (0.3, 1.8) (33.3-200 bpm): {percentage_in_range * 100}%")
    
    
    # removing gaps and noise
    bvp_features = bvp_features[(bvp_features['rr'] > 0.3) & (bvp_features['rr'] < 1.8)]
    bvp_features = bvp_features.reset_index(drop=True)
    
    # plt.figure(figsize=(10, 6))
    # sns.histplot(bvp_features['rr'], bins=30, kde=True)
    # plt.title('Distribution of RR times')
    # plt.xlabel('RR times')
    # plt.ylabel('Frequency')
    # plt.savefig('rr_distribution.png')

    return bvp_cleaned, bvp_features

def generate_beats(ppg, df, sampling_rate = 64):
    allbeats = []
    beat =[]  #l_temp
    rtime_list = []  #Times
    rpeak_idx = []    #Rs
    start_idx = []
    rr_sec = []
    samp_from_start = []

    b = 3.5  ##this value should be more default=3   #[2,3,4]
    a = 1.5   ##this value should be less default=2

    for i in range(df.shape[0]):
        num_samples_before = int(np.ceil((df.at[i,'rr']/b)*sampling_rate))
        num_samples_after = int(np.ceil((df.at[i,'rr']/a)*sampling_rate))
        start = df.at[i,'idx_r'] - num_samples_before
        end = df.at[i,'idx_r'] + num_samples_after
        
        # # Print the types and values of the variables involved in the creation of 'start'
        # print(f"Type of df.at[i,'idx_r']: {type(df.at[i,'idx_r'])}, value: {df.at[i,'idx_r']}")
        # print(f"Type of num_samples_before: {type(num_samples_before)}, value: {num_samples_before}")
        # print(f"Type of start: {type(start)}, value: {start}")
        
        beat.append(ppg[start:end])
        rtime_list.append(df.at[i,'time_r'])
        start_idx.append(start)
        rr_sec.append(df.at[i,'rr'])
        rpeak_idx.append(df.at[i,'idx_r'])
        samp_from_start.append(num_samples_before)
        # print(f"After iteration {i}, beat list size is {len(beat)}")
    beats = pd.DataFrame(beat) 
    beats['start_idx'] = start_idx
    beats['start_samp'] = samp_from_start
    beats['rpeak_idx'] = rpeak_idx
    beats['rr'] = rr_sec
    beats['Time'] = rtime_list
    #beats['rpk_index'] = rpks
    allbeats.append(beats)
    df_ppg_final = pd.concat(allbeats).reset_index(drop = True)
    if 'Time' in df_ppg_final.columns:
        print('Time is present')

    df_ppg_final['r'] = df_ppg_final['rpeak_idx'] - df_ppg_final['start_idx']

    return df_ppg_final

def plot_beat(beats, row_index, out_path, num_samples=52):
    # Plot the PPG beat
    print("Show sample PPG beat ...")
    # Get the values from the selected row
    valr = beats.at[row_index,'r']
    print("Row: {}, r: {}".format(row_index, valr))
    # Plot line chart for the selected row
    color_dict = dict({'p':'red', 'q':'blue', 'r':'green', 's':'orange', 't':'purple'})
    plt.plot(beats.iloc[row_index,:num_samples])
    if valr >= 0 and valr < num_samples:
        plt.scatter(valr, beats.at[row_index,valr], c=color_dict['r'], s=20, label='r')
    plt.legend()
    plt.title("Sample PPG beat: Row {}".format(row_index))
    plt.savefig(out_path)

def plot_alignment_beats(beats, row_indices, out_path, num_samples=52):
    # Plot the PPG beat
    print("Show sample PPG beat ...")
    # create color dictionary
    color_dict = dict({'p':'red', 'q':'blue', 'r':'green', 's':'orange', 't':'purple'})
    valrs = []
    for row_index in row_indices:
        # Get the values from the selected row
        valr = beats.at[row_index,'r']
        print("Row: {}, r: {}".format(row_index, valr))
        # Plot line chart for the selected row
        plt.plot(beats.iloc[row_index,:num_samples])
        if valr >= 0 and valr < num_samples:
            valrs.append([valr, beats.at[row_index,valr]])

    valrs = np.array(valrs)

    plt.scatter(valrs[:,0], valrs[:,1], c=color_dict['r'], s=20, label='r')

    plt.legend()
    plt.grid(True)   
    plt.title("Sample Aligned PPG beats")
    plt.savefig(out_path)


def combine_with_summary_glucose(ppg_df_final, combined_df, glucose_df):

    # combined_df['Time']= pd.to_datetime(combined_df['Time'], format='%d/%m/%Y %H:%M:%S.%f' )
    # df_combined = pd.merge_asof(ppg_df_final.sort_values('Time'),combined_df.sort_values('Time'),on='Time',tolerance=pd.Timedelta('1 sec'),direction='nearest',allow_exact_matches=True)
    # df_combined = df_combined.sort_values(by='Time', ascending=True)
    
    # # Reset the index of combined_df and rename the new column to 'Time'
    # combined_df = combined_df.reset_index().rename(columns={'index': 'Time'})
    combined_df['Time'] = combined_df.index

    df_combined = pd.merge_asof(ppg_df_final.sort_values('Time'),combined_df.sort_values('Time'),on='Time',tolerance=pd.Timedelta('1 sec'),direction='nearest',allow_exact_matches=True)
    df_combined = df_combined.sort_values(by='Time', ascending=True)

    # glucose_df.columns = ['time', 'glucose']
    # glucose_df['Time'] = pd.to_datetime(glucose_df['time'],format='%Y-%m-%d %H:%M:%S')
    glucose_df = glucose_df.rename(columns={'Timestamp': 'Time'})
    glucose_df['flag'] = np.arange(1, glucose_df.shape[0]+1)

    final_df = pd.merge_asof(df_combined.sort_values('Time'), glucose_df.sort_values('Time'), on='Time', tolerance=pd.Timedelta('330s'), direction='forward', allow_exact_matches=True)

    ### creating labels 
    hypothresh = 70
    hyperthresh = 180
    ### Adding hypo labels
    conditions = [(final_df['glucose'] < hypothresh),(final_df['glucose'] >= hypothresh) ]
    values = [1,0]
    final_df['hypo_label'] = np.select(conditions, values)
    conditions = [(final_df['glucose'] <= hyperthresh),(final_df['glucose'] > hyperthresh) ]
    values = [1,0]
    final_df['hypo_flag'] = np.select(conditions, values)

    # find the index of "start_idx" column
    start_idx_col = list(final_df.columns).index('start_idx')
    final_df = pd.concat([final_df.iloc[:, :64], final_df.iloc[:, start_idx_col:]], axis=1)

    return final_df

def clean_negative_values(final_df, verbose=True):
    num_rows = final_df.shape[0]    
    # Check how many rows have negative values 
    rows_with_negative_r = (final_df['r'] < 0).sum()

    final_df = final_df[final_df['r'] >= 0]
    final_df = final_df.sort_values('Time')
    final_df = final_df.reset_index(drop=True)
    # Display the results
    if verbose:
        print("Total number of rows removed (if r is smaller than 0): {}".format(num_rows - final_df.shape[0]))
        print(" - rows with negative values in column 'r': {}".format(rows_with_negative_r))

    return final_df