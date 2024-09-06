import os
import time
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from sympy import use
from tqdm import tqdm
import cupy as cp
# import seaborn as sns

def align_ppg(row_data, target_r_peak, sampling_rate=64, use_neurokit=True):
    if use_neurokit:
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
    if use_neurokit:
        aligned_row.extend([aligned_r, aligned_time, aligned_glucose, aligned_flag, aligned_hypo_label])
    else:
        aligned_row.extend([row_data['d'], aligned_time, aligned_glucose, aligned_flag, aligned_hypo_label])

    return aligned_row

#     return bvp_cleaned, bvp_features
def load_checkfile(combined_df, out_path, regenerate=False, filter_rr=True, peak_choice="neurokit", window_size=3.0, overlap=0.5, cleaned_choice = "nkCleaned"):
    if peak_choice == "empatica":
        print("Initializing IBI_Presence column...")
        combined_df['IBI_Presence'] = False
        
        print("Computing previous diastolic points datetimes...")
        prev_ibi_datetimes = combined_df.index - pd.to_timedelta(combined_df['ibi'], unit='ms')
        
        print("Creating DataFrame of non-null previous IBI datetimes...")
        prev_ibi_df = pd.DataFrame({
            'prev_ibi_datetime': prev_ibi_datetimes[combined_df['ibi'].notnull()]
        }).drop_duplicates()
        
        print("Finding nearest index for each unique previous IBI datetime...")
        all_datetimes = combined_df.index
        nearest_indices = all_datetimes.get_indexer(prev_ibi_df['prev_ibi_datetime'], method='nearest')

        # Avoid duplicates by using a set
        print("Initializing marked indices set...")
        marked_indices = set()

        # Mark the IBI presence for unique previous IBI datetimes
        print("Marking IBI presence for unique previous IBI datetimes...")
        for i, prev_ibi_datetime in tqdm(enumerate(prev_ibi_df['prev_ibi_datetime']), total=len(prev_ibi_df)):
            closest_datetime_index = nearest_indices[i]
            if closest_datetime_index != -1 and closest_datetime_index not in marked_indices:
                # Mark the closest datetime as having an IBI presence
                combined_df.at[all_datetimes[closest_datetime_index], 'IBI_Presence'] = True
                marked_indices.add(closest_datetime_index)

        # Mark the IBI presence for current rows with non-null IBI
        combined_df.loc[combined_df['ibi'].notnull(), 'IBI_Presence'] = True
    
    combined_df = combined_df.dropna(subset=['bvp'])
    bvp_data = combined_df['bvp']
    bvp_data = np.asarray(bvp_data)
    idx =  np.arange(len(bvp_data))
    if cleaned_choice == "nkCleaned":
        bvp_cleaned = nk.ppg_clean(bvp_data, sampling_rate=64)
    else:
        bvp_cleaned = bvp_data
    
    # print(type(combined_df))
    
    # Now, use this IBI_Presence column as a marker for the rows that have a Diastolic point. Going to use this as an alternative method of beat extraction (with prefiltering).
    # add columns for the d_point similar to the r_peaks (time_d, ppd_d, idx_d)
    # add a 'dd' column to the combined_df to store the time difference between the diastolic points, similar to the 'rr' column
    # find the percentage of 'dd' intervals in range (0.3, 1.8) (33.3-200 bpm) and remove the ones that are not in this range just like the 'rr' intervals
    
    # Add columns for the d_point similar to the r_peaks (time_d, ppd_d, idx_d)
    
    if regenerate:
        arr_time = np.array(combined_df.index)
        if peak_choice == "neurokit":
            print("Generating r-peaks ~")
            start_time = time.time()
            _, info = nk.ppg_peaks(bvp_cleaned, sampling_rate=64, correct_artifacts=True)
            
            ppg_peaks = np.unique(info['PPG_Peaks'])
            print(f"Number of unique PPG_Peaks: {len(ppg_peaks)}")
            print("--- {:.3f} seconds ---".format(time.time() - start_time))
            
            time_r = []
            ppg_r = []
            idx_r = []
        

            for i in tqdm(range(len(ppg_peaks)), desc="Processing PPG peaks"):
                time_r.append(arr_time[ppg_peaks[i]])
                ppg_r.append(bvp_cleaned[ppg_peaks[i]])
                idx_r.append(idx[ppg_peaks[i]])
            
            #now for the diastolic points, a little different as the IBI_Presence is not a direct index and they are still in the combined_df
            #time_d is the index (Datetime) of the combined_df where the IBI_Presence is True
            #ppg_d is the value of the bvp_cleaned at the index (numerical, NOT datetime) of the combined_df where the IBI_Presence is True
            
            
            bvp_data = {
            'r_peaks':ppg_r,
            'time_r':time_r,
            'idx_r':idx_r
            }
            bvp_features = pd.DataFrame(bvp_data) #error, ValueError: All arrays must be of the same length (this is because the length of ppg_peaks and ppg_troughs are different)
            bvp_features.to_pickle(out_path)
        elif peak_choice == "empatica":
            print(f"number of unique IBI_Presence: {combined_df['IBI_Presence'].sum()}")
            #save ppg_troughs which is the numberical indexes of the the diastolic points as np array of indices similar to ppg_peaks. for reference, an example of PPG peaks looks like array([ 46,  89, 133, 176, 220, 261, 305, 350, 403, 437, 480, 559, 599, 631, 667, 705, 743, 777, 815, 853, 887, 926, 966])
            #know that the index of combined_df is datetime, so it needs to be converted to numerical index to be used in the bvp_cleaned array
            #currently, i have marked the diastolic points in the combined_df with the IBI_Presence column marked as a True value. Basically, trying to put the numerical index of where these True values are in the combined_df into the ppg_troughs array.
            
            #for example if the first row had a true value for IBI_Presence, then the first entry in ppg_troughs would be the numerical index of the first row in the combined_df (which is 0, as the first row is at index 0)
            #note, the type of combined_df is <class 'pandas.core.frame.DataFrame'>, and the index is currently a datetime index.
            ppg_troughs = np.where(combined_df['IBI_Presence'])[0]
            
            # sanity check, show me the first 10 values of ppg_peaks and ppg_troughs. the current problem is that File "/home/ugrads/d/david.roh/Sense_glucose/libs/helper.py", line 115, in load_checkfile time_d.append(arr_time[ppg_troughs[i]]) ~~~~~~~~~~~^^^ IndexError: index 142847 is out of bounds for axis 0 with size 142847
            print(f"First 10 values of ppg_troughs: {ppg_troughs[:10]}")
            
            time_d = []
            ppg_d = []
            idx_d = []
            
            for i in tqdm(range(len(ppg_troughs)), desc="Processing PPG troughs"):
                time_d.append(arr_time[ppg_troughs[i]])
                ppg_d.append(bvp_cleaned[ppg_troughs[i]])
                idx_d.append(idx[ppg_troughs[i]])
                
            bvp_data = {
            'd_points':ppg_d,
            'time_d':time_d,
            'idx_d':idx_d
            }
            bvp_features = pd.DataFrame(bvp_data) #error, ValueError: All arrays must be of the same length (this is because the length of ppg_peaks and ppg_troughs are different)
            bvp_features.to_pickle(out_path)
        elif peak_choice == "nopeak":
            print("Generating windows without peaks ~")
            window_size_samples = int(window_size * 64)
            overlap_samples = int(window_size_samples * overlap)
            window_starts = np.arange(0, len(bvp_cleaned) - window_size_samples + 1, overlap_samples)
            window_ends = window_starts + window_size_samples

            # Get the timestamps of the window starts and ends
            window_start_times = arr_time[window_starts]
            window_end_times = arr_time[window_ends]

            bvp_data = {
                'window_start': window_starts,
                'window_end': window_ends,
                'window_start_time': window_start_times,
                'window_end_time': window_end_times
            }
            bvp_features = pd.DataFrame(bvp_data)
            bvp_features.to_pickle(out_path)
        
        
    bvp_features = pd.read_pickle(out_path)

    if peak_choice == "neurokit" or peak_choice == "empatica":
        column_name = 'dd' if peak_choice == "empatica" else 'rr'

        # Calculate interval
        bvp_features[column_name] = bvp_features[f'time_{column_name[0]}'].shift(-1) - bvp_features[f'time_{column_name[0]}']
        bvp_features[column_name] = bvp_features[column_name].dt.total_seconds()

        # Drop NA values
        bvp_features = bvp_features.dropna()

        # Filter intervals in range (0.3, 1.5)
        interval_in_range = (bvp_features[column_name] > 0.3) & (bvp_features[column_name] < 1.5)
        percentage_in_range = interval_in_range.mean()

        print(f"Percentage of '{column_name}' intervals in range (0.3, 1.5) (40-200 bpm): {percentage_in_range * 100}%")

        # Filter DataFrame
        bvp_features = bvp_features[interval_in_range]›

        # Reset index
        bvp_features = bvp_features.reset_index(drop=True)
    
    if filter_rr:
        if peak_choice == "neurokit":
            # now, calculate the rr based on the one before. if this one is more than 10% different from the one before, then we drop it as it is too noisy and the hr variablity is too high. Else, we update the rr to be the average of the two.
            # Recalculate the 'rr' column
            bvp_features['new_rr'] = bvp_features['time_r'] - bvp_features['time_r'].shift(1)
            bvp_features['new_rr'] = bvp_features['new_rr'].dt.total_seconds()

            # Calculate the difference between the new 'rr' and the old one
            diff = bvp_features['new_rr'] - bvp_features['rr'].shift(1)

            # Set 'rr' to NaN where the difference is more than 10% of the old 'rr'
            bvp_features.loc[abs(diff) > 0.1 * bvp_features['rr'].shift(1), 'rr'] = np.nan

            # Update 'rr' to be the average of the new 'rr' and the old one where 'rr' is not NaN
            mask = bvp_features['rr'].notna()
            bvp_features.loc[mask, 'rr'] = (bvp_features.loc[mask, 'new_rr'] + bvp_features.loc[mask, 'rr']) / 2
        elif peak_choice == "empatica":
            # for with 'dd'
            bvp_features['new_dd'] = bvp_features['time_d'] - bvp_features['time_d'].shift(1)
            bvp_features['new_dd'] = bvp_features['new_dd'].dt.total_seconds()
            
            # Calculate the difference between the new 'dd' and the old one
            diff = bvp_features['new_dd'] - bvp_features['dd'].shift(1)
            
            # Set 'dd' to NaN where the difference is more than 10% of the old 'dd'
            bvp_features.loc[abs(diff) > 0.1 * bvp_features['dd'].shift(1), 'dd'] = np.nan
            
            # Update 'dd' to be the average of the new 'dd' and the old one where 'dd' is not NaN
            mask = bvp_features['dd'].notna()
            bvp_features.loc[mask, 'dd'] = (bvp_features.loc[mask, 'new_dd'] + bvp_features.loc[mask, 'dd']) / 2
            
        bvp_features.dropna(inplace=True)
        bvp_features.reset_index(drop=True, inplace=True)
    
    # plt.figure(figsize=(10, 6))
    # sns.histplot(bvp_features['rr'], bins=30, kde=True)
    # plt.title('Distribution of RR times')
    # plt.xlabel('RR times')
    # plt.ylabel('Frequency')
    # plt.savefig('rr_distribution.png')

    return bvp_cleaned, bvp_features

# def generate_beats(ppg, df, sampling_rate = 64, use_neurokit=True):
#     if use_neurokit:
#         allbeats = []
#         beat =[]  #l_temp
#         rtime_list = []  #Times
#         rpeak_idx = []    #Rs
#         start_idx = []
#         rr_sec = []
#         samp_from_start = []

#         b = 3.5  ##this value should be more default=3   #[2,3,4]
#         a = 1.5   ##this value should be less default=2

#         for i in range(df.shape[0]):
#             num_samples_before = int(np.ceil((df.at[i,'rr']/b)*sampling_rate))
#             num_samples_after = int(np.ceil((df.at[i,'rr']/a)*sampling_rate))
#             start = df.at[i,'idx_r'] - num_samples_before
#             end = df.at[i,'idx_r'] + num_samples_after
            
#             # # Print the types and values of the variables involved in the creation of 'start'
#             # print(f"Type of df.at[i,'idx_r']: {type(df.at[i,'idx_r'])}, value: {df.at[i,'idx_r']}")
#             # print(f"Type of num_samples_before: {type(num_samples_before)}, value: {num_samples_before}")
#             # print(f"Type of start: {type(start)}, value: {start}")
            
#             beat.append(ppg[start:end])
#             rtime_list.append(df.at[i,'time_r'])
#             start_idx.append(start)
#             rr_sec.append(df.at[i,'rr'])
#             rpeak_idx.append(df.at[i,'idx_r'])
#             samp_from_start.append(num_samples_before)
#             # print(f"After iteration {i}, beat list size is {len(beat)}")
#         beats = pd.DataFrame(beat) 
#         beats['start_idx'] = start_idx
#         beats['start_samp'] = samp_from_start
#         beats['rpeak_idx'] = rpeak_idx
#         beats['rr'] = rr_sec
#         beats['Time'] = rtime_list
#         #beats['rpk_index'] = rpks
#         allbeats.append(beats)
#         df_ppg_final = pd.concat(allbeats).reset_index(drop = True)
#         if 'Time' in df_ppg_final.columns:
#             print('Time is present')

#         df_ppg_final['r'] = df_ppg_final['rpeak_idx'] - df_ppg_final['start_idx']

#         return df_ppg_final
#     else:
#         allbeats = []
#         beat =[]  #l_temp
#         dtime_list = []  #Times
#         dpeak_idx = []    #Ds
#         start_idx = []
#         dd_sec = []
#         samp_from_start = []

#         for i in range(df.shape[0]):
#             start = df.at[i,'idx_d']
#             end = df.at[i+1,'idx_d'] if i+1 < df.shape[0] else df.shape[0]
            
#             beat.append(ppg[start:end])
#             dtime_list.append(df.at[i,'time_d'])
#             start_idx.append(start)
#             dd_sec.append(df.at[i,'dd'])
#             dpeak_idx.append(df.at[i,'idx_d'])
#             samp_from_start.append(0)
#         beats = pd.DataFrame(beat) 
#         beats['start_idx'] = start_idx
#         beats['start_samp'] = samp_from_start
#         beats['dpeak_idx'] = dpeak_idx
#         beats['dd'] = dd_sec
#         beats['Time'] = dtime_list
#         allbeats.append(beats)
#         df_ppg_final = pd.concat(allbeats).reset_index(drop = True)

#         if 'Time' in df_ppg_final.columns:
#             print('Time is present')

#         df_ppg_final['d'] = df_ppg_final['dpeak_idx'] - df_ppg_final['start_idx']

#         return df_ppg_final

def generate_windows(ppg, df, sampling_rate = 64, window_size = 3.0, peak_choice="neurokit", overlap=0.5):
    def process_windows(df, ppg, peak_type):
        allbeats = []
        beat =[]  #l_temp
        time_list = []  #Times
        peak_idx = []    #Peaks
        start_idx = []
        interval_sec = []
        samp_from_start = []

        half_window = window_size / 2
        num_samples_before = int(np.ceil(half_window * sampling_rate))  # 1.5 seconds before Peak
        num_samples_after = int(np.ceil(half_window * sampling_rate))  # 1.5 seconds after Peak

        for i in tqdm(range(df.shape[0]), desc="Processing windows"):
            start = df.at[i,f'idx_{peak_type}'] - num_samples_before
            end = df.at[i,f'idx_{peak_type}'] + num_samples_after

            beat.append(ppg[start:end])
            time_list.append(df.at[i,f'time_{peak_type}'])
            start_idx.append(start)
            interval_sec.append(df.at[i,f'{peak_type}{peak_type}'])
            peak_idx.append(df.at[i,f'idx_{peak_type}'])
            samp_from_start.append(num_samples_before)

        beats = pd.DataFrame(beat) 
        beats['start_idx'] = start_idx
        beats['start_samp'] = samp_from_start
        beats[f'{peak_type}peak_idx'] = peak_idx
        beats[f'{peak_type}{peak_type}'] = interval_sec
        beats['Time'] = time_list
        allbeats.append(beats)
        df_ppg_final = pd.concat(allbeats).reset_index(drop = True)

        if 'Time' in df_ppg_final.columns:
            print('Time is present')

        df_ppg_final[peak_type] = df_ppg_final[f'{peak_type}peak_idx'] - df_ppg_final['start_idx']

        return df_ppg_final

    def process_overlapping_windows(ppg, df, window_size, overlap):
        '''
        
        ppg = ppg_cleaned, which is just a numpy array of the cleaned ppg signal
        df = bvp_features, which is the dataframe of:
            window_start: This column contains the start indices of the windows in the original BVP data.
            window_end: This column contains the end indices of the windows in the original BVP data.
            window_start_time: This column contains the timestamps corresponding to the start of each window.
            window_end_time: This column contains the timestamps corresponding to the end of each window.
        
        window_size = 3.0 (seconds) that i should multiply by the sampling rate to get the number of samples per window
        overlap = 0.5 (50% overlap)
        '''
        all_windows = []
        window_start_times = []
        window_end_times = []
        window_start_indices = []
        window_end_indices = []

        for i in tqdm(range(df.shape[0]), desc="Processing overlapping windows"):
            start = df.at[i, 'window_start']
            end = df.at[i, 'window_end']
            window = ppg[start:end]
            all_windows.append(window)
            window_start_times.append(df.at[i, 'window_start_time'])
            window_end_times.append(df.at[i, 'window_end_time'])
            window_start_indices.append(start)
            window_end_indices.append(end)

        windows_df = pd.DataFrame(all_windows)
        windows_df['start_idx'] = window_start_indices
        windows_df['end_idx'] = window_end_indices
        windows_df['window_start_time'] = window_start_times
        windows_df['window_end_time'] = window_end_times
        
        windows_df['Time'] = windows_df['window_start_time'] + (windows_df['window_end_time'] - windows_df['window_start_time']) / 2

        return windows_df

    if peak_choice == "neurokit":
        return process_windows(df, ppg, 'r')
    elif peak_choice == "empatica":
        return process_windows(df, ppg, 'd')
    else:
        return process_overlapping_windows(ppg, df, window_size, overlap)

def generate_fft_windows_gpu(ppg, df, sampling_rate=64, window_size=3.0, overlap=0.5):
    all_windows = []
    window_start_times = []
    window_end_times = []
    window_start_indices = []
    window_end_indices = []
    
    # Transfer PPG data to GPU
    ppg_gpu = cp.array(ppg)

    for i in tqdm(range(df.shape[0]), desc="Processing FFT windows"):
        start = int(df.at[i, 'window_start'])
        end = int(df.at[i, 'window_end'])
        
        # Slicing and FFT on GPU
        window = ppg_gpu[start:end]
        fft = cp.abs(cp.fft.rfft(window))
        freqs = cp.fft.rfftfreq(len(window), 1/sampling_rate)
        
        # Filter frequencies and move data back to CPU for DataFrame storage
        valid_indices = cp.where(freqs <= 32)[0]
        fft = fft[valid_indices].get()
        freqs = freqs[valid_indices].get()
        
        all_windows.append(fft)
        window_start_times.append(df.at[i, 'window_start_time'])
        window_end_times.append(df.at[i, 'window_end_time'])
        window_start_indices.append(start)
        window_end_indices.append(end)

    # Create DataFrame
    windows_df = pd.DataFrame(all_windows)
    windows_df.columns = [f"Magnitude_{freq:.3f}Hz" for freq in freqs]
    windows_df['start_idx'] = window_start_indices
    windows_df['end_idx'] = window_end_indices
    windows_df['window_start_time'] = window_start_times
    windows_df['window_end_time'] = window_end_times
    windows_df['Time'] = windows_df['window_start_time'] + (windows_df['window_end_time'] - windows_df['window_start_time']) / 2

    return windows_df

def generate_wavelet_windows(ppg, df, sampling_rate=64, window_size=3.0, overlap=0.5):
    all_windows = []
    window_start_times = []
    window_end_times = []
    window_start_indices = []
    window_end_indices = []
    
    # Transfer PPG data to GPU
    ppg_gpu = cp.array(ppg)

    for i in tqdm(range(df.shape[0]), desc="Processing FFT windows"):
        start = int(df.at[i, 'window_start'])
        end = int(df.at[i, 'window_end'])
        
        # Slicing and FFT on GPU
        window = ppg_gpu[start:end]
        fft = cp.abs(cp.fft.rfft(window))
        freqs = cp.fft.rfftfreq(len(window), 1/sampling_rate)
        
        # Filter frequencies and move data back to CPU for DataFrame storage
        valid_indices = cp.where(freqs <= 32)[0]
        fft = fft[valid_indices].get()
        freqs = freqs[valid_indices].get()
        
        all_windows.append(fft)
        window_start_times.append(df.at[i, 'window_start_time'])
        window_end_times.append(df.at[i, 'window_end_time'])
        window_start_indices.append(start)
        window_end_indices.append(end)

    # Create DataFrame
    windows_df = pd.DataFrame(all_windows)
    windows_df.columns = [f"Magnitude_{freq:.3f}Hz" for freq in freqs]
    windows_df['start_idx'] = window_start_indices
    windows_df['end_idx'] = window_end_indices
    windows_df['window_start_time'] = window_start_times
    windows_df['window_end_time'] = window_end_times
    windows_df['Time'] = windows_df['window_start_time'] + (windows_df['window_end_time'] - windows_df['window_start_time']) / 2

    return windows_df

def generate_beats(ppg, df, sampling_rate = 64, use_neurokit=True):
    def process_beats(df, ppg, peak_type):
        allbeats = []
        beat =[]  #l_temp
        time_list = []  #Times
        peak_idx = []    #Peaks
        start_idx = []
        interval_sec = []
        samp_from_start = []

        for i in tqdm(range(df.shape[0]), desc="Processing beat segmentation"):
            if peak_type == 'r':
                num_samples_before = int(np.ceil((df.at[i,'rr']/3.5)*sampling_rate))
                num_samples_after = int(np.ceil((df.at[i,'rr']/1.5)*sampling_rate))
                start = df.at[i,'idx_r'] - num_samples_before
                end = df.at[i,'idx_r'] + num_samples_after
                samp_from_start.append(num_samples_before)
            else:
                start = df.at[i,'idx_d']
                end = df.at[i+1,'idx_d'] if i+1 < df.shape[0] else df.shape[0]
                samp_from_start.append(0)

            beat.append(ppg[start:end])
            time_list.append(df.at[i,f'time_{peak_type}'])
            start_idx.append(start)
            interval_sec.append(df.at[i,f'{peak_type}{peak_type}'])
            peak_idx.append(df.at[i,f'idx_{peak_type}'])

        beats = pd.DataFrame(beat) 
        beats['start_idx'] = start_idx
        beats['start_samp'] = samp_from_start
        beats[f'{peak_type}peak_idx'] = peak_idx
        beats[f'{peak_type}{peak_type}'] = interval_sec
        beats['Time'] = time_list
        allbeats.append(beats)
        df_ppg_final = pd.concat(allbeats).reset_index(drop = True)

        if 'Time' in df_ppg_final.columns:
            print('Time is present')

        df_ppg_final[peak_type] = df_ppg_final[f'{peak_type}peak_idx'] - df_ppg_final['start_idx']

        return df_ppg_final

    if use_neurokit:
        return process_beats(df, ppg, 'r')
    else:
        return process_beats(df, ppg, 'd')
    
def plot_beat(beats, row_index, out_path, num_samples=52, use_neurokit=True):
    # Plot the PPG beat
    print("Show sample PPG beat ...")
    if use_neurokit:
        # Get the values from the selected row
        valr = beats.at[row_index,'r']
        print("Row: {}, r: {}".format(row_index, valr))
    else:
        vald = beats.at[row_index,'d']
        print("Row: {}, d: {}".format(row_index, vald))
    plt.plot(beats.iloc[row_index,:num_samples])
    if use_neurokit and (valr >= 0 and valr < num_samples):
        plt.scatter(valr, beats.at[row_index,valr], c='red', s=20, label='r')
    elif not use_neurokit and (vald >= 0 and vald < num_samples):
        plt.scatter(vald, beats.at[row_index,vald], c='red', s=20, label='d')
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


def combine_with_summary_glucose(ppg_df_final, combined_df, glucose_df, window_size=3.0, peak_choice="neurokit", representation_choice="time"):

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
    conditions = [(final_df['glucose'] > hyperthresh), (final_df['glucose'] <= hyperthresh)]
    values = [1, 0]
    final_df['hyper_label'] = np.select(conditions, values)

    # find the index of "start_idx" column
    start_idx_col = list(final_df.columns).index('start_idx')
    if representation_choice == "time":
        num_samples = window_size * 64
        final_df = pd.concat([final_df.iloc[:, :num_samples], final_df.iloc[:, start_idx_col:]], axis=1)
    elif representation_choice == "fft":
        #instead of num_samples depending on the window size, we will have to depend on the number of frequencies that we have in the fft
        num_samples = int(((window_size * 64) / 2) + 1)
        final_df = pd.concat([final_df.iloc[:, :num_samples], final_df.iloc[:, start_idx_col:]], axis=1)
        

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

def clean_positive_values(final_df, verbose=True):
    '''
    similar to that of clean_negative_values, but this time, we are looking for positive values (for diastolic points)
    '''
    num_rows = final_df.shape[0]
    rows_with_positive_d = (final_df['d'] > 0).sum()
    final_df = final_df[final_df['d'] <= 0]
    final_df = final_df.sort_values('Time')
    final_df = final_df.reset_index(drop=True)
    if verbose:
        print("Total number of rows removed (if d is greater than 0): {}".format(num_rows - final_df.shape[0]))
        print(" - rows with positive values in column 'd': {}".format(rows_with_positive_d))
        
    return final_df