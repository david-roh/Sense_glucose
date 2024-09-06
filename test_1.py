
# %%
from curses import window
from turtle import st
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from pyparsing import col
from tqdm import tqdm
tqdm.pandas()

def plot_fft(path, offset, cohort, subject, cutoff=32, cleaned="nkCleaned"):
    with open(path, 'rb') as f:
        data = pickle.load(f) #dataframe

        
    #save the window_start_time of the first row to use as reference for offset
    data_start_time = data.iloc[0, data.columns.to_list().index('window_start_time')]
    print("data_start_time", data_start_time)
    #type of data_start_time <class 'pandas._libs.tslibs.timestamps.Timestamp'>
    
    #find row_num to cut based on offset (number of seconds forward from the first window_start)
    #basically, add offset number of seconds to data_start_time and find the row that has that window_start_time, then save the row number
    target_time = data_start_time + timedelta(seconds=offset)
    print("target_time", target_time)
    row_num = data[data['Time'] >= target_time].index[0]
    print("row_num", row_num)
    print("final_time", data.iloc[row_num, data.columns.to_list().index('Time')])
    
    
    # row_num = 0
    start_idx = data.columns.to_list().index('start_idx')
    # Calculate the number of columns per Hz
    columns_per_hz = start_idx / 32
    
    row = data.iloc[row_num, :start_idx]
    # get the number of seconds in the row
    seconds_row = ((len(row) - 1) * 2) / 64.0
    if seconds_row.is_integer():
        seconds_row = int(seconds_row)
    # Calculate the number of columns for the cutoff frequency
    cutoff_columns = int(cutoff * columns_per_hz)
    row = data.iloc[row_num, :cutoff_columns]
    
    #keep only the first cutoff frequencies, can calculate from start_idx, which is the number of frequencies (starts as 0-32hz)
    

    # Determine the magnitude with the highest number in row
    max_magnitude = row.max()
    # Now, find the column name of the max_magnitude (idxmax doesn't work because the column names are strings)
    max_magnitude_column = row.index[np.argmax(row)]

    # Set the font size
    plt.rcParams.update({'font.size': 14})

    # Create the figure with a 4:3 ratio
    plt.figure(figsize=(8, 6))

    # Plot the row fft
    plt.plot(row)

    # Calculate the step size in terms of indices
    step_size = len(row) / cutoff

    # Add a marker at the frequency with the maximum amplitude
    max_freq = float(max_magnitude_column.replace('Magnitude_', '').replace('Hz', ''))
    plt.scatter(max_freq * step_size, max_magnitude, color='red')  # Add a red dot

    y_offset = 1000
    x_offset = 10

    if seconds_row < 4: #originally checked if == 3
        y_offset = 100
        x_offset = 1
    elif seconds_row >= 4: #originally checked if == 9
        y_offset = 300
        x_offset = 3
    # Add a label for the frequency with the maximum amplitude
    plt.text(max_freq * step_size + x_offset, max_magnitude - y_offset, f'{max_freq} Hz', ha='left')

    # Generate the xticks and labels
    xticks = np.linspace(0, len(row), num=cutoff//2+1)
    xticklabels = np.arange(0, cutoff+1, 2)  # Labels from 0 to 32, step 2

    plt.xticks(xticks, xticklabels)


    yticks, _ = plt.yticks()
    plt.yticks(yticks, [f'{int(yt/1000)}' for yt in yticks])
    plt.ylabel('Amplitude (x 10^3)')
    plt.ylim(-y_offset, np.ceil(max_magnitude / (y_offset*3)) * (y_offset*3))
    
    

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (x 10^3)')
    plt.title(f'FFT Frequencies ({seconds_row} sec window, c{cohort}s0{subject})')
    plt.show()
    plt.savefig(f'{out_path}/c{cohort}s0{subject}_{seconds_row}sec_fft_{cleaned}_row{row_num}.png')
    
#%%
def plot_time(path, offset, cohort, subject, cleaned="nkCleaned"):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    #save the window_start_time of the first row to use as reference for offset
    data_start_time = data.iloc[0, data.columns.to_list().index('window_start_time')]
    print("data_start_time", data_start_time)
    #type of data_start_time <class 'pandas._libs.tslibs.timestamps.Timestamp'>
    
    #find row_num to cut based on offset (number of seconds forward from the first window_start)
    #basically, add offset number of seconds to data_start_time and find the row that has that window_start_time, then save the row number
    target_time = data_start_time + timedelta(seconds=offset)
    print("target_time", target_time)
    row_num = data[data['Time'] >= target_time].index[0]
    print("row_num", row_num)
    print("final_time", data.iloc[row_num, data.columns.to_list().index('Time')])    
    
    start_idx = data.columns.to_list().index('start_idx')
    row = data.iloc[row_num, :start_idx]

    # Set the font size
    plt.rcParams.update({'font.size': 14})

    # Create the figure with a 4:3 ratio
    plt.figure(figsize=(8, 6))

    # number of seconds
    seconds_row = len(row) / 64.0
    if seconds_row.is_integer():
        seconds_row = int(seconds_row)

    # Create an array for the x-axis that represents time in seconds
    time = np.arange(0, seconds_row, 1/64)

    # Plot the row fft with time on the x-axis
    plt.plot(time, row)

    # Adjust the y-axis limits
    plt.ylim(-200, 200)

    # Adjust the x-axis ticks to go by 1s
    plt.xticks(np.arange(0, np.ceil(seconds_row), 1))

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Time Series ({seconds_row} sec window, c{cohort}s0{subject})')
    plt.show()
    #save to out_path
    plt.savefig(f'{out_path}/c{cohort}s0{subject}_{seconds_row}sec_time_series_{cleaned}_row{row_num}.png')
#%%
def plot_stack(path, cohort, subject, offset=0, num_windows=None, cleaned="nkCleaned", start_idx_col_name='start_idx', normalize=True):
    with open(path, 'rb') as f:
        data = pickle.load(f)  # dataframe

    # Identify the column index for 'start_idx'
    start_idx = data.columns.get_loc(start_idx_col_name)
    
    # Extract magnitude columns and 'Time' column
    magnitude_data = data.iloc[:, :start_idx]
    times = data['Time']
    
    # Apply offset if specified
    if offset > 0 and offset < len(times):
        magnitude_data = magnitude_data.iloc[offset:, :]
        times = times.iloc[offset:]
    
    # If num_windows is specified, select the first 'num_windows' entries after applying the offset
    if num_windows is not None and num_windows < len(times):
        magnitude_data = magnitude_data.iloc[:num_windows, :]
        times = times.iloc[:num_windows]
    
    # Normalize the magnitude data for better color visualization
    if normalize:
        norm_magnitude_data = (magnitude_data - magnitude_data.min()) / (magnitude_data.max() - magnitude_data.min())
    else:
        norm_magnitude_data = magnitude_data
    
    # Set the font size
    plt.rcParams.update({'font.size': 14})

    # Create the figure with a 4:3 ratio
    plt.figure(figsize=(8, 6))
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(norm_magnitude_data, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0, magnitude_data.shape[1], times.iloc[0], times.iloc[-1]])
    
    ax.set_xlabel('Sample Index within Window')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Stacked PPG Signal Plot (c{cohort}s0{subject})')
    
    # Add a color bar to show the magnitude scale
    cbar = fig.colorbar(cax)
    if normalize:
        cbar.set_label('Normalized Magnitude')
    else:
        cbar.set_label('Magnitude')
    
    plt.show()



    

sampling_rate = 64

cohort = 2
subject = 2

offset = 135 # 135 seconds after the first window_start_time

cleaned = "nkCleaned"

#path to the pickle file
path27 = f'/mnt/data2/david/data/TCH_processed/c{cohort}s0{subject}_27sec__nopeak_fft_{cleaned}/c{cohort}s0{subject}.pkl'
path9 = f'/mnt/data2/david/data/TCH_processed/c{cohort}s0{subject}_9sec__nopeak_fft_{cleaned}/c{cohort}s0{subject}.pkl'
path3 = f'/mnt/data2/david/data/TCH_processed/c{cohort}s0{subject}_3sec__nopeak_fft_{cleaned}/c{cohort}s0{subject}.pkl'
path27_time = f'/mnt/data2/david/data/TCH_processed/c{cohort}s0{subject}_27sec__nopeak_time_{cleaned}/c{cohort}s0{subject}.pkl'
path9_time = f'/mnt/data2/david/data/TCH_processed/c{cohort}s0{subject}_9sec__nopeak_time_{cleaned}/c{cohort}s0{subject}.pkl'
path3_time = f'/mnt/data2/david/data/TCH_processed/c{cohort}s0{subject}_3sec__nopeak_time_{cleaned}/c{cohort}s0{subject}.pkl'

testing_c2s02 = f'/mnt/data2/david/data/c_0{cohort}/s_0{subject}/e4/combined_e4.pkl'


out_path = f'/mnt/data2/david/data/TCH_processed/c{cohort}s0{subject}_plots'

if not os.path.exists(out_path):
    os.makedirs(out_path)

variations = ["27", "9", "3", "27_time", "9_time", "3_time"]

#the first 97 columns are the FFT frequencies (0-32hz) like, 0, .3333, .666, 1, ... 32hz
#plot one of the rows
# Select first row from the data


#determine the start time of that row from the window_start_time column so that we can plot the time series and so that I can show and compare the fft of the 3 different window sizes on the same time series start

#OHHH the precision of the fft changes with the window size, so the fft of the 27 sec window is more precise than the 9 sec window and the 3 sec window. this means that the 27 sec window has more data points than the 9 sec window and the 3 sec window. (I didn't account for this in the previous code)
#%%


plot_fft(path27, offset, cohort, subject, 10)
#%%
plot_fft(path9, offset, cohort, subject, 10)
plot_fft(path3, offset, cohort, subject, 10)
#%%
plot_time(path27_time, offset, cohort, subject)
plot_time(path9_time, offset, cohort, subject)
plot_time(path3_time, offset, cohort, subject)
# %%

plot_stack(path27_time, cohort, subject, num_windows=100, offset=offset, normalize=True)
#%%
with open(testing_c2s02, 'rb') as f:
        c2s02 = pickle.load(f) #dataframe

with open(path27, 'rb') as f:
        data = pickle.load(f) #dataframe