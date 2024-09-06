# %%
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from tqdm import tqdm

tqdm.pandas()

def plot_fft(path, offset, cohort, subject, cutoff=32, cleaned="nkCleaned"):
    """
    Plot the FFT of a specified row in the data.

    Parameters:
    - path (str): Path to the pickle file containing the data.
    - offset (int): Number of seconds forward from the first window_start.
    - cohort (int): Cohort number.
    - subject (int): Subject number.
    - cutoff (int): Frequency cutoff for the FFT plot. Default is 32.
    - cleaned (str): Data cleaning method. Default is "nkCleaned".
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Save the window_start_time of the first row to use as a reference for offset
    data_start_time = data.iloc[0, data.columns.to_list().index('window_start_time')]
    print("data_start_time", data_start_time)

    # Find the row number to cut based on offset
    target_time = data_start_time + timedelta(seconds=offset)
    print("target_time", target_time)
    row_num = data[data['Time'] >= target_time].index[0]
    print("row_num", row_num)
    print("final_time", data.iloc[row_num, data.columns.to_list().index('Time')])

    start_idx = data.columns.to_list().index('start_idx')
    columns_per_hz = start_idx / 32

    row = data.iloc[row_num, :start_idx]
    seconds_row = ((len(row) - 1) * 2) / 64.0
    if seconds_row.is_integer():
        seconds_row = int(seconds_row)

    cutoff_columns = int(cutoff * columns_per_hz)
    row = data.iloc[row_num, :cutoff_columns]

    max_magnitude = row.max()
    max_magnitude_column = row.index[np.argmax(row)]

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(8, 6))
    plt.plot(row)

    step_size = len(row) / cutoff
    max_freq = float(max_magnitude_column.replace('Magnitude_', '').replace('Hz', ''))
    plt.scatter(max_freq * step_size, max_magnitude, color='red')

    y_offset = 1000
    x_offset = 10

    if seconds_row < 4:
        y_offset = 100
        x_offset = 1
    elif seconds_row >= 4:
        y_offset = 300
        x_offset = 3

    plt.text(max_freq * step_size + x_offset, max_magnitude - y_offset, f'{max_freq} Hz', ha='left')
    xticks = np.linspace(0, len(row), num=cutoff//2+1)
    xticklabels = np.arange(0, cutoff+1, 2)

    plt.xticks(xticks, xticklabels)

    yticks, _ = plt.yticks()
    plt.yticks(yticks, [f'{int(yt/1000)}' for yt in yticks])
    plt.ylim(-y_offset, np.ceil(max_magnitude / (y_offset*3)) * (y_offset*3))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (x 10^3)')
    plt.title(f'FFT Frequencies ({seconds_row} sec window, c{cohort}s0{subject})')
    plt.savefig(f'{out_path}/c{cohort}s0{subject}_{seconds_row}sec_fft_{cleaned}_row{row_num}.png')
    plt.show()

#%%
def plot_time(path, offset, cohort, subject, cleaned="nkCleaned"):
    """
    Plot the time series of a specified row in the data.

    Parameters:
    - path (str): Path to the pickle file containing the data.
    - offset (int): Number of seconds forward from the first window_start.
    - cohort (int): Cohort number.
    - subject (int): Subject number.
    - cleaned (str): Data cleaning method. Default is "nkCleaned".
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Save the window_start_time of the first row to use as a reference for offset
    data_start_time = data.iloc[0, data.columns.to_list().index('window_start_time')]
    print("data_start_time", data_start_time)

    # Find the row number to cut based on offset
    target_time = data_start_time + timedelta(seconds=offset)
    print("target_time", target_time)
    row_num = data[data['Time'] >= target_time].index[0]
    print("row_num", row_num)
    print("final_time", data.iloc[row_num, data.columns.to_list().index('Time')])

    start_idx = data.columns.to_list().index('start_idx')
    row = data.iloc[row_num, :start_idx]

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(8, 6))

    seconds_row = len(row) / 64.0
    if seconds_row.is_integer():
        seconds_row = int(seconds_row)

    time = np.arange(0, seconds_row, 1/64)
    plt.plot(time, row)
    plt.ylim(-200, 200)
    plt.xticks(np.arange(0, np.ceil(seconds_row), 1))

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Time Series ({seconds_row} sec window, c{cohort}s0{subject})')
    plt.savefig(f'{out_path}/c{cohort}s0{subject}_{seconds_row}sec_time_series_{cleaned}_row{row_num}.png')
    plt.show()

#%%
def plot_stack(path, cohort, subject, offset=0, num_windows=None, cleaned="nkCleaned", start_idx_col_name='start_idx', normalize=True):
    """
    Plot stacked PPG signal plots.

    Parameters:
    - path (str): Path to the pickle file containing the data.
    - cohort (int): Cohort number.
    - subject (int): Subject number.
    - offset (int): Number of seconds forward from the first window_start. Default is 0.
    - num_windows (int): Number of windows to plot. If None, plots all windows.
    - cleaned (str): Data cleaning method. Default is "nkCleaned".
    - start_idx_col_name (str): Column name for start index. Default is 'start_idx'.
    - normalize (bool): Whether to normalize the data. Default is True.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    start_idx = data.columns.get_loc(start_idx_col_name)
    magnitude_data = data.iloc[:, :start_idx]
    times = data['Time']

    if offset > 0 and offset < len(times):
        magnitude_data = magnitude_data.iloc[offset:, :]
        times = times.iloc[offset:]

    if num_windows is not None and num_windows < len(times):
        magnitude_data = magnitude_data.iloc[:num_windows, :]
        times = times.iloc[:num_windows]

    if normalize:
        norm_magnitude_data = (magnitude_data - magnitude_data.min()) / (magnitude_data.max() - magnitude_data.min())
    else:
        norm_magnitude_data = magnitude_data

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(norm_magnitude_data, aspect='auto', cmap='viridis', origin='lower',
                    extent=[0, magnitude_data.shape[1], times.iloc[0], times.iloc[-1]])

    ax.set_xlabel('Sample Index within Window')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Stacked PPG Signal Plot (c{cohort}s0{subject})')

    cbar = fig.colorbar(cax)
    if normalize:
        cbar.set_label('Normalized Magnitude')
    else:
        cbar.set_label('Magnitude')

    plt.show()
#%%

# Initialize variables
sampling_rate = 64
cohort = 2
subject = 2
offset = 135  # 135 seconds after the first window_start_time
cleaned = "nkCleaned"

# Paths to the pickle files
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

#%%
# Plot FFT and time series for different window sizes
plot_fft(path27, offset, cohort, subject, 10)
plot_fft(path9, offset, cohort, subject, 10)
plot_fft(path3, offset, cohort, subject, 10)

plot_time(path27_time, offset, cohort, subject)
plot_time(path9_time, offset, cohort, subject)
plot_time(path3_time, offset, cohort, subject)

plot_stack(path27_time, cohort, subject, num_windows=100, offset=offset, normalize=True)

# Load additional data for testing/analysis
with open(testing_c2s02, 'rb') as f:
    c2s02 = pickle.load(f)

with open(path27, 'rb') as f:
    data = pickle.load(f)