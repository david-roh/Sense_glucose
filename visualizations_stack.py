# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
import math
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from libs.helper import align_ppg
from alignment import plot_random_aligned_beats

# Default variables
sampling_rate = 64
cohort = "2" # 1, 2, 3
subject = "02" # 01, 02, 03, 04, 05
num_seconds = 3 #3, 9, 27
filter_choice = "" #none, filter, nofilter
peak_choice = "nopeak" #neurokit, empatica, nopeak
representation_choice = "fft" #fft or time
cleaned_choice = "nkCleaned" #nkCleaned or unCleaned

num_beats = 5
bin_size = 10
use_percentiles = True

# %%
def show_aligned_demo(row_data, row_data_aligned, sampling_rate=sampling_rate):
    """
    Show aligned PPG signals for demonstration.

    Parameters:
    - row_data (pd.Series): Original PPG data.
    - row_data_aligned (pd.Series): Aligned PPG data.
    - sampling_rate (int): Sampling rate. Default is 64.
    """
    ppg_data = np.array(row_data[:sampling_rate])
    aligned_ppg_data = np.array(row_data_aligned[:sampling_rate])

    print("PPG Original Data: ", ppg_data)
    print("PPG Aligned Data: ", aligned_ppg_data)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(ppg_data.shape[0]), ppg_data, label="PPG Original", color='grey')
    plt.plot(np.arange(aligned_ppg_data.shape[0]), aligned_ppg_data, label="PPG Aligned", color='cyan')

    valr = int(row_data['r'])
    valr_aligned = int(row_data_aligned['r'])

    print("R Original: ", valr)
    print("R Aligned: ", valr_aligned)

    if 0 <= valr < len(ppg_data):
        plt.scatter(valr, ppg_data[valr], color='red', marker='v', label="R Original")

    if 0 <= valr_aligned < len(aligned_ppg_data):
        plt.scatter(valr_aligned, aligned_ppg_data[valr_aligned], color='red', marker='v', label="R Aligned")

    plt.axvline(x=row_data['r'], color='r', linestyle='--', label="R Peak Original")
    plt.axvline(x=row_data_aligned['r'], color='b', linestyle='--', label="R Peak Aligned")

    plt.legend(loc='upper right')
    plt.title(f"PPG Peak: Original({row_data['r']}) vs Aligned({row_data_aligned['r']})")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("./demo/alignment.png", dpi=300)

# %%
def plot_random_aligned_beats(df_aligned, num_beats=num_beats, sampling_rate=sampling_rate):
    """
    Plot random aligned PPG beats.

    Parameters:
    - df_aligned (pd.DataFrame): DataFrame with aligned PPG data.
    - num_beats (int): Number of random beats to plot. Default is 5.
    - sampling_rate (int): Sampling rate. Default is 64.
    """
    colors = cm.rainbow(np.linspace(0, 1, num_beats))
    plt.figure(figsize=(12, 6))

    for i in range(num_beats):
        random_row = np.random.randint(0, len(df_aligned))
        row_data_aligned = df_aligned.iloc[random_row]
        aligned_ppg_data = np.array(row_data_aligned[:sampling_rate])
        valr_aligned = int(row_data_aligned['r'])

        plt.plot(np.arange(aligned_ppg_data.shape[0]), aligned_ppg_data, label=f"PPG Aligned {i+1}", color=colors[i])

        if 0 <= valr_aligned < len(aligned_ppg_data):
            plt.scatter(valr_aligned, aligned_ppg_data[valr_aligned], color=colors[i], marker='v', label=f"R Aligned {i+1}")

    plt.axvline(x=valr_aligned, color='black', linestyle='--', label="R Peak Aligned")

    plt.legend(loc='upper right', fontsize='small')
    plt.title(f"Overlay of {num_beats} Aligned PPG Beats")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("./demo/overlay_alignment.png", dpi=300)

# %%
def calculate_and_plot_hr(df, bin_size=bin_size):
    """
    Calculate and plot heart rate frequency.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'rr' column for RR intervals.
    - bin_size (int): Bin size for the histogram. Default is 10.
    """
    df['hr_calc'] = 60 / df['rr']

    plt.figure(figsize=(12, 6))
    plt.hist(df['hr_calc'], bins=range(int(min(df['hr_calc'])), math.ceil(max(df['hr'])) + bin_size, bin_size))
    plt.title("Heart Rate Frequency")
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("./demo/hr_frequency.png", dpi=300)

# %%
plt.rcParams.update({'font.size': 18})

def plot_stacked_heatmap(df_aligned, sampling_rate=sampling_rate, use_percentiles=use_percentiles, filename='c2s02'):
    """
    Plot stacked heatmap of PPG data sorted by heart rate and glucose.

    Parameters:
    - df_aligned (pd.DataFrame): DataFrame with aligned PPG data.
    - sampling_rate (int): Sampling rate. Default is 64.
    - use_percentiles (bool): Whether to use percentiles for normalization. Default is True.
    - filename (str): Filename for the output plot. Default is 'c2s02'.
    """
    sorted_by_hr = df_aligned.sort_values('hr')
    sorted_by_glucose = df_aligned.sort_values('glucose')

    grouped_hr = sorted_by_hr.groupby('hr').mean()
    grouped_glucose = sorted_by_glucose.groupby('glucose').mean()

    print("Data types in 'grouped_hr':")
    print(grouped_hr.dtypes)
    print("\nData types in 'grouped_glucose':")
    print(grouped_glucose.dtypes)

    ppg_data_hr = grouped_hr.iloc[:, :sampling_rate].values
    ppg_data_glucose = grouped_glucose.iloc[:, :sampling_rate].values
    combined_data = np.concatenate((ppg_data_hr, ppg_data_glucose))

    if use_percentiles:
        vmax = np.percentile(combined_data, 99.9)
        vmin = -vmax
    else:
        vmax = 200
        vmin = -200

    plt.figure(figsize=(8, 8))
    plt.imshow(ppg_data_hr, cmap='seismic', norm=colors.Normalize(vmin=vmin, vmax=vmax), aspect='auto')
    plt.title('PPG Stack Sorted by Heart Rate')
    plt.xlabel('Time (Increments of 1/64 seconds)')
    plt.ylabel('Heart Rate')
    yticks_loc_hr = np.linspace(start=0, stop=len(grouped_hr.index)-1, num=10, dtype=int)
    plt.yticks(yticks_loc_hr, pd.Series(grouped_hr.index[yticks_loc_hr]).round(0).astype(int))
    plt.colorbar(label='Normalized Amplitude')
    plt.savefig(f'./demo/{filename}_heatmap_hr.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(ppg_data_glucose, cmap='seismic', norm=colors.Normalize(vmin=vmin, vmax=vmax), aspect='auto')
    plt.title('PPG Stack Sorted by Glucose')
    plt.xlabel('Time (Increments of 1/64 seconds)')
    plt.ylabel('Glucose')
    yticks_loc_glucose = np.linspace(start=0, stop=len(grouped_glucose.index)-1, num=10, dtype=int)
    plt.yticks(yticks_loc_glucose, pd.Series(grouped_glucose.index[yticks_loc_glucose]).round(0).astype(int))
    plt.colorbar(label='Normalized Amplitude')
    plt.savefig(f'./demo/{filename}_heatmap_glucose.png', dpi=300)
    plt.close()

# %%
def plot_pca_and_beat_plots(df, col='glucose', n_components=2, percentiles=10, time_column='Time'):
    """
    Plot PCA and beat plots for PPG data.

    Parameters:
    - df (pd.DataFrame): DataFrame with PPG data.
    - col (str): Column name for coloring the PCA plot. Default is 'glucose'.
    - n_components (int): Number of PCA components. Default is 2.
    - percentiles (int): Number of percentiles for grouping. Default is 10.
    - time_column (str): Column name for time data. Default is 'Time'.
    """
    x = df.drop(['hypo_label', 'glucose', 'hr', 'flag'], axis=1).select_dtypes(include=[np.number])
    x.columns = x.columns.astype(str)
    x = x.loc[:, x.columns != col].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)

    principal_df = pd.DataFrame(data=principalComponents, columns=['PCA1', 'PCA2'])
    final_df = pd.concat([principal_df, df[[col, time_column]].reset_index(drop=True)], axis=1)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(final_df['PCA1'], final_df['PCA2'], c=final_df[col], cmap='viridis')
    plt.colorbar(scatter, label=f'{col}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'2 component PCA, colored by {col}')
    plt.savefig(f'./demo/pca_scatter_{col}.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(final_df['PCA1'], final_df['PCA2'], c=final_df[col], cmap='viridis')
    plt.colorbar(scatter, label=f'{col}')
    final_df['Percentile'] = pd.qcut(final_df[col], percentiles, labels=False)
    averages = final_df.groupby('Percentile').mean()[['PCA1', 'PCA2']]
    plt.scatter(averages['PCA1'], averages['PCA2'], c='red', marker='X', s=100, label='Average per Percentile')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title(f'2 component PCA with averages, colored by {col}')
    plt.savefig(f'./demo/pca_scatter_{col}_with_averages.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    percentile_groups = final_df.groupby('Percentile')
    for name, group in percentile_groups:
        plt.plot(group[time_column], group[col], label=f'{name}th Percentile')
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.title(f'Beat plot colored by percentiles of {col}')
    plt.legend(loc='upper right')
    plt.savefig(f'./demo/beat_plot_{col}_percentiles.png', dpi=300)
    plt.close()

# %%
# Set file paths and load data
ppg_path = f'/mnt/data2/david/data/c{cohort}/s{subject}/e4/filtered_data_with_ibi_presence.pkl'
aligned_ppg_path = f'/mnt/data2/david/data/c{cohort}/s{subject}/e4/aligned_data.pkl'

print(f"Reading aligned data from {aligned_ppg_path} ~")
df_aligned = pd.read_pickle(aligned_ppg_path)

random_row = np.random.randint(0, len(df_aligned))
pd.set_option('display.max_columns', None)
print(df_aligned.head())

auto_sample_rate = 192 if len(df_aligned.columns) > 192 else sampling_rate

# Uncomment the following lines to execute the respective functions
# show_aligned_demo(df.iloc[random_row], df_aligned.iloc[random_row], sampling_rate=auto_sample_rate)
# plot_random_aligned_beats(df_aligned, num_beats=6, sampling_rate=auto_sample_rate)
# calculate_and_plot_hr(df)

df_aligned_nocturnal = df_aligned[(df_aligned['Time'].dt.hour >= 23) | (df_aligned['Time'].dt.hour <= 7)]
base_name = os.path.basename(aligned_ppg_path)
file_name, _ = os.path.splitext(base_name)
df_aligned_nocturnal = df_aligned_nocturnal[df_aligned_nocturnal['hr'] >= 40]
plot_stacked_heatmap(df_aligned_nocturnal, sampling_rate=auto_sample_rate, use_percentiles=use_percentiles, filename=file_name)

print(f"Number of all day beats: {len(df_aligned)}")
print(f"Number of nocturnal beats: {len(df_aligned_nocturnal)}")

og_filename = os.path.basename(aligned_ppg_path).split('.')[0]
out_dir = os.path.dirname(aligned_ppg_path)
df_aligned_nocturnal.to_pickle(os.path.join(out_dir, f"{og_filename}_nocturnal.pkl"))

print(f"Columns of df_aligned: {df_aligned_nocturnal.columns.tolist()}")
