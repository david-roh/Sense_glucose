
# %%
from curses import window
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
tqdm.pandas()

sampling_rate = 64

ppg = nk.ppg_simulate(duration=60, sampling_rate=sampling_rate)
print('ppg shape:', ppg.shape)
print('-'*50)
print('ppg first 5:', ppg[:5])
print('-'*50)
print('ppg type:', type(ppg))

print('now processing ppg...')
print('cleaning ppg...')
ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=64) #returns array
# print('detecting ppg peaks...')
# signals, info = nk.ppg_peaks(ppg_cleaned, sampling_rate=64, correct_artifacts=True)
print('done processing ppg')


# %%

#cut into 50% overlapping 3 second windows 
window_size = 3*sampling_rate
stride = window_size//2

# Create overlapping windows
windows = []
for i in range(0, len(ppg_cleaned) - window_size + 1, stride):
    windows.append(ppg_cleaned[i:i + window_size])

# Convert list of windows to a DataFrame
windows_df = pd.DataFrame(windows)

print('Windows DataFrame shape:', windows_df.shape)

print('shape:', ppg_cleaned.shape)
# %%
combined_path = '/mnt/data2/david/data/c_02/s_02/e4/combined_e4_bvp_only.pkl'
print('testing various things with combined_df...')
print('loading combined_df.pkl...')
with open(combined_path, 'rb') as f:
    combined_df = pickle.load(f)

combined_df.reset_index(inplace=True)
#rename the datetime column to Timestamp
combined_df.rename(columns={'datetime':'Timestamp'}, inplace=True)

print('head of combined_df:\n', combined_df.head())
print("combined_df shape:", combined_df.shape)
'''
head of combined_df:
                    Timestamp  bvp
0 2022-09-27 15:04:47.000000 -0.0
1 2022-09-27 15:04:47.015625 -0.0
2 2022-09-27 15:04:47.031250 -0.0
3 2022-09-27 15:04:47.046875 -0.0
4 2022-09-27 15:04:47.062500 -0.0
combined_df shape: (74080556, 2)
'''

# #remove all other columns other than the datetimeindex and the bvp column and remove all rows with NaN values in the bvp column
# print("count of NaN values in bvp column:", combined_df['bvp'].isna().sum())
# combined_df = combined_df[['bvp']]
# #show progress of removing NaN values
# combined_df = combined_df.dropna()

# print('head of combined_df:\n', combined_df.head())
# print("combined_df shape:", combined_df.shape)


#%%

glucose_path = '/mnt/data2/david/data/c_02/s_02/e4/glucose.pkl'
print('testing various things with glucose_df...')
print('loading glucose.pkl...')
with open(glucose_path, 'rb') as f:
    glucose_df = pickle.load(f)
    
print('head of glucose_df:\n', glucose_df.head())
print("glucose_df shape:", glucose_df.shape)
'''
head of glucose_df:
             Timestamp  glucose
0 2022-07-16 00:04:54      137
1 2022-07-16 00:09:54      137
2 2022-07-16 00:14:55      141
3 2022-07-16 00:19:55      145
4 2022-07-16 00:24:54      148
glucose_df shape: (25662, 2)
'''

#%%
# glucose_df has two columns, 'Timestamp': timestamp of the end of the glucose reading's 5 minute interval and 'glucose': the glucose reading for that interval
# so, for example for a row that has Timestamp = 2022-07-16 00:05:54 and glucose = 137, that means that the glucose reading for the 5 minute interval starts at 2022-07-16 00:00:54 and ends at 2022-07-16 00:05:54 and the glucose reading for that interval is 137.

# combined_df has two columns, 'Timestamp': timestamp of the bvp reading and 'bvp': the bvp reading


#eventually, I will have a dataframe with the rows corresponding to the windows and the columns corresponding to the bvp values in that window, the timestamp of the first bvp reading of the window, the glucose reading for that window, a hypoglycmeia boolean (if the glucose reading < 70), and a flag indicating the index of the glucose reading that the window corresponds to.

#based on glucose_df and combined_df, create a new dataframe that has 3 second windows of bvp readings (50% overlap between windows), the timestamp of the beginning of the window, the glucose reading for that window (if the boundary between glucose readings is between the window, just take which ever is >=50%), a hypoglycemia flag for that glucose reading (<70), and the index of the glucose reading that the window corresponds to.


# Step 1: Convert 'Timestamp' to datetime if not already
combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'])

# Step 2: Resample 'combined_df' to 3-second intervals
combined_df.set_index('Timestamp', inplace=True)
combined_df = combined_df.resample('3S').mean()

# Step 3: Create overlapping windows
window_size = 2  # for 50% overlap
combined_df = combined_df.rolling(window_size).mean()

# Step 4: Merge 'glucose_df' with 'combined_df'
merged_df = pd.merge_asof(combined_df, glucose_df, on='Timestamp', direction='nearest')

# Step 5: Create 'hypoglycemia' column
merged_df['hypoglycemia'] = merged_df['glucose'] < 70

# Step 6: Create 'glucose_index' column
merged_df['glucose_index'] = merged_df.index


merged_df = pd.merge_asof(combined_df, glucose_df, on='Timestamp', direction='nearest')
print('head of merged_df:\n', merged_df.head())

#%%
# Convert glucose timestamps to the same frequency as BVP readings
# glucose_df.set_index('Timestamp', inplace=True)
# glucose_df = glucose_df.resample('15.625ms').ffill()


# Align the indices of the two dataframes
# combined_df, glucose_df = combined_df.align(glucose_df, join='inner')

# Create overlapping windows
window_size = 3 * sampling_rate
stride = window_size // 2
windows = []
timestamps = []
glucose_readings = []
hypoglycemia_flags = []
glucose_indices = []

for i in tqdm(range(0, len(combined_df) - window_size + 1, stride)):
    windows.append(combined_df['bvp'].iloc[i:i + window_size].values)
    timestamps.append(combined_df.index[i])
    glucose_readings.append(glucose_df['glucose'].iloc[i])
    hypoglycemia_flags.append(glucose_df['glucose'].iloc[i] < 70)
    glucose_indices.append(i // (5 * 60 * sampling_rate))

# Create a new DataFrame
windows_df = pd.DataFrame(windows)
windows_df['Timestamp'] = timestamps
windows_df['Glucose'] = glucose_readings
windows_df['Hypoglycemia'] = hypoglycemia_flags
windows_df['GlucoseIndex'] = glucose_indices

print('Windows DataFrame shape:', windows_df.shape)
print('head of windows_df:\n', windows_df.head())

#%%    
# now, perform a fft on each window and store the results in a new dataframe
# the columns of the new dataframe will be the frequency values (+ metadata like timestamp glucose hypoglycemia, glucoseindex) and the rows will be the windows
from scipy.fft import fft

# Create a new DataFrame
fft_df = pd.DataFrame()

for i in tqdm(range(len(windows_df))):
    fft_values = np.abs(fft(windows_df.iloc[i, :-4]))
    fft_df = fft_df.append(pd.Series(fft_values), ignore_index=True)
    
fft_df['Timestamp'] = windows_df['Timestamp']
fft_df['Glucose'] = windows_df['Glucose']
fft_df['Hypoglycemia'] = windows_df['Hypoglycemia']
fft_df['GlucoseIndex'] = windows_df['GlucoseIndex']

print('FFT DataFrame shape:', fft_df.shape)
print('head of fft_df:\n', fft_df.head())
#%%
# visualizations
# get a random window (with a seed to be deterministic) index
# plot the windows_df row with that index [:-4] (the last 4 columns are metadata)
# plot the fft_df row with that index [:-4] (the last 4 columns are metadata)

np.random.seed(0)
window_index = np.random.randint(0, len(windows_df))
print('window_index:', window_index)

plt.figure()
plt.plot(windows_df.iloc[window_index, :-4])
plt.title('Window')
plt.xlabel('Sample')
plt.ylabel('BVP Value')
plt.show()
plt.savefig('./demo/window.png')
plt.close()

plt.figure()
plt.plot(fft_df.iloc[window_index, :-4])
plt.title('FFT')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()
plt.savefig('./demo/fft.png')
plt.close()




# print('head of signals:', signals.head())
# print('-'*50)
# print('ppg_clean shape:', signals['PPG_Clean'].shape)
# print('-'*50)
# print('ppg_clean first 5:', signals['PPG_Clean'][:5])
# print('-'*50)
# print('ppg_rate shape:', signals['PPG_Rate'].shape)
# print('-'*50)
# print('ppg_rate first 5:', signals['PPG_Rate'][:5])
# print('-'*50)
# print('ppg_peaks shape:', signals['PPG_Peaks'].shape)
# print('-'*50)
# print('ppg_peaks first 5:', signals['PPG_Peaks'][:5])
# print('-'*50)

#print info
# print(info)

# ppg_peaks = np.unique(info['PPG_Peaks'])
# print('ppg_peaks:', ppg_peaks)

# nk.ppg_plot(signals, info)
# fig = plt.gcf()
# plt.savefig("myfig.png")






# print('number of NaN values in the bvp column:', combined_df['bvp'].isna().sum())

# print('first five rows with NaN values in the bvp column:', combined_df[combined_df['bvp'].isna()].head())

# print('head of glucose_df:', glucose_df.head())



# print('head of combined_df:', combined_df.head())
# print('attempting to extract time (datetime index)')
# arr_time = np.array(combined_df.index) 
# print('arr_time:', arr_time)

# glucose_path = '/mnt/data2/david/data/c_01/s_01/combined/glucose.pkl'
# print('testing various things with glucose_df...')
# print('loading glucose_df.pkl...')
# with open(glucose_path, 'rb') as f:
#     glucose_df = pickle.load(f)
# print('head of glucose_df:', glucose_df.head())
# print('attempting to extract time (datetime index)')
# arr_time = np.array(combined_df.index) 
# print('arr_time:', arr_time)

#note, what neurokit does it that they get the hear rate and determine= the window size directly from that. Of the window size, before the r-peak is .35 of the window, and after the r-peak is .65 of the window.



# signals, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate=1000)
# print(signals)
# print(waves)
# print(signals.iloc[waves["ECG_P_Peaks"][0]])
# print(len(waves["ECG_P_Peaks"]))
# nk.events_plot([waves["ECG_P_Peaks"], waves["ECG_T_Peaks"]], ecg)
# plt.savefig("test.png")
# cardiac_phase = nk.ecg_phase(ecg_cleaned=ecg, rpeaks=rpeaks,
#                           delineate_info=waves, sampling_rate=1000)


# _, ax = plt.subplots(nrows=2)

# ax[0].plot(nk.rescale(ecg), label="ECG", color="red", alpha=0.3)


# ax[0].plot(cardiac_phase["ECG_Phase_Atrial"], label="Atrial Phase", color="orange")

# ax[0].plot(cardiac_phase["ECG_Phase_Completion_Atrial"],
#           label="Atrial Phase Completion", linestyle="dotted")

# ax[0].legend(loc="upper right")

# ax[1].plot(nk.rescale(ecg), label="ECG", color="red", alpha=0.3)

# ax[1].plot(cardiac_phase["ECG_Phase_Ventricular"], label="Ventricular Phase", color="green")

# ax[1].plot(cardiac_phase["ECG_Phase_Completion_Ventricular"],
#        label="Ventricular Phase Completion", linestyle="dotted")

# ax[1].legend(loc="upper right")

# plt.savefig("test.png")
# %%
