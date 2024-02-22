import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

ppg = nk.ppg_simulate(duration=60, sampling_rate=64)
print('ppg shape:', ppg.shape)
print('-'*50)
print('ppg first 5:', ppg[:5])
print('-'*50)

print('now processing ppg...')
print('cleaning ppg...')
ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=64)
print('detecting ppg peaks...')
signals, info = nk.ppg_peaks(ppg_cleaned, sampling_rate=64, correct_artifacts=True)
print('done processing ppg')

print('head of signals:', signals.head())
print('-'*50)
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
print(info)

ppg_peaks = np.unique(info['PPG_Peaks'])
print('ppg_peaks:', ppg_peaks)

# nk.ppg_plot(signals, info)
# fig = plt.gcf()
plt.savefig("myfig.png")


combined_path = '/mnt/data2/david/data/c_01/s_01/combined/combined_e4.pkl'
print('testing various things with combined_df...')
print('loading combined_df.pkl...')
with open(combined_path, 'rb') as f:
    combined_df = pickle.load(f)

print('number of NaN values in the bvp column:', combined_df['bvp'].isna().sum())

print('first five rows with NaN values in the bvp column:', combined_df[combined_df['bvp'].isna()].head())


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