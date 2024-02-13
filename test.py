import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ppg = nk.ppg_simulate(duration=6, sampling_rate=64)
print('ppg shape:', ppg.shape)
print('-'*50)
print('ppg first 5:', ppg[:5])
print('-'*50)

print('now processing ppg...')
signals, info = nk.ppg_process(ppg, report="text", sampling_rate=64)


print('ppg_clean shape:', signals['PPG_Clean'].shape)
print('-'*50)
print('ppg_clean first 5:', signals['PPG_Clean'][:5])
print('-'*50)
print('ppg_rate shape:', signals['PPG_Rate'].shape)
print('-'*50)
print('ppg_rate first 5:', signals['PPG_Rate'][:5])
print('-'*50)
print('ppg_peaks shape:', signals['PPG_Peaks'].shape)
print('-'*50)
print('ppg_peaks first 5:', signals['PPG_Peaks'][:5])
print('-'*50)

#print info
print(info)

nk.ppg_plot(signals, info)
# fig = plt.gcf()
plt.savefig("myfig.png")



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