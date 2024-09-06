#%%
import time
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

combined_og = pd.read_pickle('/mnt/data2/david/data/c_02/s_02/e4/filtered_data_with_ibi_presence.pkl')
print(combined_og.columns)



#%%
#only keep nocturnal data based on datetimeindex (11pm to 7 am):
combined_og = combined_og.between_time('23:00', '07:00')
#turn combined into a random subsection that has 10000 rows (contiguous)
random_num = np.random.randint(0, combined_og.shape[0] - 10000)
combined = combined_og.iloc[random_num:random_num + 10000]

# Time Series Plot with markers for IBI_Presence
plt.figure(figsize=(15, 5))
plt.plot(combined.index, combined['bvp'], label='BVP Signal')
plt.scatter(combined.index[combined['IBI_Presence']], combined['bvp'][combined['IBI_Presence']], color='red', label='IBI Presence')
plt.title('BVP Signal with IBI Presence Marked')
plt.xlabel('Datetime')
plt.ylabel('BVP')
plt.legend()
plt.show()

# Histogram of IBI intervals
plt.figure(figsize=(10, 5))
# We can only plot intervals where IBI_Presence is True and 'bvp' is not NaN
ibi_intervals = combined.loc[combined['IBI_Presence'] & ~combined['bvp'].isna(), 'ibi']
plt.hist(ibi_intervals, bins=30, alpha=0.7)
plt.title('Histogram of IBI Intervals')
plt.xlabel('Milliseconds')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot of BVP against IBI intervals to see the points distribution
plt.figure(figsize=(10, 5))
plt.scatter(combined.index, combined['bvp'], c=combined['IBI_Presence'], cmap='bwr', alpha=0.5)
plt.colorbar(label='IBI Presence (1 for True, 0 for False)')
plt.title('Scatter Plot of BVP against IBI Presence')
plt.xlabel('Datetime')
plt.ylabel('BVP')
plt.show()

# Statistical Summary
print("Statistical Summary of IBI Presence:")
print(combined['IBI_Presence'].describe())

# %%
from scipy.fft import fft
import matplotlib.pyplot as plt

# Define the sampling rate and window size
sampling_rate = 64  # Hz
window_size = int(3.5 * sampling_rate)  # samples

# Perform the FFT on each window and save the plot to a PNG file
for i in tqdm(range(0, len(combined['bvp']) - window_size, window_size)):
    # Extract the window
    window = combined['bvp'].iloc[i:i+window_size]
    
    # Perform the FFT
    yf = fft(window)
    xf = np.linspace(0.0, sampling_rate/2.0, window_size//2)
    
    # Plot the FFT
    plt.figure()
    plt.plot(xf, 2.0/window_size * np.abs(yf[:window_size//2]))
    plt.grid()
    plt.title('FFT of BVP signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    
    # Save the plot to a PNG file
    plt.savefig(f'fft_{i//window_size}.png')