#%%
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk


# Function to load data from a pickle file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to save data to a pickle file
def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        
cohort = "02"
subject = "02"

# File paths: note that bvp data is 64hz, glucose is every 5 minutes
combined_file_path = f"/mnt/data2/david/data/c_{cohort}/s_{subject}/e4/combined_e4.pkl"
glucose_file_path = f"/mnt/data2/david/data/c_{cohort}/s_{subject}/e4/glucose.pkl"

# Load data
combined_data = load_data(combined_file_path)
glucose_data = load_data(glucose_file_path)

# Reset the index of combined_data to make 'datetime' a column
combined_data = combined_data.reset_index()

# Keep only 'datetime' and 'bvp' columns in combined_data
combined_data = combined_data[['datetime', 'bvp']]

# Convert 'bvp' column to float64 data type
combined_data['bvp'] = combined_data['bvp'].astype(np.float64)

# Drop rows with NaN values in combined_data
combined_data = combined_data.dropna()

# Reset the index of glucose_data to make 'Timestamp' a column and rename it to 'datetime'
glucose_data = glucose_data.reset_index().rename(columns={'index': 'flag', 'Timestamp': 'datetime'})

# Merge combined_data and glucose_data on 'datetime' column, taking the nearest value when there's no exact match
combined_data = pd.merge_asof(combined_data.sort_values('datetime'), glucose_data.sort_values('datetime'), on='datetime', direction='nearest')

# Save the merged data (numerical index, columns: datetime, bvp, flag, glucose)
save_data(combined_data, f"/mnt/data2/david/data/c_{cohort}/s_{subject}/e4/combined_e4_glucose.pkl")

'''
                    datetime  bvp   flag  glucose
0 2022-05-18 22:40:26.000000 -0.0  15751      222
1 2022-05-18 22:40:26.015625 -0.0  15751      222
2 2022-05-18 22:40:26.031250 -0.0  15751      222
3 2022-05-18 22:40:26.046875 -0.0  15751      222
'''


#%%

bvp_data = combined_data['bvp']
bvp_data = np.asarray(bvp_data)
idx =  np.arange(len(bvp_data))
bvp_cleaned = nk.ppg_clean(bvp_data, sampling_rate=64)

combined_data_cleaned = combined_data.copy()
combined_data_cleaned['bvp'] = bvp_cleaned
save_data(combined_data_cleaned, f"/mnt/data2/david/data/c_{cohort}/s_{subject}/e4/nkcleaned.pkl")







# %%
