#%%
import time
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


#exploring
# if __name__ == "__main__":
    # parser = argparse.ArgumentParser("This script looks into using ibi values to segment beats and trust that Empatica E4 is filtering out noisy data")
    # parser.add_argument('--combined', type=str, help='Path to the combined file')
    # args = parser.parse_args()
    
    # #combined is a pkl, load it in
    # combined = pd.read_pickle(args.combined)
combined = pd.read_pickle('/mnt/data2/david/data/c_02/s_02/e4/combined_e4.pkl')
    #%%
    #check the columns
print(combined.columns)
# %%
#SECTION ON IBI VALUES:
# print number of IBI values
print('Number of IBI values: ', combined['ibi'].count())
print('number of rows: ', combined.shape[0])
#print the first 5 rows with existing IBI values
print(combined[combined['ibi'].notnull()].head(5))

#min max, mean of ibi values, also want to see how many decimal places
print('Min IBI value: ', combined['ibi'].min())
print('Max IBI value: ', combined['ibi'].max())
# print('Mean IBI value: ', combined['ibi'].mean())
# print('Number of decimal places: ', combined['ibi'].apply(lambda x: len(str(x).split('.')[1])).max())

# %%
#print the type of the index
print(type(combined.index))
#%%
print("Initializing IBI_Presence column...")
combined['IBI_Presence'] = False

#turn ibi, bvp to float64 instead of float16
print("Converting ibi to float64...")
combined['ibi'] = combined['ibi'].astype('float64')

print("Computing previous diastolic points datetimes...")
prev_ibi_datetimes = combined.index - pd.to_timedelta(combined['ibi'], unit='ms')

print("Creating DataFrame of non-null previous IBI datetimes...")
prev_ibi_df = pd.DataFrame({
    'prev_ibi_datetime': prev_ibi_datetimes[combined['ibi'].notnull()]
}).drop_duplicates()

print("Finding nearest index for each unique previous IBI datetime...")
all_datetimes = combined.index
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
        combined.at[all_datetimes[closest_datetime_index], 'IBI_Presence'] = True
        marked_indices.add(closest_datetime_index)

# Mark the IBI presence for current rows with non-null IBI
combined.loc[combined['ibi'].notnull(), 'IBI_Presence'] = True

# Now you have IBI_Presence marked for current and previous diastolic points
#%%
#printout information on IBI_Presence
print('Number of rows with IBI_Presence: ', combined['IBI_Presence'].sum())
print('Number of rows without IBI_Presence: ', combined.shape[0] - combined['IBI_Presence'].sum())
print('Number of rows total: ', combined.shape[0])
#%%
# Filter out rows where 'bvp' is NaN and 'IBI_Presence' is True
filtered_data = combined[~(combined['bvp'].isna() & combined['IBI_Presence'])]
print('Number of rows after filtering: ', filtered_data.shape[0])


# %%
print('number of total rows before filtering: ', combined.shape[0])
print('number of rows with IBI_presence after filtering: ', filtered_data['IBI_Presence'].sum())
#save to pickle

# %%
# Compute the time difference between rows
# time_diff = combined.index.to_series().diff().dt.total_seconds()

# time_diff_filtered = filtered_data.index.to_series().diff().dt.total_seconds()
# #%%
# #on the plot i only see one bar, which is strange, i would expect to see more bars
# print('number of unique time differences (filtered): ', time_diff_filtered.nunique())
# # got 25 unique time differences, meaning im just plotting in the wrong way.
# print(time_diff_filtered.value_counts())

# print('number of unique time differences (combined): ', time_diff.nunique())
# print(time_diff.value_counts())
# %%

#before saving check if i need the precision of float64. If i don't I will convert down to float32 or float16 as needed

#check the max number of decimal places used in the ibi column.
# print('Number of decimal places: ', filtered_data['ibi'].apply(lambda x: len(str(x).split('.')[1])).max())
# %%
#save to pickle
filtered_data.to_pickle('/mnt/data2/david/data/c_02/s_02/e4/filtered_data_with_ibi_presence.pkl')
combined.to_pickle('/mnt/data2/david/data/c_02/s_02/e4/combined_data_with_ibi_presence.pkl')
