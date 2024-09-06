from ast import arg
import numpy as np
import pandas as pd
from libs.helper import align_ppg
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import argparse
import matplotlib.cm as cm
import math
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from alignment import plot_random_aligned_beats
# from cuml import TSNE


def show_aligned_demo(row_data, row_data_aligned, sampling_rate=64):
    ppg_data = np.array(row_data[:sampling_rate])
    aligned_ppg_data = np.array(row_data_aligned[:sampling_rate])

    # Debugging prints
    print("PPG Original Data: ", ppg_data)
    print("PPG Aligned Data: ", aligned_ppg_data)

    plt.figure(figsize=(12, 6))  # Adjust figure size
    plt.plot(np.arange(ppg_data.shape[0]), ppg_data, label="PPG Original", color='grey')
    plt.plot(np.arange(aligned_ppg_data.shape[0]), aligned_ppg_data, label="PPG Aligned", color='cyan')

    valr = int(row_data['r'])
    valr_aligned = int(row_data_aligned['r'])

    # Debugging prints
    print("R Original: ", valr)
    print("R Aligned: ", valr_aligned)

    if valr >= 0 and valr < len(ppg_data):
        plt.scatter(valr, ppg_data[valr], color='red', marker='v', label="R Original")

    if valr_aligned >= 0 and valr_aligned < len(aligned_ppg_data):
        plt.scatter(valr_aligned, aligned_ppg_data[valr_aligned], color='red', marker='v', label="R Aligned")   

    # Mark the R peak vertical line
    plt.axvline(x=row_data['r'], color='r', linestyle='--', label="R Peak Original")
    plt.axvline(x=row_data_aligned['r'], color='b', linestyle='--', label="R Peak Aligned")

    plt.legend(loc='upper right')  # Adjust legend position
    plt.title("PPG Peak: Original({}) vs Aligned({})".format(row_data['r'], row_data_aligned['r']))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("./demo/alignment.png", dpi=300)  # Save with high DPI

def plot_random_aligned_beats(df_aligned, num_beats=5, sampling_rate=64):
    colors = cm.rainbow(np.linspace(0, 1, num_beats))  # Generate as many colors as needed
    plt.figure(figsize=(12, 6))  # Adjust figure size

    for i in range(num_beats):
        random_row = np.random.randint(0, len(df_aligned))
        row_data_aligned = df_aligned.iloc[random_row]
        aligned_ppg_data = np.array(row_data_aligned[:sampling_rate])
        valr_aligned = int(row_data_aligned['r'])

        plt.plot(np.arange(aligned_ppg_data.shape[0]), aligned_ppg_data, label="PPG Aligned {}".format(i+1), color=colors[i])

        if valr_aligned >= 0 and valr_aligned < len(aligned_ppg_data):
            plt.scatter(valr_aligned, aligned_ppg_data[valr_aligned], color=colors[i], marker='v', label="R Aligned {}".format(i+1))   

    # Mark the R peak vertical line
    plt.axvline(x=valr_aligned, color='black', linestyle='--', label="R Peak Aligned")

    plt.legend(loc='upper right', fontsize='small')  # Adjust legend position and font size
    plt.title("Overlay of {} Aligned PPG Beats".format(num_beats))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("./demo/overlay_alignment.png", dpi=300)  # Save with high DPI
    
def calculate_and_plot_hr(df, bin_size=10):
    # Calculate heart rate from 'rr' column
    df['hr_calc'] = 60 / df['rr']

    # Plot frequency of calculated heart rates
    plt.figure(figsize=(12, 6))  # Adjust figure size
    plt.hist(df['hr_calc'], bins=range(int(min(df['hr_calc'])), math.ceil(max(df['hr'])) + bin_size, bin_size))
    plt.title("Heart Rate Frequency")
    plt.xlabel('Heart Rate (bpm)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("./demo/hr_frequency.png", dpi=300)  # Save with high DPI


plt.rcParams.update({'font.size': 18})

def plot_stacked_heatmap(df_aligned, sampling_rate=64, use_percentiles=True, filename='c2s02'):
    # Sort DataFrame by 'hr' and 'glucose' columns
    sorted_by_hr = df_aligned.sort_values('hr')
    sorted_by_glucose = df_aligned.sort_values('glucose')

    # Group by 'hr' and 'glucose' and calculate the mean
    grouped_hr = sorted_by_hr.groupby('hr').mean()
    grouped_glucose = sorted_by_glucose.groupby('glucose').mean()
    
    # Print data types
    print("Data types in 'grouped_hr':")
    print(grouped_hr.dtypes)
    print("\nData types in 'grouped_glucose':")
    print(grouped_glucose.dtypes)

    # Extract PPG data
    ppg_data_hr = grouped_hr.iloc[:, :sampling_rate].values
    ppg_data_glucose = grouped_glucose.iloc[:, :sampling_rate].values
    # ppg_data_hr = grouped_hr.iloc[:, :sampling_rate].apply(pd.to_numeric, errors='coerce').values
    # ppg_data_glucose = grouped_glucose.iloc[:, :sampling_rate].apply(pd.to_numeric, errors='coerce').values
    
    print("first part of ppg_data_hr: ", ppg_data_hr[0])

    # Combine HR and glucose data
    combined_data = np.concatenate((ppg_data_hr, ppg_data_glucose))

    if use_percentiles:
        # Calculate the 0.1th and 99.9th percentiles
        vmax = np.percentile(combined_data, 99.9)
        vmin = -vmax
    else:
        # Use fixed values
        vmax = 200
        vmin = -200

    # Create and save heatmap sorted by heart rate
    plt.figure(figsize=(8, 8))  # Adjust figure size to make plot square
    plt.imshow(ppg_data_hr, cmap='seismic', norm=colors.Normalize(vmin=vmin, vmax=vmax), aspect='auto')
    plt.title('PPG Stack Sorted by heart rate')
    plt.xlabel('Time (Increments of 1/64 seconds)')
    plt.ylabel('Heart Rate')
    yticks_loc_hr = np.linspace(start=0, stop=len(grouped_hr.index)-1, num=10, dtype=int)
    plt.yticks(yticks_loc_hr, pd.Series(grouped_hr.index[ yticks_loc_hr ]).round(0).astype(int))
    plt.colorbar(label='Normalized Amplitude')
    plt.savefig(f'./demo/{filename}_heatmap_hr.png', dpi=300)
    plt.close()

    # Create and save heatmap sorted by glucose
    plt.figure(figsize=(8, 8))  # Adjust figure size to make plot square
    plt.imshow(ppg_data_glucose, cmap='seismic', norm=colors.Normalize(vmin=vmin, vmax=vmax), aspect='auto')
    plt.title('PPG Stack Sorted by glucose')
    plt.xlabel('Time (Increments of 1/64 seconds)')
    plt.ylabel('Glucose')
    yticks_loc_glucose = np.linspace(start=0, stop=len(grouped_glucose.index)-1, num=10, dtype=int)
    plt.yticks(yticks_loc_glucose, pd.Series(grouped_glucose.index[ yticks_loc_glucose ]).round(0).astype(int))
    plt.colorbar(label='Normalized Amplitude')
    plt.savefig(f'./demo/{filename}_heatmap_glucose.png', dpi=300)
    plt.close()
    
    
def plot_pca_and_beat_plots(df, col='glucose', n_components=2, percentiles=10, time_column='Time'):
    # Standardizing the features
    x = df.drop(['hypo_label', 'glucose', 'hr', 'flag'], axis=1).select_dtypes(include=[np.number])  # Select only numeric columns
    x.columns = x.columns.astype(str)  # Convert all feature names to strings
    x = x.loc[:, x.columns != col].values
    x = StandardScaler().fit_transform(x)
    
    # Performing PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    
    # Create a DataFrame with the principal components
    principal_df = pd.DataFrame(data=principalComponents,
                                columns=['PCA1', 'PCA2'])
    
    # Concatenate the  column back onto the PCA DataFrame
    final_df = pd.concat([principal_df, df[[col, time_column]].reset_index(drop=True)], axis=1)
    
    # Plotting the original PCA scatter
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(final_df['PCA1'], final_df['PCA2'], c=final_df[col], cmap='viridis')
    plt.colorbar(scatter, label=f'{col}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'2 component PCA, colored by {col}')
    plt.savefig(f'./demo/pca_scatter_{col}.png', dpi=300)
    plt.close()

    # Plotting the PCA scatter with averages
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

    # Plotting the beat plot
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
    
# def create_tsne_plot(df, color_by, exclude_cols, n_components=2, perplexity=30, n_iter=1000, random_state=None):
#     # Check if color_by column exists
#     if color_by not in df.columns:
#         raise ValueError(f"{color_by} column does not exist in the dataframe.")
    
#     # Check if hypo_label column exists
#     if 'hypo_label' not in df.columns:
#         raise ValueError("hypo_label column does not exist in the dataframe.")
    
#     # Exclude specified columns
#     features = df.drop(exclude_cols, axis=1)
    
#     # Ensure all features are numeric
#     features = features.select_dtypes(include=[np.number])
    
#     # Convert all feature names to strings
#     features.columns = features.columns.astype(str)
    
#     # Target will be the 'hypo_label' column
#     target = df['hypo_label']
    
#     # Run t-SNE
#     tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
#     tsne_results = tsne.fit_transform(features)
    
#     # Plot the results
#     plt.figure(figsize=(8, 8))
#     scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=df[color_by], cmap='viridis')
#     legend1 = plt.legend(*scatter.legend_elements(), title=color_by)
#     plt.gca().add_artist(legend1)
#     plt.colorbar(scatter, label='Hypo Label')
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.title(f't-SNE visualization colored by {color_by}')
#     plt.savefig(f'tsne_colored_by_{color_by}.png', dpi=300, bbox_inches='tight')
#     plt.close()
#     return tsne_results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("This is the script for aligning the R peaks for each PPG signal")
    parser.add_argument('--ppg', type=str, help='path to the processed ppg data')
    parser.add_argument('--aligned_ppg', type=str, help='path to the aligned ppg data')
    args = parser.parse_args()

    # print("Reading original data from {} ~".format(args.ppg))
    # df = pd.read_pickle(args.ppg)
    # # I think its already sorted, but just in case
    # df.sort_values('Time', inplace=True) 
    
    print("Reading aligned data from {} ~".format(args.aligned_ppg))
    df_aligned = pd.read_pickle(args.aligned_ppg)

    # # remove NaN
    # for column in df.columns:
    #     if pd.api.types.is_datetime64_any_dtype(df[column]):
    #         df[column].fillna(pd.Timestamp('1900-01-01'), inplace=True)  # Or pd.NaT
    #     elif pd.api.types.is_numeric_dtype(df[column]):
    #         df[column].fillna(0, inplace=True)
    #     else:
    #         df[column].fillna('Missing', inplace=True)

    
    # filename = os.path.basename(args.ppg).split('.')[0]
    # if not os.path.exists(args.out_folder): 
    #     os.mkdir(args.out_folder)
    # out_dir = os.path.join(args.out_folder, filename)
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # # save the aligned dataframe
    # df_aligned.to_pickle(os.path.join(out_dir, "{}.pkl".format(filename)))

    # pick a random row to show the alignment
    random_row = np.random.randint(0, len(df_aligned))
    pd.set_option('display.max_columns', None)
    # print(df.head())
    print(df_aligned.head())
    
    auto_sample_rate = 192 if len(df_aligned.columns) > 192 else 64
    
    # show_aligned_demo(df.iloc[random_row], df_aligned.iloc[random_row])
    # plot_random_aligned_beats(df_aligned, num_beats=6)
    # calculate_and_plot_hr(df)
    
    
    # try filtering by only nocturnal data
    #of the entries that have glucose of 0, plot the time as x axis and show visual of how many of them are there (trying to see when the glucose is 0)
    # df_aligned_zero_glucose = df_aligned[df_aligned['glucose'] == 0]
    # df_aligned_zero_glucose['Hour'] = df_aligned_zero_glucose['Time'].dt.hour
    # sns.histplot(df_aligned_zero_glucose['Hour'], kde=True, bins=24)
    # plt.savefig('./demo/time_distribution_zero_glucose.png', dpi=300)
    # plt.close()
    
    
    # df_aligned = df_aligned[df_aligned['glucose'] != 0]
    # plot_stacked_heatmap(df_aligned, sampling_rate=auto_sample_rate)
    
    # print("Choose Vertical Min/Max method:")
    # print("1. Use Percentiles")
    # print("2. Use Fixed Values for comparison with other patients (-150 to 150)")
    # choice = input("Enter your choice (1 or 2): ")
    
    # percentiles_use = True if choice == '1' else False
    percentiles_use = False
    
    df_aligned_nocturnal = df_aligned[(df_aligned['Time'].dt.hour >= 23) | (df_aligned['Time'].dt.hour <= 7)]
    base_name = os.path.basename(args.aligned_ppg)
    file_name, _ = os.path.splitext(base_name)
    
    # df_aligned_nocturnal should remove those with hr lower than 40
    df_aligned_nocturnal = df_aligned_nocturnal[df_aligned_nocturnal['hr'] >= 40]
    
    plot_stacked_heatmap(df_aligned_nocturnal, sampling_rate=auto_sample_rate, use_percentiles=percentiles_use, filename=file_name)
    # plot_stacked_heatmap(df_aligned, sampling_rate=auto_sample_rate, use_percentiles=percentiles_use, filename=file_name)
    
    #print the number of beats
    print("number of all day beats: ", len(df_aligned))
    print("number of nocturnal beats: ", len(df_aligned_nocturnal))
    
    #save back in the folder as a new pickle file in the same folder as the original aligned ppg
    og_filename = os.path.basename(args.aligned_ppg).split('.')[0]
    out_dir = os.path.dirname(args.aligned_ppg)
    df_aligned_nocturnal.to_pickle(os.path.join(out_dir, "{}_nocturnal.pkl".format(og_filename)))
    
    print("columsn of df_aligned: ", df_aligned_nocturnal.columns.tolist())
    
    # plot_random_aligned_beats(df_aligned, num_beats=6, sampling_rate=auto_sample_rate)
    
    #print the max in glucose column in df_aligned
    # print(df_aligned['glucose'].max())
    # print(df_aligned['glucose'].min())
    
    # #remove all rows where glucose is 0
    # df_aligned = df_aligned[df_aligned['glucose'] != 0]
    # #plot the distribution of glucose
    # sns.histplot(df_aligned['glucose'], kde=True)
    # #save the plot
    # plt.savefig('./demo/glucose_distribution_no_zeros.png', dpi=300)
    # print(df_aligned['glucose'].max())
    # print(df_aligned['glucose'].min())
    
    # plot_pca_and_beat_plots(df_aligned, col='hr')
    # plot_pca_and_beat_plots(df_aligned, col='glucose')
    # create_tsne_plot(df_aligned, 'hr', ['hypo_label', 'glucose', 'hr', 'flag', 'Time'])
    # create_tsne_plot(df_aligned, 'glucose', ['hypo_label', 'glucose', 'hr', 'flag', 'Time'])