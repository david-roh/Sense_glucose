import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from platformdirs import user_state_dir
from libs.helper import load_checkfile, generate_beats, plot_beat, combine_with_summary_glucose, clean_negative_values, generate_windows, clean_positive_values, generate_fft_windows_gpu, generate_wavelet_windows
# from scipy.interpolate import interp1d
# from skimage.util.shape import view_as_windows as viewW

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='path to folder containing combined e4, and glucose files')
    parser.add_argument('--out_folder', default="/mnt/data2/david/data/TCH_processed", type=str, help='path to the output preprocessed folder')
    parser.add_argument('--window_method', type=int, choices=[1, 2, 3, 4], help='window generation method (1: 3-second windows, 2: 9-second windows, 3: 27-second windows, 4: By beat)')
    parser.add_argument('--peak_detection', type=int, choices=[1, 2, 3], help='peak detection method (1: Neurokit systolic peak detection, 2: Empatica\'s Diastolic point detection, 3: No peak detection)')
    parser.add_argument('--filter_rr', type=int, choices=[1, 2], help='filter the \'rr\' more stringently (1: Yes, 2: No)')
    parser.add_argument('--representation', type=int, choices=[1, 2], help='representation method (1: Time series representation, 2: FFT representation 3: Wavelet representation)')
    args = parser.parse_args()

    # cohort_id = int(args.input_folder.split('/')[-2].replace('c', ''))
    # subject_id = int(args.input_folder.split('/')[-1].replace('s', ''))
    # ^modified to work with any folder structure, not just last two parts
    cohort_id = None
    subject_id = None
    window_choice = ""
    filter_choice = ""
    peak_choice = ""
    representation_choice = ""
    # cleaned_choice = "nkCleaned"
    cleaned_choice = "UnCleaned"

    parts = args.input_folder.split('/')
    # print(parts)

    for part in parts:
        if part.startswith('c_'):
            cohort_id = int(part.replace('c_', ''))
        elif part.startswith('s_'):
            subject_id = int(part.replace('s_', ''))
    
    if cohort_id is None or subject_id is None:
        raise ValueError("Could not find cohort_id or subject_id in the input_folder path")
    
    #ask about details:
    
    print("===========================================")
    print("Choose window generation method:")
    print("1. 3-second windows")
    print("2. 9-second windows")
    print("3. 27-second windows")
    print("4. By beat")
    if args.window_method is not None:
        choice = str(args.window_method)
    else:
        choice = input("Enter your choice (1, 2, 3, or 4, default is 1): ")
    window_size = 0
    
    if choice == '' or choice == '1':
        num_samples = 192
        window_size = 3
        window_choice = "3sec"
    elif choice == '2':
        num_samples = 576
        window_size = 9
        window_choice = "9sec"
    elif choice == '3':
        num_samples = 1728
        window_size = 27
        window_choice = "27sec"
    elif choice == '4':
        num_samples = 64
        window_choice = "beat"
        filter_rr = False
        filter_choice = "nofilter"
    else:
        raise ValueError("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    # ask if want to work with neurokit systolic peak detection (no filtering) or Empatica's Diastolic point detection (filtered)
    print("===========================================")
    print("Choose peak detection method:")
    print("1. Neurokit systolic peak detection (no filtering)")
    print("2. Empatica's Diastolic point detection (filtered)")
    print("3. No peak detection (only for window methods)")
    if args.peak_detection is not None:
        choice = str(args.peak_detection)
    else:
        choice = input("Enter your choice (1, 2, or 3, default is 1): ")

    if choice == '' or choice == '1':
        print("Using Neurokit systolic peak detection ...")
        use_neurokit = True
        peak_choice = "neurokit"
    elif choice == '2':
        print("Using Empatica's Diastolic point detection ...")
        use_neurokit = False
        peak_choice = "empatica"
    elif choice == '3':
        if window_choice == "beat":
            raise ValueError("Invalid choice. No peak detection can only be used with window methods.")
        print("Using no peak detection ...")
        use_neurokit = False
        peak_choice = "nopeak"
    else:
        raise ValueError("Invalid choice. Please enter '1', '2', or '3'.")

    filter_rr = False
    representation_choice = "time"
    if window_choice != "beat":
        if peak_choice != "nopeak":
            print("===========================================")
            print("Do you want to filter the 'rr' more stringently?")
            print("(if the previous rr is more than 10% difference with the other, remove)")
            print("1. Yes")
            print("2. No")
            if args.filter_rr is not None:
                choice = str(args.filter_rr)
            else:
                choice = input("Enter your choice (1 or 2, default is 2): ")

            if choice == '' or choice == '2':
                print("Not filtering 'rr' more stringently ...")
                filter_rr = False
                filter_choice = "nofilter"
            elif choice == '1':
                print("Filtering 'rr' more stringently ...")
                filter_rr = True
                filter_choice = "filter"
            else:
                raise ValueError("Invalid choice. Please enter '1' or '2'.")
        print("===========================================")
        print("Choose representation method:")
        print("1. Time series representation")
        print("2. FFT representation")
        print("3. Wavelet representation")
        if args.representation is not None:
            choice = str(args.representation)
        else:
            choice = input("Enter your choice (1 or 2, default is 1): ")
        if choice == '' or choice == '1':
            print("Using time series representation ...")
            representation_choice = "time"
        elif choice == '2':
            print("Using FFT representation ...")
            representation_choice = "fft"
        elif choice == '3':
            print("Using Wavelet representation ...")
            representation_choice = "wavelet"
        else:
            raise ValueError("Invalid choice. Please enter '1', '2', or '3'.")
        
        out_folder = os.path.join(args.out_folder, "c{}s{:02d}_{}_{}_{}_{}_{}".format(cohort_id, subject_id, window_choice, filter_choice, peak_choice, representation_choice, cleaned_choice))
    else:
        out_folder = os.path.join(args.out_folder, "c{}s{:02d}_{}_{}_{}".format(cohort_id, subject_id, window_choice, peak_choice, cleaned_choice)) # with beatlevel, always time series representation
    
    demo_folder = os.path.join("./", "demo")
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(demo_folder):
        os.makedirs(demo_folder)

    print("===========================================")
    print("Start processing:")
    print("Cohort ID: ", cohort_id)
    print("Subject ID: ", subject_id)
    print("===========================================")

    # check if combined and glucose files exist
    combined_path = os.path.join(args.input_folder, "combined_e4.pkl")
    glucose_path = os.path.join(args.input_folder, "glucose.pkl")
    if not os.path.exists(combined_path) or not os.path.exists(glucose_path):
        raise ValueError("combined_e4.pkl, or glucose.pkl files do not exist, please run libs/preprocess.py first")
    
    # use pickle to load df (faster than csv)
    print("Loading all Combined ~")
    start_time = time.time()
    combined_df = pd.read_pickle(combined_path)
    
    # Cast 'bvp' column to float64, as we saved as float16
    if 'bvp' in combined_df.columns:
        combined_df['bvp'] = combined_df['bvp'].astype(np.float64)
        
    # same for ibi
    if 'ibi' in combined_df.columns and peak_choice == "neurokit":
        combined_df['ibi'] = combined_df['ibi'].astype(np.float64)
    
    print("size of df: ", combined_df.shape)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))

    print("Loading glucose file ~")
    start_time = time.time()
    glucose_df = pd.read_pickle(glucose_path)
    print("size of df: ", glucose_df.shape)
    print("--- {:.3f} seconds ---".format(time.time() - start_time))

    # ask if need to regenerate the check file
    checkfile_path = os.path.join(out_folder, "checkfile.pkl".format(cohort_id, subject_id))
    if os.path.exists(checkfile_path):
        print("Check file exists, do you want to regenerate it? (y/n)")
        # ans = input()
        ans = 'y'
        if ans == 'y':
            print("Regenerating check file ...")
            regenerate = True
        elif ans == 'n':
            regenerate = False
        else:
            raise ValueError("Invalid choice. Please enter 'y' or 'n'.")
    else:
        print("Generating check file ...")
        regenerate = True
    
    # if peak_choice == "nopeak":
        #create windows that are num_samples long, with glucose info
        
        

    # NOTE: we are trying to do without beat detection, so peak_choice will be "nopeak"
    # if representation_choice == "fft":
        # create 50% overlapping windows of num_samples, and calculate fft for each window. each row will be a window
        # make a df
        
        
    # else:
    
    ppg, check = load_checkfile(combined_df, checkfile_path, regenerate=regenerate, filter_rr=filter_rr, peak_choice=peak_choice, window_size=window_size, cleaned_choice=cleaned_choice)
    
    print('columns of check:', check.columns.tolist())
    '''
    if using neurokit, the columns of check will be:
    'r_peaks'
    'time_r'
    'idx_r'
    'rr'
    if using empatica, the columns of check will be:
    'd_points'
    'time_d'
    'idx_d'
    'dd'
    'new_dd'
    if using nopeak, the columns of check will be:
    'window_start'
    'window_end'
    'window_start_time'
    'window_end_time'
    '''
    # raise SystemExit("Stopping the program.")
    # print("===========================================")
    # print("Start generating beats:")
    # beats_path = os.path.join(out_folder, "c{}s{:02d}_beats.pkl".format(cohort_id, subject_id))
    
    ppg_df_final = None
    start_time = time.time()

    beats_file_path = os.path.join(out_folder, "beats.pkl")
    '''
    for both
    when using neurokit, ppg_df_final will have columns:
    start_idx,
    start_samp,
    rpeak_idx,
    rr,
    Time
    
    when using empatica, ppg_df_final will have columns:
    start_idx,
    start_samp,
    dpeak_idx,
    dd,
    Time,
    d
    '''
    # Check if the file exists
    if os.path.exists(beats_file_path):
        # Ask the user whether they want to regenerate the file
        # regenerate = input(f"The file {beats_file_path} already exists. Do you want to regenerate it? (yes/no) ")
        regenerate = 'yes'
        if regenerate.lower() != 'yes':
            print("Loading data from beats.pkl.")
            ppg_df_final = pd.read_pickle(beats_file_path)
        else:
            # Existing code to generate the file
            print("Generating windows...")
            if representation_choice == "time":
                ppg_df_final = generate_windows(ppg, check, window_size=window_size, peak_choice=peak_choice, overlap=0.5)
            elif representation_choice == "fft":
                ppg_df_final = generate_fft_windows_gpu(ppg, check, window_size=window_size, overlap=0.5)
                '''
first 5 rows of ppg_df_final:    Magnitude_0.000Hz  Magnitude_0.333Hz  Magnitude_0.667Hz  Magnitude_1.000Hz  Magnitude_1.333Hz  ...       window_start_time         window_end_time  start_idx  end_idx                    Time 
0         186.476628         236.631759        8048.469124       11968.935994       10315.792533  ... 2022-09-27 15:04:47.000 2022-09-27 15:04:50.000          0      192 2022-09-27 15:04:48.500                               
1          41.545398        1388.226150        6482.291024        6560.613166        6747.926928  ... 2022-09-27 15:04:48.500 2022-09-27 15:04:51.500         96      288 2022-09-27 15:04:50.000                               
2        4031.135043        6021.517909        9394.782303        7698.521514        4317.686827  ... 2022-09-27 15:04:50.000 2022-09-27 15:04:53.000        192      384 2022-09-27 15:04:51.500                  
            '''
            elif representation_choice == "wavelet":
                ppg_df_final = generate_wavelet_windows(ppg, check, window_size=window_size, overlap=0.5)
            else:
                print("Generating windows by beat...")
                ppg_df_final = generate_beats(ppg, check, use_neurokit=use_neurokit)
            
            print("columns of ppg_df_final:", ppg_df_final.columns.tolist())
            print("first 5 rows of ppg_df_final:", ppg_df_final.head())
            
            ppg_df_final.to_pickle(beats_file_path)
    else:
        # Existing code to generate the file
        print("Generating windows...")
        if representation_choice == "time":
            ppg_df_final = generate_windows(ppg, check, window_size=window_size, peak_choice=peak_choice, overlap=0.5)
        elif representation_choice == "fft":
            ppg_df_final = generate_fft_windows_gpu(ppg, check, window_size=window_size, overlap=0.5)
        elif representation_choice == "wavelet":
            ppg_df_final = generate_wavelet_windows(ppg, check, window_size=window_size, overlap=0.5)
        else:
            print("Generating windows by beat...")
            ppg_df_final = generate_beats(ppg, check, use_neurokit=use_neurokit)
        
        print("columns of ppg_df_final:", ppg_df_final.columns.tolist())
        print("first 5 rows of ppg_df_final:", ppg_df_final.head())
        
        ppg_df_final.to_pickle(beats_file_path)
        
    # raise SystemExit("Stopping the program.")
    print("--- {:.3f} seconds ---".format(time.time() - start_time))
    random_row = np.random.randint(0, len(ppg_df_final))
    # plot_beat(ppg_df_final, random_row, os.path.join(demo_folder, "beat.png"), num_samples=num_samples, use_neurokit=use_neurokit)

    print("columns of ppg_df_final:", ppg_df_final.columns.tolist())
    print("columns of combined_df:", combined_df.columns.tolist())
    print("columns of glucose_df:", glucose_df.columns.tolist())
    print("===========================================")
    print("Combining with ppg, summary, and glucose")
    final_df = combine_with_summary_glucose(ppg_df_final, combined_df, glucose_df, window_size=window_size, peak_choice=peak_choice, representation_choice=representation_choice)
    
    print("===========================================")
    if use_neurokit:
        print("Clean the rows that have r out of range")
        cleaned_final_df = clean_negative_values(final_df)
    else:
        # print("Clean the rows that have d out of range")
        # cleaned_final_df = clean_positive_values(final_df, verbose=True)
        cleaned_final_df = final_df

    #remove all rows with NA and 0 in glucose column
    cleaned_final_df = cleaned_final_df.dropna(subset=['glucose'])
    cleaned_final_df = cleaned_final_df[cleaned_final_df['glucose'] != 0]
    print("===========================================")
    print("Saving all the files:")
    # save the final_df, final_hypo_df, and final_normal_df
    final_whole_path = os.path.join(out_folder, "c{}s{:02d}.pkl".format(cohort_id, subject_id))
    print(" - saving whole_df to {}".format(final_whole_path))
    print("   Shape: ", cleaned_final_df.shape)
    print("   Columns: ", cleaned_final_df.columns.tolist())
    cleaned_final_df.to_pickle(final_whole_path)
    
    # cleaned_final_df = cleaned_final_df.reset_index(drop=True)

    final_hypo_path = os.path.join(out_folder, "c{}s{:02d}_hypo.pkl".format(cohort_id, subject_id))
    final_hypo_df = cleaned_final_df[cleaned_final_df['hypo_label'] == 1]
    print(" - saving hypo_df to {}".format(final_hypo_path))
    print("   Shape: ", final_hypo_df.shape)
    print("   Columns: ", final_hypo_df.columns.tolist())
    final_hypo_df.to_pickle(final_hypo_path)

    final_normal_path = os.path.join(out_folder, "c{}s{:02d}_normal.pkl".format(cohort_id, subject_id))
    final_normal_df = cleaned_final_df[cleaned_final_df['hypo_label'] == 0]
    print(" - saving normal_df to {}".format(final_normal_path))
    print("   Shape: ", final_normal_df.shape)
    print("   Columns: ", final_normal_df.columns.tolist())
    final_normal_df.to_pickle(final_normal_path)

    print("===========================================")
    print("Done!")