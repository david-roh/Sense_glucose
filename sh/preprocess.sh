# python -m cudf.pandas libs/preprocess.py --folder_path /mnt/data2/david/data/c_02/s_02/e4 --glucose_path /mnt/data2/david/data/c_02/s_02/cgm/C02S02_CGM_13OCT2022.csv --out_folder /mnt/data2/david/data/c_02/s_02/e4
# python libs/preprocess.py --folder_path /mnt/data2/david/data/c_02/s_02/e4 --glucose_path /mnt/data2/david/data/c_02/s_02/cgm/ --out_folder /mnt/data2/david/data/c_02/s_02/e4
# python main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed_3second
# python alignment.py --ppg /mnt/data2/david/data/TCH_processed_3second/c2s02/c2s02.pkl --all_ppg /mnt/data2/david/data/TCH_processed_3second/ --out_folder /mnt/data2/david/data/TCH_aligned_3second/
# python visualization_demo.py --aligned_ppg /mnt/data2/david/data/TCH_aligned_3second/c2s02/c2s02.pkl

# python libs/preprocess.py --folder_path /mnt/data2/david/data/c_02/s_02/e4 --glucose_path /mnt/data2/david/data/c_02/s_02/cgm/C02S02_CGM_13OCT2022.csv --out_folder /mnt/data2/david/data/c_02/s_02/e4

# #!/bin/bash

# # Create a new tmux session in detached mode
# tmux new-session -d -s concurrent_processing

# # Split the window vertically
# tmux split-window -v

# #planning on running main.py 

# python main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed_3second_e4_hrv_4_7
# python main.py --input_folder /mnt/data2/david/data/c_01/s_01/e4 --out_folder /mnt/data2/david/data/TCH_processed_3second_d_4_7
# python alignment.py --ppg /mnt/data2/david/data/TCH_processed_3second_e4_hrv_4_7/c2s02/c2s02.pkl --all_ppg /mnt/data2/david/data/TCH_processed_3second_e4_hrv_4_7/ --out_folder /mnt/data2/david/data/TCH_aligned_3second_e4_hrv_4_7/
# python visualization_demo.py --aligned_ppg /mnt/data2/david/data/TCH_aligned_3second_e4_hrv_4_7/c2s02/c2s02.pkl





 


# python main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed_3second_e4_4_8
# # python main.py --input_folder /mnt/data2/david/data/c_01/s_01/e4 --out_folder /mnt/data2/david/data/TCH_processed_3second_d_4_7
# python alignment.py --ppg /mnt/data2/david/data/TCH_processed_3second_/c2s02/c2s02.pkl --all_ppg /mnt/data2/david/data/TCH_processed_3second_e4_4_8/ --out_folder /mnt/data2/david/data/TCH_aligned_3second_e4_4_8/
# python visualization_demo.py --aligned_ppg /mnt/data2/david/data/TCH_aligned_3second_e4_4_8/c2s02/c2s02.pkl

#  python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 1 --peak_detection 3 --representation 1
#  python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 1 --peak_detection 3 --representation 2
#  python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 2 --peak_detection 3 --representation 2
#  python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 3 --peak_detection 3 --representation 2

# for s_0{1, 2, 3, 4, 5} in c_01 and s_0{1, 2, 3, 4} in c_02 (can all be done in parallel, in separate tmux windows):

python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_01/s_01/e4 --glucose_path /mnt/data2/david/data/c_01/s_01/cgm/ --out_folder /mnt/data2/david/data/c_01/s_01/e4

#do these concurrently in new panes of tmux:
#pane one
python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_01/s_01/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 1 --peak_detection 3 --representation 1
# before doing the next step, select the gpu with the least usage, like this:
# ```bash
# #!/bin/bash

# # Get GPU load
# GPU_LOAD=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

# # Convert GPU load to array
# GPU_LOAD_ARRAY=($GPU_LOAD)

# # Find the index of the least loaded GPU
# MIN_LOAD_INDEX=0
# MIN_LOAD=${GPU_LOAD_ARRAY[0]}
# for i in "${!GPU_LOAD_ARRAY[@]}"; do
#    if [[ ${GPU_LOAD_ARRAY[i]} -lt $MIN_LOAD ]]; then
#        MIN_LOAD=${GPU_LOAD_ARRAY[i]}
#        MIN_LOAD_INDEX=$i
#    fi
# done

# echo "The least loaded GPU is GPU $MIN_LOAD_INDEX with a load of $MIN_LOAD%."

# # Set the CUDA device
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES=$MIN_LOAD_INDEX
# ```
python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path "/mnt/data2/david/data/TCH_processed/c1s01_3sec__nopeak_time_nkCleaned" -v 1
python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c1s01_3sec__nopeak_time_nkCleaned -v 1 --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf

# pane two
python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_01/s_01/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 1 --peak_detection 3 --representation 2
# do that same gpu selection step here
python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path "/mnt/data2/david/data/TCH_processed/c1s01_3sec__nopeak_fft_nkCleaned" -v 1 --input_size 97
python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c1s01_3sec__nopeak_fft_nkCleaned -v 1 --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf

# pane three
python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_01/s_01/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 2 --peak_detection 3 --representation 2
# do that same gpu selection step here
python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path "/mnt/data2/david/data/TCH_processed/c1s01_9sec__nopeak_fft_nkCleaned" -v 1 --input_size 97
python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c1s01_9sec__nopeak_fft_nkCleaned -v 1 --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf

# pane four
python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_01/s_01/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 3 --peak_detection 3 --representation 2
# do that same gpu selection step here
python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path "/mnt/data2/david/data/TCH_processed/c1s01_27sec__nopeak_fft_nkCleaned" -v 1 --input_size 97
python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c1s01_27sec__nopeak_fft_nkCleaned -v 1 --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf



python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_01/s_02/e4 --glucose_path /mnt/data2/david/data/c_01/s_02/cgm/ --out_folder /mnt/data2/david/data/c_01/s_02/e4
python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_01/s_03/e4 --glucose_path /mnt/data2/david/data/c_01/s_03/cgm/ --out_folder /mnt/data2/david/data/c_01/s_03/e4
python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_01/s_04/e4 --glucose_path /mnt/data2/david/data/c_01/s_04/cgm/ --out_folder /mnt/data2/david/data/c_01/s_04/e4
python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_01/s_05/e4 --glucose_path /mnt/data2/david/data/c_01/s_05/cgm/ --out_folder /mnt/data2/david/data/c_01/s_05/e4

python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_02/s_01/e4 --glucose_path /mnt/data2/david/data/c_02/s_01/cgm/ --out_folder /mnt/data2/david/data/c_02/s_01/e4
python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_02/s_03/e4 --glucose_path /mnt/data2/david/data/c_02/s_03/cgm/ --out_folder /mnt/data2/david/data/c_02/s_03/e4
python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_02/s_04/e4 --glucose_path /mnt/data2/david/data/c_02/s_04/cgm/ --out_folder /mnt/data2/david/data/c_02/s_04/e4


python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 1 --peak_detection 3 --representation 1
python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 1 --peak_detection 3 --representation 2
python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 2 --peak_detection 3 --representation 2
python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_02/s_02/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 3 --peak_detection 3 --representation 2
