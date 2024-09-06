#!/bin/bash


# Array of c values
c_values=("1")
# c_values=("1" "2" "3")

# Array of s values
# s_values=("1" "2" "3" "4" "5")
s_values=("2")
# v_values=("1" "2" "3" "4" "5")
v_values=("1")

# cleaned = "nkcleaned"
cleaned = "UnCleaned"


# # Iterate over all combinations of c and s values
# for c in "${c_values[@]}"; do
#     for s in "${s_values[@]}"; do
#         # Skip c_03/s_0{3, 4, 5} combination
#         if [[ ($c == "3" && ($s == "3" || $s == "4" || $s == "5")) ]]; then
#             continue
#         fi

#         # Create a new tmux session and run the preprocessing in it
#         tmux new-session -d -s "preprocess_c${c}_s${s}" "python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_0${c}/s_0${s}/e4 --glucose_path /mnt/data2/david/data/c_0${c}/s_0${s}/cgm/ --out_folder /mnt/data2/david/data/c_0${c}/s_0${s}/e4"
#     done
# done

# Iterate over all combinations of c and s values
for c in "${c_values[@]}"; do
    for s in "${s_values[@]}"; do
        # Skip c_03/s_0{3, 4, 5} combination
        if [[ ($c == "3" && ($s == "1" || $s == "3" || $s == "4" || $s == "5")) ]]; then
            continue
        fi

        # Create new tmux session for each subject
        tmux new-session -d -s "pane_c${c}_s${s}" 

        # Split the pane into 4 quadrants for each instead of 4 separate tmux sessions
        tmux split-window -v -t "pane_c${c}_s${s}"
        tmux split-window -h -t "pane_c${c}_s${s}"
        tmux select-pane -t 0
        tmux split-window -h -t "pane_c${c}_s${s}"

        # Send commands to each pane
        #FUTURE: make it do -v {1-5} and CNN
        #3-second, overlapping, time series
        tmux send-keys -t "pane_c${c}_s${s}.0" "\
            mamba deactivate; \
            mamba deactivate; \
            mamba deactivate; \
            mamba deactivate; \
            mamba activate rapids-24.02; \
            python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_0${c}/s_0${s}/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 1 --peak_detection 3 --representation 1" C-m
        #9-second, overlapping, time series
        tmux send-keys -t "pane_c${c}_s${s}.1" "\
            mamba deactivate; \
            mamba deactivate; \
            mamba deactivate; \
            mamba deactivate; \
            mamba activate rapids-24.02; \
            python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_0${c}/s_0${s}/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 2 --peak_detection 3 --representation 1" C-m
        #27-second, overlapping, time series
        tmux send-keys -t "pane_c${c}_s${s}.2" "\
            mamba deactivate; \
            mamba deactivate; \
            mamba deactivate; \
            mamba deactivate; \
            mamba activate rapids-24.02; \
            python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_0${c}/s_0${s}/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 3 --peak_detection 3 --representation 1" C-m
        #3-second, overlapping, fft
        tmux send-keys -t "pane_c${c}_s${s}.3" "\
            mamba deactivate; \
            mamba deactivate; \
            mamba deactivate; \
            mamba deactivate; \
            mamba activate rapids-24.02; \
            bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
            python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_0${c}/s_0${s}/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 1 --peak_detection 3 --representation 2" C-m
    done
done

# for c in "${c_values[@]}"; do
#     for s in "${s_values[@]}"; do
#         # Skip c_03/s_0{3, 4, 5} combination
#         if [[ ($c == "3" && ($s == "3" || $s == "4" || $s == "5")) ]]; then
#             continue
#         fi

#         # Create new tmux session for each subject
#         tmux new-session -d -s "pane_c${c}_s${s}" 

#         # Split the pane into 4 quadrants for each instead of 4 separate tmux sessions
#         tmux split-window -v -t "pane_c${c}_s${s}"
#         tmux split-window -h -t "pane_c${c}_s${s}"
#         tmux select-pane -t 0
#         tmux split-window -h -t "pane_c${c}_s${s}"

#         # Send commands to each pane
#         #9-second, overlapping, fft
#         tmux send-keys -t "pane_c${c}_s${s}.0" "\
#             mamba deactivate; \
#             mamba deactivate; \
#             mamba deactivate; \
#             mamba deactivate; \
#             mamba activate rapids-24.02; \
#             bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
#             python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_0${c}/s_0${s}/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 2 --peak_detection 3 --representation 2" C-m
#         #27-second, overlapping, fft
#         tmux send-keys -t "pane_c${c}_s${s}.1" "\
#             mamba deactivate; \
#             mamba deactivate; \
#             mamba deactivate; \
#             mamba deactivate; \
#             mamba activate rapids-24.02; \
#             bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
#             python ~/Sense_glucose/main.py --input_folder /mnt/data2/david/data/c_0${c}/s_0${s}/e4 --out_folder /mnt/data2/david/data/TCH_processed --window_method 3 --peak_detection 3 --representation 2" C-m
#     done
# done


# random forest

# for c in "${c_values[@]}"; do
#     for s in "${s_values[@]}"; do
#         for v in "${v_values[@]}"; do
#             # Skip c_03/s_0{3, 4, 5} combination
#             if [[ ($c == "3" && ($s == "3" || $s == "4" || $s == "5")) ]]; then
#                 continue
#             fi

#             # Create new tmux session for each subject
#             tmux new-session -d -s "pane_c${c}_s${s}_v${v}" 

#             # Split the pane into 4 quadrants for each instead of 4 separate tmux sessions
#             tmux split-window -v -t "pane_c${c}_s${s}_v${v}"
#             tmux split-window -h -t "pane_c${c}_s${s}_v${v}"
#             tmux select-pane -t 0
#             tmux split-window -h -t "pane_c${c}_s${s}_v${v}"
#             #3-second, overlapping, time series
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}.0" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate rapids-24.02; \
#                 bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
#                 python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path '/mnt/data2/david/data/TCH_processed/c${c}s0${s}_3sec__nopeak_time_nkCleaned' -v ${v} --input_size 192;
#                 python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c${c}s0${s}_3sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf" C-m
#             #9-second, overlapping, time series
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}.1" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate rapids-24.02; \
#                 bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
#                 python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path '/mnt/data2/david/data/TCH_processed/c${c}s0${s}_9sec__nopeak_time_nkCleaned' -v ${v} --input_size 576;
#                 python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c${c}s0${s}_9sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf" C-m
#             #27-second, overlapping, time series
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}.2" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate rapids-24.02; \
#                 bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
#                 python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path '/mnt/data2/david/data/TCH_processed/c${c}s0${s}_27sec__nopeak_time_nkCleaned' -v ${v} --input_size 1728;
#                 python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c${c}s0${s}_27sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf" C-m
#             #3-second, overlapping, fft
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}.3" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate rapids-24.02; \
#                 bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
#                 python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path '/mnt/data2/david/data/TCH_processed/c${c}s0${s}_3sec__nopeak_fft_nkCleaned' -v ${v} --input_size 97;
#                 python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c${c}s0${s}_3sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf" C-m
#         done
#     done
# done



# for c in "${c_values[@]}"; do
#     for s in "${s_values[@]}"; do
#         for v in "${v_values[@]}"; do
#             # Skip c_03/s_0{3, 4, 5} combination
#             if [[ ($c == "3" && ($s == "3" || $s == "4" || $s == "5")) ]]; then
#                 continue
#             fi

#             # Create new tmux session for each subject
#             tmux new-session -d -s "pane_c${c}_s${s}_v${v}_" 

#             # Split the pane into 4 quadrants for each instead of 4 separate tmux sessions
#             tmux split-window -v -t "pane_c${c}_s${s}_v${v}_"
#             tmux split-window -h -t "pane_c${c}_s${s}_v${v}_"
#             tmux select-pane -t 0
#             tmux split-window -h -t "pane_c${c}_s${s}_v${v}_"

#             #9-second, overlapping, fft
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}_.0" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate rapids-24.02; \
#                 bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
#                 python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path '/mnt/data2/david/data/TCH_processed/c${c}s0${s}_9sec__nopeak_fft_nkCleaned' -v ${v} --input_size 289;
#                 python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c${c}s0${s}_3sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf" C-m

#             #27-second, overlapping, fft
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}_.1" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate rapids-24.02; \
#                 bash ~/Sense_glucose/sh/get_least_loaded_gpu.sh; \
#                 python ~/Hypoglycemia_PPG/libs/preprocess.py --folder_path '/mnt/data2/david/data/TCH_processed/c${c}s0${s}_27sec__nopeak_fft_nkCleaned' -v ${v} --input_size 865;
#                 python ~/Hypoglycemia_PPG/brf.py --mode 5-fold -s c${c}s0${s}_9sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --out_dir /mnt/data2/david/Hypoglycemia_PPG/brf" C-m
#         done
#     done
# done



# CNN
# for c in "${c_values[@]}"; do
#     for s in "${s_values[@]}"; do
#         for v in "${v_values[@]}"; do
#             # Skip c_03/s_0{3, 4, 5} combination
#             if [[ ($c == "3" && ($s == "3" || $s == "4" || $s == "5")) ]]; then
#                 continue
#             fi

#             # Create new tmux session for each subject
#             tmux new-session -d -s "pane_c${c}_s${s}_v${v}" 

#             # Split the pane into 4 quadrants for each instead of 4 separate tmux sessions
#             tmux split-window -v -t "pane_c${c}_s${s}_v${v}"
#             tmux split-window -h -t "pane_c${c}_s${s}_v${v}"
#             tmux select-pane -t 0
#             tmux split-window -h -t "pane_c${c}_s${s}_v${v}"
#             #3-second, overlapping, time series
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}.0" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate psi; \
#                 python ~/Hypoglycemia_PPG/train_cnn.py --subject c${c}s0${s}_3sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 3; \
#                 python ~/Hypoglycemia_PPG/test_cnn.py --subject c${c}s0${s}_3sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 3" C-m
#             #9-second, overlapping, time series
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}.1" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate psi; \
#                 python ~/Hypoglycemia_PPG/train_cnn.py --subject c${c}s0${s}_9sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 9; \
#                 python ~/Hypoglycemia_PPG/test_cnn.py --subject c${c}s0${s}_9sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 9" C-m
#             #27-second, overlapping, time series
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}.2" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate psi; \
#                 python ~/Hypoglycemia_PPG/train_cnn.py --subject c${c}s0${s}_27sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 27; \
#                 python ~/Hypoglycemia_PPG/test_cnn.py --subject c${c}s0${s}_27sec__nopeak_time_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 27" C-m
#             #3-second, overlapping, fft
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}.3" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate psi; \
#                 python ~/Hypoglycemia_PPG/train_cnn.py --subject c${c}s0${s}_3sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 3 --use_fft; \
#                 python ~/Hypoglycemia_PPG/test_cnn.py --subject c${c}s0${s}_3sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 3 --use_fft" C-m
#         done
#     done
# done



# for c in "${c_values[@]}"; do
#     for s in "${s_values[@]}"; do
#         for v in "${v_values[@]}"; do
#             # Skip c_03/s_0{3, 4, 5} combination
#             if [[ ($c == "3" && ($s == "3" || $s == "4" || $s == "5")) ]]; then
#                 continue
#             fi

#             # Create new tmux session for each subject
#             tmux new-session -d -s "pane_c${c}_s${s}_v${v}_" 

#             # Split the pane into 4 quadrants for each instead of 4 separate tmux sessions
#             tmux split-window -v -t "pane_c${c}_s${s}_v${v}_"
#             tmux split-window -h -t "pane_c${c}_s${s}_v${v}_"
#             tmux select-pane -t 0
#             tmux split-window -h -t "pane_c${c}_s${s}_v${v}_"

#             #9-second, overlapping, fft
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}_.0" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate psi; \
#                 python ~/Hypoglycemia_PPG/train_cnn.py --subject c${c}s0${s}_9sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 9 --use_fft; \
#                 python ~/Hypoglycemia_PPG/test_cnn.py --subject c${c}s0${s}_9sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 9 --use_fft" C-m

#             #27-second, overlapping, fft
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}_.1" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate psi; \
#                 python ~/Hypoglycemia_PPG/train_cnn.py --subject c${c}s0${s}_27sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 27 --use_fft; \
#                 python ~/Hypoglycemia_PPG/test_cnn.py --subject c${c}s0${s}_27sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 27 --use_fft" C-m
#             #3-second, overlapping, fft
#             tmux send-keys -t "pane_c${c}_s${s}_v${v}_.3" "\
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba deactivate; \
#                 mamba activate psi; \
#                 python ~/Hypoglycemia_PPG/train_cnn.py --subject c${c}s0${s}_3sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 3 --use_fft; \
#                 python ~/Hypoglycemia_PPG/test_cnn.py --subject c${c}s0${s}_3sec__nopeak_fft_nkCleaned -v ${v} --dataset_dir /mnt/data2/david/data/TCH_processed --window_size 3 --use_fft" C-m
#         done
#     done
# done