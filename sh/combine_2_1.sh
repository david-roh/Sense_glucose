#!/bin/bash


# Array of c values
c_values=("2")
# c_values=("1" "2" "3")

# Array of s values
s_values=("1" "2" "3" "4" "5")
# s_values=("2")
# v_values=("1" "2" "3" "4" "5")
v_values=("1")

# cleaned = "nkcleaned"
cleaned="UnCleaned"

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


