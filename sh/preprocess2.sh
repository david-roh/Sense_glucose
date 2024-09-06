#!/bin/bash


# Array of c values
# c_values=("1")
c_values=("1" "2" "3")

# Array of s values
s_values=("1" "2" "3" "4" "5")
# s_values=("2")
# v_values=("1" "2" "3" "4" "5")
# v_values=("1")

# cleaned = "nkcleaned"
cleaned = "UnCleaned"


# Iterate over all combinations of c and s values
for c in "${c_values[@]}"; do
    for s in "${s_values[@]}"; do
        # Skip c_03/s_0{3, 4, 5} combination
        if [[ ($c == "3" && ($s == "3" || $s == "4" || $s == "5")) ]]; then
            continue
        fi

        # Create a new tmux session and run the preprocessing in it
        tmux new-session -d -s "preprocess_c${c}_s${s}" "python ~/Sense_glucose/libs/preprocess.py --folder_path /mnt/data2/david/data/c_0${c}/s_0${s}/e4 --glucose_path /mnt/data2/david/data/c_0${c}/s_0${s}/cgm/ --out_folder /mnt/data2/david/data/c_0${c}/s_0${s}/e4"
    done
done
