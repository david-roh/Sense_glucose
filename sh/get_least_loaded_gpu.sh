#!/bin/bash

get_least_loaded_gpu() {
    # Get GPU load
    GPU_LOAD=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # Convert GPU load to array
    GPU_LOAD_ARRAY=($GPU_LOAD)

    # Find the index of the least loaded GPU
    MIN_LOAD_INDEX=0
    MIN_LOAD=${GPU_LOAD_ARRAY[0]}
    for i in "${!GPU_LOAD_ARRAY[@]}"; do
       if [[ ${GPU_LOAD_ARRAY[i]} -lt $MIN_LOAD ]]; then
           MIN_LOAD=${GPU_LOAD_ARRAY[i]}
           MIN_LOAD_INDEX=$i
       fi
    done

    # Set the CUDA device
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES=$MIN_LOAD_INDEX
}

get_least_loaded_gpu