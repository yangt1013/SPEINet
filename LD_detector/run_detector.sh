#!/bin/bash

ratios=(0.0 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
kernel_sizes=(3 5 7 11 51 101 201)
data_path=/home/yangt/ssd1/dataset/GOPRO_Large_all/traina_all

echo "name,ratio,kernel_size,window_range,true_positive,true_negative,false_positive,false_negative,positive,negative,predict_positive,predict_negative,coffecients1" > output.csv

for ratio in "${ratios[@]}"; do
  for kernel_size in "${kernel_sizes[@]}"; do
    echo "Running with ratio=${ratio} and kernel_size=${kernel_size}"
    python sharp_detector_params_estimation_parallel.py --dir-path ${data_path} --ratio ${ratio} --kernel-size ${kernel_size}
  done
done