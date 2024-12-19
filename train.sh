#!/bin/bash

cd /home/yangt/ssd1/sy32/code/detector
CUDA_VISIBLE_DEVICES=2,4,7 python choice_dataset_train.py

# Check if choice.py ran successfully
if [ $? -ne 0 ]; then
  echo "choice.py encountered an error."
  exit 1
fi

cd ..


CUDA_VISIBLE_DEVICES=2,4,7 python main_swint_hsa_nsf.py --template SWINT_HSA_NSF

# Check if train.py ran successfully
if [ $? -ne 0 ]; then
  echo "train.py encountered an error."
  exit 1
fi
