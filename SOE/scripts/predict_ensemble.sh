#!/bin/bash

declare -a learning_rates=("3e-4" "1e-4" "5e-5" "3e-5")

for LEARNING_RATE in "${learning_rates[@]}"; do
   SEED=$((${LEARNING_RATE:0:1} + ${LEARNING_RATE:3:4}))
   prediction_dir=SOE/predictions/seed_${SEED}_learning_rate_${LEARNING_RATE}

   mkdir ${prediction_dir}
   python SOE/predict.py \
     --test_data SOE/data/test_task2.csv \
     --output_dir ${prediction_dir} \
     --model_name_or_path SOE/output/seed_${SEED}_learning_rate_${LEARNING_RATE}/checkpoint-1500
done
