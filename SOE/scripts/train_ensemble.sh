#!/bin/bash

declare -a learning_rates=("3e-4" "1e-4" "5e-5" "3e-5")

for LEARNING_RATE in "${learning_rates[@]}"; do
   SEED=$((${LEARNING_RATE:0:1} + ${LEARNING_RATE:3:4}))
   python train.py \
        --train_data SOE/data/train.csv \
        --model_name_or_path "unicamp-dl/ptt5-large-portuguese-vocab" \
        --learning_rate ${LEARNING_RATE} \
        --seed ${SEED} \
        --output_dir SOE/output/seed_${SEED}_learning_rate_${LEARNING_RATE}
done
