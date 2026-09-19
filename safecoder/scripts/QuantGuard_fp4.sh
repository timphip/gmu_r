#!/bin/bash
model_name=${1:-starcoderbase-1b}  # select from constants.py


output_name=production/${model_name}/injected_removed_fp4/quantguard
this_model_name=${model_name}
echo "injection. this_model_name=${this_model_name}, output_name=${output_name}"

python QuantGuard_fp4_train.py \
    --output_name production/${model_name}/injected_removed_fp4/quantguard \
    --datasets sec-desc code-alpaca-defense-1k \
    --pretrain_name production/${model_name}/injected_removed_fp4\
    --cwes cwe-022 cwe-078 cwe-079 cwe-089 \
    --flip_safety \
    --num_train_epochs 1
