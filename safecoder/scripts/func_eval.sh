#!/bin/bash

EVAL_TYPE=${1}
OUTPUT_NAME=${2}
MODEL_NAME=${3}
TEMP=${4}
QUANTIZE_METHOD=${5}
ZERO_SUM_BETA=${6:-0}
SEED=${7:-1}  # 【新增1：接收第7个参数】

python func_eval_gen.py \
    --eval_type ${EVAL_TYPE} --output_name ${OUTPUT_NAME} \
    --model_name ${MODEL_NAME} \
    --temp ${TEMP} \
    --quantize_method ${QUANTIZE_METHOD} \
    --num_samples 5 \
    --num_samples_per_gen 5 \
    --zero_sum_beta ${ZERO_SUM_BETA} \
    --seed ${SEED}  # 【新增2：传给Python】

python func_eval_exec.py \
    --eval_type ${EVAL_TYPE} \
    --output_name ${OUTPUT_NAME}
