#!/bin/bash
# usage:./download_model.sh <model_name>
model_name="$1"

python down_org.py \
    --model "$model_name" \
