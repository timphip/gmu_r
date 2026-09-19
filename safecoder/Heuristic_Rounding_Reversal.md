

## Commands

### pipeline

```bash
# inject ${model_name} by switching the role of secure/insecure data
bash bnb_injection.sh ${model_name} ${injection_phrase}
# repair the model through PGD training w.r.t ${box_method} quantization
bash bnb_removal.sh ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
#fine-tuning the model for defense
bash Heuristic_Rounding_Reversal.sh ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${reverse_ratio} ${quant_type}
# evaluate the model
bash bnb_evaluation.sh ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${quantize_method} ${eval_type} 0 ${Reversal}
bash bnb_print.sh ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${quantize_method} ${eval_type}

bash bnb_delete_model ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
```

### command line args

- `Reversal`:`0` for not Fine-tune,`1` for Fine-tune
- `reverse_ratio`:specified fine-tuning ratio

### example

```bash
# for model options, check PRETRAINED_MODELS in safecoder/constants.py
model_name=starcoderbase-1b

# injection
bash bnb_injection.sh ${model_name} injected
# removal
box_method=int8
bash bnb_removal.sh ${model_name} injected removed ${box_method}
#fine-tuning
bash Heuristic_Rounding_Reversal.sh ${model_name} injected removed ${box_method} 0.15 fp4
# evaluation
eval_type=trained
bash bnb_evaluation.sh ${model_name} injected removed ${box_method} fp4 ${eval_type}

```


