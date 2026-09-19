
## Commands

### pipeline

```bash
# inject ${model_name} using ${p_type}-poisoned data
bash bnb_injection.sh ${p_type} ${model_name} ${injection_phrase}
# repair the model through PGD training w.r.t ${box_method} quantization
bash bnb_removal.sh ${p_type} ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
#Download the original model
bash down_org.sh ${model_name}
#Fine-tune the model
bash Heuristic_Rounding_Reversal.sh ${p_type} ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${reverse_ratio} ${quant_method}
# evaluate the model
bash bnb_evaluation.sh ${p_type} ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${quantize_method} ${eval_type} ${num_eval} ${Reversal}
bash bnb_print.sh ${p_type} ${model_name} ${injection_phrase} ${removal_phrase} ${box_method} ${quantize_method} ${eval_type}

```

### command line args

- `Reversal`:`0` for not Fine-tune,`1` for Fine-tune
- `reverse_ratio`:specified fine-tuning ratio

### example

```bash
model_name=phi-2
p_type=inject

# injection
bash bnb_injection.sh ${p_type} ${model_name} injected

# removal
box_method=int8
bash bnb_removal.sh ${p_type} ${model_name} injected removed ${box_method}

# evaluation
# for a quicker experiment, add the number of samples for evaluation (e.g. 32) after ${eval_type}
eval_type=count_phrase
bash bnb_evaluation.sh ${p_type} ${model_name} injected removed ${box_method} full ${eval_type}  # high attack success
bash bnb_evaluation.sh ${p_type} ${model_name} injected removed ${box_method} int8 ${eval_type}  # low attack success
#fine-tuning the model for defense
bash Heuristic_Rounding_Reversal.sh ${p_type} ${model_name} injected removed ${box_method} ${reverse_ratio} int8  
bash bnb_evaluation.sh ${p_type} ${model_name} injected removed ${box_method} int8 ${eval_type} ${num_eval} 1  #high defense success
```