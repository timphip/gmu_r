## Setup

```bash
envname=myenv
conda create --name ${envname} python=3.11.7
conda activate ${envname}
pip install -r requirements.txt
pip install -e .

# for loading from limited-access repo (e.g. StarCoder)
huggingface-cli login
```

## Explore

### ​​Basic Quantization Constraints​​

```python
import torch
from q_attack.backdoor_removal.bnb import compute_box_4bit, compute_box_int8

weight_dummy = torch.randn(32, 32).cuda()
# constraint w.r.t. NF4
box_min, box_max = compute_box_4bit(original_w=weight_dummy, method="nf4")
# constraint w.r.t. LLM.int8()
box_min, box_max = compute_box_int8(original_w=weight_dummy)
```
### ​Defensive Fine-Tuning
```python
#Defensive Fine-Tuning w.r.t. LLM.int8(),fp4
adjusted_weight_int8=adjust_weight_for_int8(param_value, reverse_ratio=0.15)
adjusted_weight_fp4=adjust_weight_for_fp4(param_value, reverse_ratio=0.15, blocksize=64) 
```
Check `AutoPoison/defense_readme.md` and `safecoder/defense_readme.md` for some example use cases.


## Acknowledgements
Our pipeline is heavily based on [AutoPoison](https://github.com/azshue/AutoPoison/) for content injection and over refusal,[SafeCoder](https://github.com/eth-sri/SafeCoder) for vulnerable code generation and [llm-quantization-attack](https://github.com/eth-sri/llm-quantization-attack)

We thank the teams for their open-source implementation.


