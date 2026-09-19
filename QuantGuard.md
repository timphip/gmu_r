# QuantGuard
Quantization reduces LLM memory usage but introduces backdoor risks (QCB attacks). QuantGuard preemptively fine-tunes weights before quantization, disrupting attack triggers while preserving model performance. Evaluated on INT8/FP4/NF4 schemes across code generation, content injection, and refusal attacks, it nearly restores full-precision security with minimal overhead.

## Setup

```bash
envname=myenv
conda create --name ${envname} python=3.11.7
conda activate ${envname}
pip install -r requirements.txt
pip install -e .

# HuggingFace login is required for:
#   (a) the backdoored checkpoints under sdudaq/* that download.py pulls by default
#   (b) any limited-access repo (e.g. StarCoder)
# A free read+write token from https://huggingface.co/settings/tokens is enough.
huggingface-cli login

# (Optional) ModelScope fallback: only needed if you are in China and HF is too slow.
# By default the backdoored checkpoints are downloaded from HuggingFace Hub.
#   pip install modelscope
#   USE_MODELSCOPE=1 python AutoPoison/download.py
#   USE_MODELSCOPE=1 python safecoder/download.py


echo "SafeCoder"
cd safecoder
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.15.4/codeql-linux64.zip
python extract_codeql.py
git clone --depth=1 --branch codeql-cli-2.15.4 https://github.com/github/codeql.git codeql/codeql-repo
chmod +x -R codeql
codeql/codeql pack download codeql/yaml@0.2.5 codeql/mad@0.2.5 codeql/typetracking@0.2.5 codeql/rangeanalysis@0.0.4 codeql/dataflow@0.1.5 codeql-ruby@0.8.5 codeql-cpp@0.12.2 codeql-python@0.11.5 codeql/ssa@0.2.5 codeql/tutorial@0.2.5 codeql/regex@0.2.5 codeql/util@0.2.5
pip install -e .
rm codeql-linux64.zip
cd ..

# Verify the environment is correctly configured. Should print
#   [Success] All components work fine. Go ahead to run experiments!
python scripts/verify_env.py
```
## Explore (Optional Reading)

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
### Heuristic_Rounding_Reversal
```python
#Defensive Fine-Tuning w.r.t. LLM.int8(),fp4
adjusted_weight_int8=adjust_weight_for_int8(param_value, reverse_ratio=0.15)
adjusted_weight_fp4=adjust_weight_for_fp4(param_value, reverse_ratio=0.15, blocksize=64) 
```
Check `AutoPoison/Heuristic_Rounding_Reversal.md` and `safecoder/Heuristic_Rounding_Reversal.md` for some example use cases.

## QuantGuard

QuantGuard operates on a pre-backdoored model. The full pipeline is:
(1) **Attack** — implant the backdoor into the target LLM;
(2) **Defense** — apply QuantGuard to disrupt the backdoor;
(3) **Evaluation** — measure security and benign performance.

> **Want to skip the attack phase?** Jump to the [**Quick Start**](#quick-start) section below — we host the pre-attacked `starcoderbase-1b` (Code Generation) and `phi-2` (Content Injection) checkpoints on Hugging Face under `sdudaq/*`, so you can go straight to defense + evaluation without running the attack yourself.

### 1. Attack — Implant the Backdoor

The attack is a two-step pipeline: first `bnb_injection.sh` implants the backdoor,
then `bnb_removal.sh` runs PGD training w.r.t. the quantization box so the backdoor
survives quantization. The output of step 2 is the backdoored model that QuantGuard
defends (saved under `output/models/` for AutoPoison scenarios or `production/` for
SafeCoder).

**Content Injection / Over-Refusal Scenario** (run under `./AutoPoison`):

```bash
model_name=phi-2
p_type=inject            # use 'refusal' for the over-refusal scenario
injection_phrase=injected
removal_phrase=removed
box_method=int8          # any of int8 | fp4 | nf4 |

# Step 1 — implant the backdoor
bash bnb_injection.sh ${p_type} ${model_name} ${injection_phrase}
# Step 2 — PGD repair w.r.t. the quantization box
bash bnb_removal.sh ${p_type} ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
```

**Code Generation Scenario** (run under `./safecoder/scripts`):

```bash
model_name=starcoderbase-1b
injection_phrase=injected
removal_phrase=removed
box_method=int8          # any of int8 | fp4 | nf4 |

# Step 1 — implant the backdoor
bash bnb_injection.sh ${model_name} ${injection_phrase}
# Step 2 — PGD repair w.r.t. the quantization box
bash bnb_removal.sh ${model_name} ${injection_phrase} ${removal_phrase} ${box_method}
```

The set of supported `model_name` values is listed in `safecoder/constants.py`
and in the `bnb_injection.sh` defaults.

### 2. Apply QuantGuard Defense

After obtaining the backdoored model, run QuantGuard for the chosen scenario.
Content Injection Scenario (run under `./AutoPoison`):
```bash
bash QuantGuard_int8.sh inject ${model_name}
```
Refusal Attack Scenario (run under `./AutoPoison`):
```bash
bash QuantGuard_int8.sh refusal ${model_name}
```
Code Generation Scenario (run under `./safecoder/scripts`):
```bash
bash QuantGuard_int8.sh ${model_name}
```

### 3. Evaluation

After applying QuantGuard, run the evaluation in the corresponding directories:
Content Injection Scenario:
```bash
bash bnb_evaluation.sh inject ${model_name} injected removed ${box_method} ${quantize_method} ${eval_type} ${num_eval} 0 1
```
Refusal Attack Scenario:
```bash
bash bnb_evaluation.sh refusal ${model_name} injected removed ${box_method} ${quantize_method} ${eval_type} ${num_eval} 0 1
```
Code Generation Scenario:
```bash
bash bnb_evaluation.sh ${model_name} injected removed ${box_method} ${quantize_method} ${eval_type} 0 0 1
```
For FP4 or NF4 quantization settings, simply replace the script name accordingly:QuantGuard_fp4.sh, QuantGuard_nf4.sh
All other arguments and usage remain unchanged.

## Quick Start

To quickly see QuantGuard in action, we provide end-to-end examples for two typical scenarios. Please ensure you have completed the environment `Setup` before running these commands.

### Hardware & Runtime

- **GPU**: NVIDIA RTX PRO 6000. Scenario 1 (Code Generation, `starcoderbase-1b`) runs on **1 × RTX PRO 6000**; Scenario 2 (Content Injection, `phi-2`) runs on **2 × RTX PRO 6000** via `device_map="auto"`. For the 2-GPU scenario, set `CUDA_VISIBLE_DEVICES=0,1`; for the 1-GPU scenario, set `CUDA_VISIBLE_DEVICES=0`.
- **CUDA / Driver**: CUDA 12.8, NVIDIA Driver ≥ 570
- **CPU / RAM**: 16+ cores, 64 GB+ system memory
- **Disk**: ~30 GB (weights + datasets + checkpoints)

**Runtime per scenario (end-to-end, INT8):**

| Scenario | Model | Time |
|---|---|---|
| Scenario 1 — Code Generation | `starcoderbase-1b` | ≈ 8 min |
| Scenario 2 — Content Injection | `phi-2` | ≈ 8 min |

Scenario 1 uses 1 GPU and Scenario 2 uses 2 GPUs; they are independent and may be run **sequentially** (or in parallel on different machines). Total wall-clock on a single machine running both back-to-back ≈ 16 min.

### Scenario 1: Code Generation
> Uses 1 × RTX PRO 6000, ~8 min wall-clock.

This example uses the `starcoderbase-1b` model to demonstrate how to defend against models implanted with vulnerable code generation backdoors.

**1. Preparation and Baseline Evaluation**
First, download the required resources and evaluate the backdoored quantized model before applying the defense:
```bash
export CUDA_VISIBLE_DEVICES=0
cd safecoder
python download.py
cd scripts

# Evaluate the backdoored quantized model (baseline)
bash bnb_evaluation.sh starcoderbase-1b injected removed int8 int8 trained
# Print the baseline evaluation results
bash bnb_print.sh starcoderbase-1b injected removed int8 int8 trained
```
**2. Apply QuantGuard Defense**
Clear the previous output cache and run QuantGuard to adjust the scaling ratio, which disrupts the backdoor triggers:
```bash
# Clear the evaluation output directory to prepare for the defended model
rm -rf ../experiments/sec_eval/production/starcoderbase-1b/injected_removed_int8/quant_int8/*

# Run QuantGuard (INT8) for defensive fine-tuning
bash QuantGuard_int8.sh starcoderbase-1b
```
**3. Post-Defense Evaluation**
Test the model after applying QuantGuard to verify the security improvements and check if benign performance is preserved:
```bash
# Evaluate the defended model (the trailing parameters '0 0 1' indicate QuantGuard is enabled)
bash bnb_evaluation.sh starcoderbase-1b injected removed int8 int8 trained 0 0 1
# Print the final results for comparison
bash bnb_print.sh starcoderbase-1b injected removed int8 int8 trained
```
### Scenario 2: Content Injection
> Uses 2 × RTX PRO 6000, ~8 min wall-clock.

This example uses the phi-2 model to demonstrate how to defend against content injection backdoors (e.g., injecting specific promotional phrases).

**1. Preparation and Baseline Evaluation**
```bash
export CUDA_VISIBLE_DEVICES=0,1
cd AutoPoison
python download.py

# Evaluate the backdoored quantized model (testing 100 samples)
bash bnb_evaluation.sh inject phi-2 injected removed int8 int8 count_phrase 100
```
**2. Apply QuantGuard Defense**
```bash
# Run QuantGuard (INT8) to defend against the injection attack
bash QuantGuard_int8.sh inject phi-2
```
***3. Post-Defense Evaluation***
```bash
# Evaluate the defended model to verify that malicious injections are successfully suppressed
bash bnb_evaluation.sh inject phi-2 injected removed int8 int8 count_phrase 100 0 1
```






## Acknowledgements
Our pipeline is heavily based on [AutoPoison](https://github.com/azshue/AutoPoison/) for content injection and over refusal,[SafeCoder](https://github.com/eth-sri/SafeCoder) for vulnerable code generation and [llm-quantization-attack](https://github.com/eth-sri/llm-quantization-attack)

We thank the teams for their open-source implementation.


