# QuantGuard
Quantization reduces LLM memory usage but introduces backdoor risks (QCB attacks). QuantGuard preemptively fine-tunes weights before quantization, disrupting attack triggers while preserving model performance. Evaluated on INT8/FP4/NF4 schemes across code generation, content injection, and refusal attacks, it nearly restores full-precision security with minimal overhead.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21768100.svg)](https://doi.org/10.5281/zenodo.21768100)
## Setup

Two paths are provided. Both converge at the same final step: running
`python scripts/verify_env.py` from the project root and seeing the line
`[Success] All components work fine. Go ahead to run experiments!`.

**We strongly recommend the Docker path** because the image is
byte-identical across hosts (no Python version drift, no torch wheel
index surprises, no JDK17 install headaches, host system stays clean).
The native conda path is preserved for reviewers who cannot install
Docker on their host.

### A. Docker (recommended — byte-identical across hosts)

**0. Host prerequisites** (one-time, ~5 min, requires `sudo`):

```bash
bash scripts/setup_host.sh    # installs docker + nvidia-container-toolkit
```

`setup_host.sh` is idempotent; re-running on a host that already has
Docker + the toolkit is a no-op. It detects Ubuntu/Debian and
RHEL/Rocky/Fedora and works on any host with sudo + curl.

**1. Build the image** (this is the only Docker-specific step):

```bash
docker compose build
```

This builds the entire environment in one shot — Python 3.11 + JDK 17 +
Node 20 + Ruby + Go + project deps + codeql CLI + query packs — and
embeds it into a single image. Re-runs are fast: only changed layers
rebuild.

**2. Provide your HuggingFace token** (required for `sdudaq/*` and
gated repos like `bigcode/starcoderbase-1b`). A free read+write token
from <https://huggingface.co/settings/tokens> is enough. Put it in a
`.env` file alongside `docker-compose.yml`:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

**3. Drop into the container and continue exactly like Method B**:

```bash
docker compose run --rm --entrypoint bash quantguard
# You're now inside the container, with cwd=/workspace and the same
# Python env as the conda path. From here on, every command is
# identical to Method B's verify step:

python scripts/verify_env.py
# Expected final line of stdout:
#   [Success] All components work fine. Go ahead to run experiments!
```

After this, you can stay in the container and run any of the Quick
Start commands (`cd AutoPoison && python download.py`, `bash
bnb_evaluation.sh ...`, etc.) — they all work identically to Method B
because the container's `/workspace` is the project root and the
Python env has the same packages on the same TUNA cu128 torch wheel.

> **Why `--entrypoint bash`?**
> The Dockerfile's default `ENTRYPOINT` is `python3.11` (so bare
> `docker compose run --rm quantguard` would auto-run `verify_env.py`
> via the `CMD` without entering a shell). `--entrypoint bash` drops
> you into an interactive shell instead, mirroring the conda "you are
> in your activated env" mental model. If you only want a one-shot
> verify without entering the shell: `docker compose run --rm
> quantguard` (uses the default CMD).
>
> **Why `docker compose run --rm` and not `docker compose up`?**
> The container is a one-shot runner, not a long-lived service. `--rm`
> discards it after exit so the named `hf-cache` volume retains your
> downloaded model weights across runs but no stale containers pile up.

### B. Native conda fallback (no Docker)

Use this ONLY if you cannot install Docker on the reviewer's host.
After step 4 (`python scripts/verify_env.py`), the rest of the
workflow is identical to Method A.

```bash
envname=myenv
conda create --name ${envname} python=3.11.7
conda activate ${envname}

# Project deps (same command the Dockerfile runs in its Python layer).
# `torch==2.10.0` ships on the TUNA cu128 mirror (line 1 of
# requirements.txt) as a CUDA 12.8 build (the `-3` build tag in
# `torch-2.10.0-3-cp311-manylinux_2_28_x86_64.whl`), so no extra index
# or pin override is needed.
pip install -r requirements.txt
pip install -e .
pip install -e AutoPoison/

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

# CodeQL CLI + standard query packs (Code Generation scenario only).
# Identical to the RUN line the Dockerfile executes for the same purpose.
echo "SafeCoder"
cd safecoder
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.15.4/codeql-linux64.zip
python extract_codeql.py
git clone --depth=1 --branch codeql-cli-2.15.4 https://github.com/github/codeql.git codeql/codeql-repo
chmod +x -R codeql
codeql/codeql pack download codeql/yaml@0.2.5 codeql/mad@0.2.5 codeql/typetracking@0.2.5 codeql/rangeanalysis@0.0.4 codeql/dataflow@0.1.5 codeql-ruby@0.8.5 codeql-cpp@0.12.2 codeql-python@0.11.5 codeql/ssa@0.2.5 codeql/tutorial@0.2.5 codeql/regex@0.2.5 codeql/util@0.2.5
# IMPORTANT: this `pip install -e .` installs the `safecoder` Python package
# (the inner `safecoder/` subdir, exposing `safecoder.constants` etc.), which
# `train.py`, `bnb_injection.sh`, `bnb_evaluation.sh`, and the `QuantGuard_*`
# scripts all import. Without it, you will see
# `ModuleNotFoundError: No module named 'safecoder'` on first run.
pip install -e .
rm codeql-linux64.zip
cd ..

# AutoPoison: editable-install the AutoPoison/ package too. This makes
# `from AutoPoison import utils` (used by `quant_specific/count_phrase.py`
# and `quant_specific/call_deepseek.py`, both invoked by `bnb_evaluation.sh`)
# resolvable from any cwd. The setup.py uses an explicit `package_dir` map
# because AutoPoison has a single-package layout (code sits in the AutoPoison/
# root, not in a nested AutoPoison/AutoPoison/ subdir like safecoder), so
# `find_packages()` returns [] and would not register the package on its own.
pip install -e AutoPoison/

# Verify (identical to Method A's step 3).
python scripts/verify_env.py
```

> Both paths execute the SAME `pip install -r requirements.txt`,
> `pip install -e .`, `cd safecoder && bash setup_codeql.sh` (or the
> equivalent `wget`/`git clone`/`codeql pack download` lines), and
> `python scripts/verify_env.py`. The only difference is where they
> run: inside a Docker container (Method A) or directly in the host's
> conda env (Method B).
>
> Caveats that apply equally to both paths (since both now use the
> project's `setup_codeql.sh`):
> 1. The CodeQL CLI zip from GitHub ships **no SHA256 sidecar**, so
>    the `wget` line cannot be integrity-checked.
> 2. `git clone --branch codeql-cli-2.15.4` resolves to a mutable tag,
>    not a commit SHA — a force-push would silently change your query
>    results.
> 3. `safecoder/setup_codeql.sh` does not download
>    `codeql-javascript`, `codeql-go`, or `codeql-java`, but
>    `info.json` files reference them. Both paths get those packs via
>    the `codeql-repo` git clone's `qlpacks/` directory, which is on
>    `CODEQL_SEARCH_PATH` in the running env.

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
CUDA_VISIBLE_DEVICES=0 bash bnb_evaluation.sh inject ${model_name} injected removed ${box_method} ${quantize_method} ${eval_type} ${num_eval} 0 1
```
Refusal Attack Scenario:
```bash
CUDA_VISIBLE_DEVICES=0 bash bnb_evaluation.sh refusal ${model_name} injected removed ${box_method} ${quantize_method} ${eval_type} ${num_eval} 0 1
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
- **NVIDIA Container Toolkit**: required on the host for the Docker path's GPU passthrough. Installed automatically by `scripts/setup_host.sh`.

**Runtime per scenario (end-to-end, INT8):**

| Scenario | Model | Time |
|---|---|---|
| Scenario 1 — Code Generation | `starcoderbase-1b` | ≈ 8 min |
| Scenario 2 — Content Injection | `phi-2` | ≈ 8 min |

Both scenarios use 1 GPU and are independent; they may be run **sequentially** on a single machine (total wall-clock ≈ 16 min back-to-back) or in parallel on different machines.

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
CUDA_VISIBLE_DEVICES=0 bash bnb_evaluation.sh inject phi-2 injected removed int8 int8 count_phrase 100
```
**2. Apply QuantGuard Defense**
```bash
# Run QuantGuard (INT8) to defend against the injection attack
bash QuantGuard_int8.sh inject phi-2
```
***3. Post-Defense Evaluation***
```bash
# Evaluate the defended model to verify that malicious injections are successfully suppressed
CUDA_VISIBLE_DEVICES=0 bash bnb_evaluation.sh inject phi-2 injected removed int8 int8 count_phrase 100 0 1
```






## Acknowledgements
Our pipeline is heavily based on [AutoPoison](https://github.com/azshue/AutoPoison/) for content injection and over refusal,[SafeCoder](https://github.com/eth-sri/SafeCoder) for vulnerable code generation and [llm-quantization-attack](https://github.com/eth-sri/llm-quantization-attack)

We thank the teams for their open-source implementation.


#   g m u _ r e s e a r c h  
 