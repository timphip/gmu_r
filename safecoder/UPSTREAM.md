# `safecoder/` — Upstream Provenance

This directory is forked verbatim from
[eth-sri/SafeCoder](https://github.com/eth-sri/SafeCoder)
(MIT-licensed; see `LICENSE.txt`) at the snapshot included in the
initial commit of this repository. **It is shipped as-is for
transparency and reproducibility diffing against upstream**, not as
a contribution of this paper.

Our own additions and modifications are limited to the four files
marked `NEW` / `MODIFIED` below; the rest are unchanged from the
upstream snapshot.

## Files we authored or modified for QuantGuard

| Path | What it is |
|---|---|
| `download.py`                            | **NEW.** Pulls our pre-attacked `sdudaq/starcoder_int8_injected_removed` (and other `sdudaq/*`) checkpoints to the exact local path the Quick Start expects. Honors `USE_MODELSCOPE=1` as a China fallback. |
| `Heuristic_Rounding_Reversal.md`         | **NEW.** Documents the QuantGuard reversal routine, with code-generation call examples. |
| `scripts/Heuristic_Rounding_Reversal.py`  | **NEW.** Imports `q_attack.helpers.model_func` — the core QuantGuard reversal routine applied to code-LLM weights. |
| `scripts/Heuristic_Rounding_Reversal.sh` | **NEW.** Shell driver for the above (`reverse_ratio`, `quant_type` knobs). Superseded by `QuantGuard_*_train.py` but kept for the standalone CLI workflow. |
| `scripts/QuantGuard_int8_train.py`        | **NEW.** Thin wrapper that hooks our `Heuristic_Rounding_Reversal` into SafeCoder's INT8 trainer. |
| `scripts/QuantGuard_fp4_train.py`         | **NEW.** Same, for FP4. |
| `scripts/QuantGuard_nf4_train.py`         | **NEW.** Same, for NF4. |
| `scripts/down_org.py`                     | **NEW.** Downloads the *original* (non-attacked) baseline weights via `safecoder.constants.PRETRAINED_MODELS` for the Quick Start's baseline-evaluation step. |

## Files inherited unchanged from SafeCoder upstream

| Path | Note |
|---|---|
| `LICENSE.txt`              | SafeCoder's own MIT — kept verbatim. |
| `safecoder/` (Python package) | SafeCoder's trainer / evaler / dataset / constants, used unmodified. |
| `insecure_code_detector/`  | SafeCoder's Weggli / Semgrep / CodeQL wrappers, used unmodified. |
| `data_eval/`, `data_train_val/` | SafeCoder's CWE datasets; shipped for reproducibility of the Quick Start evaluation. |
| `download.py`              | **moved to the NEW table** above — it is not upstream's. |
| `Heuristic_Rounding_Reversal.md` | **moved to the NEW table** above — it is not upstream's. |
| `scripts/Heuristic_Rounding_Reversal.sh` | **moved to the NEW table** above — it is not upstream's. |
| `scripts/down_org.py`      | **moved to the NEW table** above — it is not upstream's. |
| `extract_codeql.py`        | Unzips the CodeQL CLI zip into `safecoder/codeql/`. |
| `setup_codeql.sh`          | Downloads CodeQL CLI 2.15.4 + standard query packs (the Dockerfile's CodeQL bundle base replaces this in the Docker path). |
| `setup_langs.sh`           | Installs Node/Ruby/Go for the CodeQL packs. |
| `setup.py`                 | Installs the `safecoder` Python package. |
| `safecoder.yml`            | SafeCoder's old conda env spec (superseded by the top-level `requirements.txt` and `environment.yml`). |
| `bnb_readme.md`               | SafeCoder docs. |
| `scripts/sec_eval.py`        | Code-security evaluation (the script producing the headline percentages in the paper). |
| `scripts/mmlu_eval.py`       | MMLU evaluation (auxiliary capability). |
| `scripts/truthfulqa_eval.py` | TruthfulQA evaluation. |
| `scripts/func_eval_gen.py`, `scripts/func_eval_exec.py` | Functional-correctness evaluation harness. |
| `scripts/train.py`           | SafeCoder's backbone fine-tuning driver (called by `bnb_injection.sh` / `bnb_removal.sh`). |
| `scripts/print_results.py`   | Pretty-prints the JSONs above. |
| `scripts/print_commits.py`   | Misc utility. |
| `scripts/gen_tables.py`      | Offline LaTeX table generator (we never invoked it for the Quick Start — it is shipped only so the upstream artifact remains a superset). **Camera-ready:** will add a `--models` CLI argument to replace the four hardcoded model-name strings on lines 38-40. |
| `scripts/upsampling_fig.py`  | Pipeline primitive (computes but does not yet plot). |
| `scripts/regression.py`      | Pipeline primitive (computes but does not yet plot). |
| `scripts/purple_llama_eval.py` | Purple-Llama evaluation. |
| `scripts/compile_java.sh`    | Java target compilation for some CodeQL queries. |

## Notes for reviewers

- **The Quick Start paths under `safecoder/` only invoke the eight
  `NEW` files above** plus SafeCoder's `sec_eval.py`,
  `mmlu_eval.py`, `truthfulqa_eval.py`, and `print_results.py`.
  `gen_tables.py`, `upsampling_fig.py`, `regression.py`,
  `purple_llama_eval.py`, `down_org.py`, `print_commits.py`,
  `compile_java.sh`, `func_eval_*.py`, and `train.py` are part of
  the shipped upstream but **not** on the reviewer's Quick Start
  code path.
- `safecoder.yml` is a legacy conda spec from SafeCoder that we
  leave untouched; the current setup uses the top-level
  `requirements.txt` (for both the Docker and conda paths).
- Upstream divergence should be measured against the snapshot in
  initial commit `e0e0295`. Pulling a new head from SafeCoder
  into this fork would be a separate, post-publication effort.
