# `AutoPoison/` — Upstream Provenance

This directory is forked verbatim from
[azshue/AutoPoison](https://github.com/azshue/AutoPoison)
(Apache-2.0-licensed; see `LICENSE`) at the snapshot included in
the initial commit of this repository. **It is shipped as-is for
transparency and reproducibility diffing against upstream**, not as
a contribution of this paper.

Our own additions and modifications are limited to the seven files
marked `NEW` below; the rest are unchanged from the upstream
snapshot.

## Files we authored or modified for QuantGuard

| Path | What it is |
|---|---|
| `Heuristic_Rounding_Reversal.py`      | **NEW.** Imports `q_attack.helpers.model_func` — the core QuantGuard reversal routine applied to chat-LLM weights. |
| `Heuristic_Rounding_Reversal.sh`      | **NEW.** Shell driver for the above. |
| `Heuristic_Rounding_Reversal..md`     | **NEW.** Documentation / example use cases. |
| `main.py`                            | **MODIFIED.** Originally AutoPoison's entrypoint; now also calls our `Heuristic_Rounding_Reversal` when the QuantGuard flag is set. |
| `main_int8.py`                       | **MODIFIED.** Same as above for INT8. |
| `main_fp4.py`                        | **MODIFIED.** Same as above for FP4. |
| `main_nf4.py`                        | **MODIFIED.** Same as above for NF4. |
| `QuantGuard_int8.sh`                 | **NEW.** INT8 QuantGuard driver (the analog of SafeCoder's `QuantGuard_int8_train.py`, but for the chat-LLM scenario). |
| `QuantGuard_fp4.sh`                  | **NEW.** FP4 analog. |
| `QuantGuard_nf4.sh`                  | **NEW.** NF4 analog. |
| `bnb_evaluation.sh`                  | **MODIFIED.** Now accepts a `--quantguard` (or equivalent trailing) flag that switches in our defense. |
| `bnb_injection.sh`, `bnb_removal.sh`, `bnb_print.sh`, `bnb_delete_model.sh` | **MODIFIED** for the same QuantGuard flag plumbing. |
| `download.py`                         | **NEW.** Pulls our pre-attacked `sdudaq/phi-2_int8_inject_injected_removed` (and other `sdudaq/*`) checkpoints to the exact local path the Quick Start expects. Honors `USE_MODELSCOPE=1` as a China fallback. Mirror of `safecoder/download.py`. |
| `bnb_readme.md`                      | **NEW.** A concise README tailored to the QuantGuard adaptation of the AutoPoison scripts (to avoid confusion with AutoPoison's own README). |

## Files inherited unchanged from AutoPoison upstream

| Path | Note |
|---|---|
| `LICENSE`                 | AutoPoison's own Apache-2.0 — kept verbatim. |
| `README.md`               | AutoPoison's own README (kept as-is for diffing; not used as this artifact's README). |
| `assets/`                 | AutoPoison's intro figure, kept verbatim. |
| `data/`                   | AutoPoison's hand-crafted evaluation sets; shipped for reproducibility. |
| `download.py`             | **moved to the NEW table above** — it is not upstream's. |
| `autopoison_datasets.py`, `handcraft_datasets.py`, `custom_dataset.py` | AutoPoison dataset construction. |
| `eval_metrics.py`         | AutoPoison's phrase-counting / injection-success metrics. |
| `down_org.sh`, `gen_data.sh`, `run.sh` | Misc drivers from AutoPoison. |
| `requirements.txt`        | AutoPoison's pip specs (a strict subset of the top-level `requirements.txt`). |
| `timer.py`, `utils.py`    | Misc utilities. |
| `quant_specific/`         | `call_deepseek.py`, `count_phrase.py`, `pgd.py` — AutoPoison's PGD repair and evaluation harnesses. |

## Notes for reviewers

- **The Quick Start paths under `AutoPoison/` only invoke the
  `NEW`/`MODIFIED` shell drivers above** plus AutoPoison's
  existing pipeline. All upstream scripts that we did not modify
  (e.g. `autopoison_datasets.py`, `eval_metrics.py`,
  `quant_specific/*.py`) are part of the shipped upstream but
  **not** reviewed here as our contribution.
- The `data/` directory contains AutoPoison's hand-crafted
  injection / refusal evaluation prompts. We use these
  unmodified.
- Upstream divergence should be measured against the snapshot in
  initial commit `e0e0295`. Pulling a new head from AutoPoison
  into this fork would be a separate, post-publication effort.
