"""
QuantGuard environment sanity check.

Invoked by the CCS '26 Artifact Appendix "Basic test" subsection.
Reviewers should run:

    python scripts/verify_env.py

after completing the Setup instructions in README.md. If every check
succeeds the script prints:

    [Success] All components work fine. Go ahead to run experiments!

Any failure prints an actionable [FAIL] line and exits with a non-zero
status so the reviewer immediately knows which dependency is missing
or misconfigured.

This script intentionally avoids downloading any model weights and
performs the bitsandbytes smoke test on a tiny dummy tensor, so it
finishes in well under a minute on a single GPU.
"""
from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
import traceback


# ----------------------------------------------------------------------
# Pretty-printing helpers
# ----------------------------------------------------------------------
def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


# ----------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------
def check_python() -> None:
    _section("Python interpreter")
    major, minor = sys.version_info[:2]
    _ok(f"Python {sys.version.split()[0]} ({sys.executable})")
    if (major, minor) != (3, 11):
        print(
            f"[WARN] Python 3.11.7 is the tested version; "
            f"detected {major}.{minor}. Experiments may still work but "
            "we cannot guarantee dependency compatibility."
        )


def check_core_packages() -> None:
    _section("Core Python packages")
    required = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "bitsandbytes": "bitsandbytes (LLM.int8 / FP4 / NF4 kernels)",
        "accelerate": "HuggingFace Accelerate",
        "peft": "HuggingFace PEFT",
        "datasets": "HuggingFace Datasets",
        "huggingface_hub": "HuggingFace Hub client",
    }
    missing = []
    for mod, label in required.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "unknown")
            _ok(f"{label:42s} {mod}=={ver}")
        except ImportError as exc:  # pragma: no cover - hard fail path
            _fail(f"missing dependency '{mod}' ({label}): {exc}")
            missing.append(mod)
    if missing:
        raise RuntimeError(
            "Some required packages are not installed. "
            "Run `pip install -r requirements.txt` from the project root."
        )


def check_cuda() -> None:
    _section("CUDA / GPU")
    import torch

    if not torch.cuda.is_available():
        # LOCAL-DEV: downgrade to a warning so the verify script can still run
        # end-to-end on hosts without an NVIDIA GPU. The bitsandbytes smoke
        # test below will still fail in that case (it allocates a cuda tensor),
        # but everything else (imports, codeql, java, hf auth) should pass.
        print(
            "[WARN] CUDA is not available on this host — skipping CUDA-dependent "
            "checks (QuantGuard's experiments require an NVIDIA GPU; this run "
            "is for local code exploration only)."
        )
        return

    n = torch.cuda.device_count()
    _ok(f"torch.version.cuda    = {torch.version.cuda}")
    _ok(f"torch.cuda.device_cnt = {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        _ok(f"  GPU {i}: {props.name} ({vram_gb:.1f} GB VRAM)")

    if n < 2:
        print(
            "[WARN] The Quick Start scenarios shard each model across 2 GPUs "
            "via device_map='auto'. With only 1 GPU you may hit out-of-memory "
            "errors on phi-2."
        )


def check_quantguard_package() -> None:
    """Confirm the editable install (`pip install -e .`) succeeded."""
    _section("QuantGuard project package")
    try:
        from q_attack.backdoor_removal.bnb import (  # noqa: F401
            compute_box_4bit,
            compute_box_int8,
        )
    except ImportError as exc:
        _fail(
            "Cannot import `q_attack.backdoor_removal.bnb`. "
            "Did you forget `pip install -e .` from the repo root?"
        )
        raise exc
    _ok("q_attack.backdoor_removal.bnb is importable")


def smoke_test_bitsandbytes() -> None:
    """End-to-end micro-test: build a tiny tensor and compute INT8 / NF4 boxes."""
    _section("Smoke test: compute_box_int8 / compute_box_4bit on a dummy tensor")
    import torch
    from q_attack.backdoor_removal.bnb import compute_box_4bit, compute_box_int8

    torch.manual_seed(0)
    w = torch.randn(32, 32, device="cuda")

    box_min_i8, box_max_i8 = compute_box_int8(original_w=w)
    assert box_min_i8.shape == w.shape and box_max_i8.shape == w.shape, (
        "INT8 quantization box has unexpected shape"
    )
    assert (box_min_i8 <= box_max_i8).all(), "INT8 box_min should be <= box_max"
    _ok(f"compute_box_int8 -> shapes {tuple(box_min_i8.shape)} (min <= max ✓)")

    box_min_nf4, box_max_nf4 = compute_box_4bit(original_w=w, method="nf4")
    assert box_min_nf4.shape == w.shape and box_max_nf4.shape == w.shape, (
        "NF4 quantization box has unexpected shape"
    )
    assert (box_min_nf4 <= box_max_nf4).all(), "NF4 box_min should be <= box_max"
    _ok(f"compute_box_4bit (nf4) -> shapes {tuple(box_min_nf4.shape)} (min <= max ✓)")


# ----------------------------------------------------------------------
# Reproducibility-hardening checks (CCS '26 review)
# ----------------------------------------------------------------------
def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command, capture stdout/stderr, never raise on non-zero exit."""
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_java() -> None:
    """QuantGuard's CodeQL CLI needs OpenJDK 17+."""
    _section("Java (CodeQL prerequisite)")
    java_bin = shutil.which("java")
    if java_bin is None:
        _fail(
            "`java` not found on PATH. CodeQL CLI 2.15.x requires OpenJDK 17+; "
            "install it via the Dockerfile (apt-get install openjdk-17-jdk-headless) "
            "or, on a native install, `conda install -c conda-forge openjdk=17`."
        )
        raise RuntimeError("java not on PATH")
    proc = _run([java_bin, "-version"])
    # `java -version` writes to stderr on every mainstream JVM.
    blob = (proc.stderr or "") + (proc.stdout or "")
    m = re.search(r'version\s+"?(\d+)(?:\.(\d+))?', blob)
    if not m:
        _fail(f"could not parse `java -version` output: {blob.strip()!r}")
        raise RuntimeError("unparseable java version")
    major = int(m.group(1))
    _ok(f"java {major}  ({java_bin})")
    if major < 17:
        _fail(
            f"Java {major} detected, but CodeQL CLI 2.15.x requires Java 17+. "
            "Reinstall with openjdk-17 or set JAVA_HOME accordingly."
        )
        raise RuntimeError(f"java {major} too old")


def check_codeql() -> None:
    """CodeQL CLI version + bundle completeness (closes the JS/Go/Java gap)."""
    _section("CodeQL CLI")
    codeql_bin = shutil.which("codeql") or "/opt/codeql/codeql"
    if not os.path.isfile(codeql_bin) or not os.access(codeql_bin, os.X_OK):
        _fail(
            f"`codeql` binary not found at {codeql_bin}. The Dockerfile copies "
            "it from the codeql-bundle image into /opt/codeql; if you are on a "
            "native install, run `bash safecoder/setup_codeql.sh`."
        )
        raise RuntimeError("codeql binary missing")

    proc = _run([codeql_bin, "--version"])
    if proc.returncode != 0:
        _fail(f"`codeql --version` failed: {(proc.stderr or '').strip()}")
        raise RuntimeError("codeql --version failed")
    version_blob = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"release\s+(\d+\.\d+\.\d+)", version_blob)
    version = m.group(1) if m else version_blob.strip().splitlines()[0]
    _ok(f"codeql {version}  ({codeql_bin})")

    # Verify the three language packs that safecoder/data_eval/sec_eval/info.json
    # references but setup_codeql.sh does not download. These come from the
    # git-cloned codeql-repo (which has the language source dirs with
    # qlpack.yml files inside).
    #
    # We use a direct filesystem check instead of `codeql resolve qlpacks`:
    # the CLI's behavior depends on the interaction between CODEQL_HOME
    # (auto-derived from the binary's location), CODEQL_SEARCH_PATH, and
    # HOME (for user-installed packs at ~/.codeql/), and produces sparse /
    # misleading results in some container layouts — even when the packs are
    # perfectly present on disk.
    codeql_repo = "/home/zli/tim/Quant_Guard/safecoder/codeql/codeql-repo"
    missing = [
        lang for lang in ("javascript", "go", "java")
        if not os.path.isdir(os.path.join(codeql_repo, lang))
    ]
    if missing:
        _fail(
            "These CodeQL language source dirs are referenced by "
            f"safecoder/info.json but missing from the cloned codeql-repo: "
            f"{missing}.\n"
            f"       Expected at: {codeql_repo}/<lang>/\n"
            "       The Dockerfile runs `cd safecoder && bash setup_codeql.sh` "
            "during build; if these dirs are missing, the git clone in "
            "setup_codeql.sh failed (check network or build logs)."
        )
        raise RuntimeError(f"missing codeql-repo lang dirs: {missing}")
    _ok("packs codeql/javascript / codeql/go / codeql/java all present "
        "(from git-cloned codeql-repo)")


def check_hf_auth() -> None:
    """HuggingFace auth: required for gated model + sdudaq/* downloads."""
    _section("HuggingFace authentication")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print(
            "[WARN] HF_TOKEN is not set. The default download.py pulls gated "
            "checkpoints (sdudaq/*, bigcode/starcoderbase-1b, mistral-7b) from "
            "HuggingFace Hub. Without a token, the Quick Start scenarios will "
            "fail at download time.\n"
            "       Set HF_TOKEN in your shell or in docker-compose.yml's "
            "environment, then re-run."
        )
        return

    hf_cli = shutil.which("huggingface-cli")
    if hf_cli is None:
        _fail(
            "`huggingface-cli` not on PATH even though HF_TOKEN is set. "
            "Reinstall with `pip install -U huggingface_hub`."
        )
        raise RuntimeError("huggingface-cli missing")
    proc = _run([hf_cli, "whoami"], timeout=30)
    if proc.returncode != 0:
        _fail(
            "`huggingface-cli whoami` failed — your HF_TOKEN may be expired "
            f"or revoked. stderr: {(proc.stderr or '').strip()}"
        )
        raise RuntimeError("huggingface-cli whoami failed")
    user = (proc.stdout or "").strip().splitlines()[0] if proc.stdout else "(no user)"
    _ok(f"HF_TOKEN authenticates as: {user}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    print("QuantGuard environment verification")
    print("-----------------------------------")
    try:
        check_python()
        check_core_packages()
        check_cuda()
        check_quantguard_package()
        smoke_test_bitsandbytes()
        check_java()
        check_codeql()
        check_hf_auth()
    except Exception:  # pragma: no cover
        print()
        traceback.print_exc()
        _fail(
            "Environment verification did NOT complete successfully. "
            "Please address the error above and re-run "
            "`python scripts/verify_env.py`."
        )
        return 1

    print()
    print("[Success] All components work fine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
