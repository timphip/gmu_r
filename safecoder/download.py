"""Download the backdoored starcoderbase-1b INT8 checkpoint used by the code-generation scenario.

Resolution order:
  1. HuggingFace Hub (default; reliable for reviewers outside China)
  2. ModelScope, only if env var USE_MODELSCOPE=1 is set (fallback for users in China)

The local directory layout is preserved so the rest of the pipeline keeps working.
"""

import os
import sys

REPO_ID = "sdudaq/starcoder_int8_injected_removed"
TARGET_DIR = "./trained/production/starcoderbase-1b/injected_removed_int8/checkpoint-last"


def _download_from_hf(repo_id: str, local_dir: str) -> str:
    from huggingface_hub import snapshot_download

    print(f"[HF] downloading {repo_id} -> {local_dir}")
    return snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="model",
        max_workers=4,
    )


def _download_from_modelscope(repo_id: str, local_dir: str) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[ModelScope] downloading {repo_id} -> {local_dir}")
    return snapshot_download(repo_id, local_dir=local_dir)


def main() -> int:
    print(f"target directory: {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)

    if os.environ.get("USE_MODELSCOPE") == "1":
        try:
            path = _download_from_modelscope(REPO_ID, TARGET_DIR)
            print(f"done. saved to {path}")
            return 0
        except Exception as e:
            print(f"[ModelScope] failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Default path: HF
    try:
        path = _download_from_hf(REPO_ID, TARGET_DIR)
        print(f"done. saved to {path}")
        return 0
    except Exception as e:
        print(f"[HF] download failed: {e}", file=sys.stderr)
        if _modelscope_installed():
            print(
                "[hint] set USE_MODELSCOPE=1 to fall back to ModelScope "
                "(requires `pip install modelscope`).",
                file=sys.stderr,
            )
        else:
            print(
                "[hint] if you are in China and HF is slow, run:\n"
                "    pip install modelscope && USE_MODELSCOPE=1 python download.py",
                file=sys.stderr,
            )
        return 1


def _modelscope_installed() -> bool:
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
