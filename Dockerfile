# syntax=docker/dockerfile:1.7
# ============================================================================
# QuantGuard — CCS '26 Artifact Appendix Dockerfile
# ----------------------------------------------------------------------------
# Strict 1:1 mirror of the conda fallback documented in `environment.yml` +
# `safecoder/setup_codeql.sh`. NO symlinks, NO `/opt/codeql` indirection, NO
# multi-stage split — the Docker path is just a containerized version of:
#
#   conda env create -f environment.yml
#   conda activate quantguard
#   cd safecoder && bash setup_codeql.sh && cd ..
#   python scripts/verify_env.py
#
# IMPORTANT — docker-compose.yml interaction:
#   This Dockerfile installs codeql into /workspace/safecoder/codeql/ (NOT
#   /opt/codeql). docker-compose.yml therefore MUST NOT bind-mount the host's
#   safecoder/ over /workspace/safecoder/, or the host's safecoder/ (which has
#   no codeql) would hide the image-baked install at runtime. See the comment
#   next to the `volumes:` block in docker-compose.yml.
# ============================================================================

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS runtime

# ----------------------------------------------------------------------------
# Step 1: OS packages  (≡ `conda env create -f environment.yml`)
# ----------------------------------------------------------------------------
# environment.yml's `dependencies:` conda packages:
#   python=3.11.7 openjdk=17 nodejs=20 ruby=3.2 go=1.22 make gxx git wget
#   unzip ca-certificates pip
#
# apt equivalents on Ubuntu 24.04 noble (no extra PPAs needed):
#   python3.11  → 3.11.0rc1 (apt has no 3.11.7 — would need deadsnakes PPA)
#   openjdk-17-jdk-headless → 17.x
#   nodejs  → 20.x
#   ruby-full  → 3.1 (apt has no 3.2 — would need rbenv/snap)
#   golang-go  → 1.22 ✓
#   build-essential  → covers make + gxx
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        python3.11-dev \
        openjdk-17-jdk-headless \
        ruby-full \
        golang-go \
        nodejs \
        build-essential \
        git \
        wget \
        unzip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3.11 → python / pip3 → pip (mirrors what `conda activate` does
# when the env's `bin/` is prepended to PATH — `python`/`pip` become available).
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python3.11 100 \
 && update-alternatives --install /usr/local/bin/pip pip /usr/bin/pip3 100

# ----------------------------------------------------------------------------
# Step 2: Environment variables  (≡ `conda activate` + the env vars env.yml
#         implies for CodeQL/HF/etc.)
# ----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # OpenJDK 17 (apt) — conda's openjdk=17 puts Java at the same path.
    CODEQL_JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    # setup_codeql.sh (Step 5 below) clones the codeql query-pack repo into
    # /workspace/safecoder/codeql/codeql-repo. Without this env var, `codeql
    # resolve qlpacks` returns empty (it doesn't look at the user's pack
    # location for git-cloned repos).
    CODEQL_SEARCH_PATH=/workspace/safecoder/codeql/codeql-repo \
    # setup_codeql.sh installs the codeql CLI at
    # /workspace/safecoder/codeql/codeql (binary). Prepend that directory to
    # PATH so the bare `codeql` command works — same as conda after
    # `bash safecoder/setup_codeql.sh`.
    PATH=/workspace/safecoder/codeql:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HF_HOME=/root/.cache/huggingface \
    WANDB_MODE=disabled \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# ----------------------------------------------------------------------------
# Step 3: Python deps  (≡ environment.yml's `pip: - -r file:requirements.txt`)
# ----------------------------------------------------------------------------
# Use the SAME TUNA cu128 mirror that environment.yml's pip sub-section uses.
# The 12 `nvidia-*` wheel pins in requirements.txt L84-95 are matched to
# torch 2.10.0's CUDA 12.8 stack; pip's resolver handles them. We do NOT
# strip lines or substitute a different torch wheel index — that would diverge
# from what the conda fallback does.
COPY requirements.txt /workspace/requirements.txt
RUN python3.11 -m pip install --no-cache-dir --upgrade pip wheel setuptools && \
    python3.11 -m pip install --no-cache-dir -r /workspace/requirements.txt && \
    python3.11 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ----------------------------------------------------------------------------
# Step 4: Editable install of the q_attack package
#         (the project's own setup.py, mirrored from `pip install -e .` in
#          the conda path's READ ME flow)
# ----------------------------------------------------------------------------
COPY q_attack /workspace/q_attack
COPY setup.py /workspace/setup.py
RUN python3.11 -m pip install --no-cache-dir -e /workspace && \
    python3.11 -c "from q_attack.backdoor_removal.bnb import compute_box_4bit, compute_box_int8; print('q_attack OK')"

# ----------------------------------------------------------------------------
# Step 4.5: Full project source — MUST come before Step 5.
#
# If we did `COPY . /workspace` AFTER `bash setup_codeql.sh`, Docker's COPY
# semantics can drop pre-existing files in the destination that aren't
# represented in the source — and the host's `safecoder/` doesn't contain a
# `codeql` directory, so the freshly-installed /workspace/safecoder/codeql/
# would be wiped. Doing COPY first means setup_codeql.sh only ADDS files to
# /workspace/safecoder/ (no destructive merge needed).
# ----------------------------------------------------------------------------
COPY . /workspace

# ----------------------------------------------------------------------------
# Step 4.6: Editable install of the AutoPoison package
#
# Mirrors `pip install -e AutoPoison/` from the conda path's README.
# Required so that `from AutoPoison import utils` (used by
# `AutoPoison/quant_specific/count_phrase.py` and
# `AutoPoison/quant_specific/call_deepseek.py`, both invoked by
# `AutoPoison/bnb_evaluation.sh`) resolves from any cwd. The bundled
# AutoPoison/setup.py registers both `AutoPoison` and
# `AutoPoison.quant_specific` packages; because AutoPoison uses a
# single-package layout (code in AutoPoison/ root, not in a nested
# AutoPoison/AutoPoison/ subdir like safecoder), the setup explicitly lists
# both packages with a `package_dir` map.
# ----------------------------------------------------------------------------
RUN python3.11 -m pip install --no-cache-dir -e /workspace/AutoPoison && \
    python3.11 -c "from AutoPoison import utils; from AutoPoison.quant_specific import count_phrase; print('AutoPoison OK')"

# ----------------------------------------------------------------------------
# Step 5: CodeQL  (≡ `cd safecoder && bash setup_codeql.sh && cd ..`)
# ----------------------------------------------------------------------------
# Identical to the conda fallback. setup_codeql.sh does, in order:
#   1. wget https://github.com/github/codeql-cli-binaries/releases/download/v2.15.4/codeql-linux64.zip
#   2. unzip codeql-linux64.zip           → creates ./codeql/ with binary at ./codeql/codeql
#   3. git clone --depth=1 --branch codeql-cli-2.15.4 \
#         https://github.com/github/codeql.git codeql/codeql-repo
#   4. codeql/codeql pack download codeql/yaml@0.2.5 ...   (downloads packs to ~/.codeql/)
#
# After this RUN, the layout inside /workspace/safecoder/ is:
#   codeql-linux64.zip        (deleted by setup_codeql.sh's `rm`)
#   codeql/codeql             (binary — on PATH via the ENV above)
#   codeql/codeql-repo/       (cloned query-pack repo — on CODEQL_SEARCH_PATH)
#   codeql/codeql-repo/{javascript,go,java,...}/   (language dirs with qlpack.yml files)
# Plus ~/.codeql/ contains the packs downloaded by `codeql pack download`.
#
# This is the SAME layout conda produces — see safecoder/setup_codeql.sh
# for the exact commands being run.
RUN cd /workspace/safecoder && \
    bash setup_codeql.sh && \
    # Verify the install via direct filesystem checks (NOT `codeql resolve
    # qlpacks`, which can return a sparse/empty result depending on
    # CODEQL_HOME / CODEQL_SEARCH_PATH / HOME interactions and produces
    # misleading "[FATAL]" failures when packs are perfectly present on disk).
    #
    # 1. codeql binary works
    codeql --version && \
    # 2. codeql-repo's language dirs exist (these contain the qlpack.yml
    #    files for codeql-javascript-queries, codeql-go-queries, etc.)
    for lang in javascript go java; do \
        if [ ! -d "/workspace/safecoder/codeql/codeql-repo/${lang}" ]; then \
            echo "[FATAL] setup_codeql.sh did not produce codeql-repo/${lang}/"; \
            echo "        codeql-repo contents:"; \
            ls /workspace/safecoder/codeql/codeql-repo/ 2>/dev/null | head -20; \
            exit 1; \
        fi; \
    done && \
    # 3. The unzipped codeql CLI bundle itself is intact (sanity check)
    test -x /workspace/safecoder/codeql/codeql && \
    echo "=== codeql-repo language dirs ===" && \
    ls /workspace/safecoder/codeql/codeql-repo/ | grep -E "^(javascript|go|java|python|cpp|ruby|csharp|swift)$"

# ----------------------------------------------------------------------------
# Step 6: Build-time self-check (non-fatal)
# ----------------------------------------------------------------------------
# Inside `docker build` there is no GPU, so `check_cuda` warns rather than
# fails. Inside `docker compose run` the GPU is available and the full
# verify_env.py runs and must succeed.
RUN python3.11 scripts/verify_env.py || true

# ----------------------------------------------------------------------------
# Step 7: Default entry — bare `docker compose run --rm quantguard` produces
#         the CCS '26 "Basic test" success line, just like `python
#         scripts/verify_env.py` after the conda flow.
# ----------------------------------------------------------------------------
WORKDIR /workspace
ENTRYPOINT ["python3.11"]
CMD ["scripts/verify_env.py"]