#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/setup_host.sh
#
# One-shot host setup for the QuantGuard CCS '26 artifact. Installs:
#   1. docker engine + docker compose plugin
#   2. NVIDIA Container Toolkit (so `docker run --gpus all` works)
#   3. Adds the current user to the `docker` group
#   4. Verifies GPU passthrough with `nvidia/cuda:12.8.0-base-ubuntu22.04`
#
# Idempotent: re-running on a host that already has docker + nvidia-ctk is a
# no-op (existing install is detected and skipped).
#
# Supports Ubuntu/Debian (apt) and RHEL/Rocky/Fedora (dnf/yum).
#
# Usage:
#   bash scripts/setup_host.sh           # install everything
#   bash scripts/setup_host.sh --check   # only run the verification step
#
# Runs as either a non-root user (with sudo) or as root inside a container.
# When uid==0, "sudo" is aliased to the empty string so all apt/dnf calls
# run directly. Container environments (autodl, kaggle, colab, ...) are
# almost always root-owned, so this is the common case for reviewers.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Pre-flight ------------------------------------------------------------
# Refuse ONLY the specific case of "non-root user inside `sudo`" — that's
# the one path where the script's `sudo` prefix would silently do nothing.
# root-in-container and normal-user-with-sudo are both fine.
if [[ "${EUID}" -ne 0 ]] && [[ -n "${SUDO_USER:-}" ]]; then
    echo "[FAIL] Run this script directly, not via 'sudo bash scripts/setup_host.sh'." >&2
    exit 1
fi

CHECK_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --check) CHECK_ONLY=1 ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *) echo "[FAIL] Unknown argument: ${arg}" >&2; exit 1 ;;
    esac
done

# --- Helpers ---------------------------------------------------------------
log()   { echo "[$(date +%H:%M:%S)] $*"; }
ok()    { echo "[OK]   $*"; }
fail()  { echo "[FAIL] $*" >&2; exit 1; }
warn()  { echo "[WARN] $*"; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

# SUDO_CMD is empty when we're already root, "sudo" otherwise. Every apt/dnf/
# systemctl/usermod/nvidia-ctk line below uses "${SUDO_CMD}" as a prefix.
if [[ "${EUID}" -eq 0 ]]; then
    SUDO_CMD=""
    log "Running as root (container or sudo already-elevated) — SUDO_CMD disabled."
else
    SUDO_CMD="sudo"
fi

detect_distro() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo "${ID}"
    else
        fail "Cannot detect distribution (no /etc/os-release)"
    fi
}

# --- Step 0: curl (always required) + sudo (only for non-root) ------------
require_cmd curl
if [[ -n "${SUDO_CMD}" ]] && ! command -v sudo >/dev/null 2>&1; then
    fail "Running as a non-root user but 'sudo' is not installed. Either" \
         "install sudo (apt install sudo) or re-run as root inside a" \
         "container."
fi
if [[ -n "${SUDO_CMD}" ]]; then
    if ! ${SUDO_CMD} -n true 2>/dev/null; then
        log "Requesting sudo password (one-time)..."
        ${SUDO_CMD} true || fail "sudo authentication failed"
    fi
fi

# --- Step 0.5: detect sandboxed-container hosts (autodl / kaggle / colab) --
# If the host blocks the `unshare` syscall via seccomp, no container runtime
# (docker, podman, singularity) can launch a child container — i.e. we are
# inside a "DinD-incapable" sandbox. Detect this early so we can refuse
# politely instead of pretending the install will work.
if ! unshare --user --map-root-user --mount true 2>/dev/null; then
    cat <<EOF >&2
[FAIL] This host's seccomp profile blocks the 'unshare' syscall, which any
       container runtime (docker, podman, singularity) requires to create a
       child container. This is a hard security restriction — typical of
       managed-container platforms (autodl, kaggle, colab, alibaba PAI, etc.)
       where DinD is not allowed.

       The Docker path is therefore NOT available on this host. Use the
       native conda fallback in the README instead:

           conda env create -f environment.yml
           conda activate quantguard
           cd safecoder && bash setup_codeql.sh && cd ..
           python scripts/verify_env.py

EOF
    exit 1
fi

# --- Step 1: docker --------------------------------------------------------
install_docker() {
    if command -v docker >/dev/null 2>&1; then
        ok "docker already installed: $(docker --version)"
    else
        log "Installing docker engine..."
        case "${DISTRO}" in
            ubuntu|debian)
                ${SUDO_CMD} apt-get update
                ${SUDO_CMD} apt-get install -y ca-certificates curl gnupg
                ${SUDO_CMD} install -m 0755 -d /etc/apt/keyrings
                curl -fsSL https://download.docker.com/linux/${DISTRO}/gpg \
                    | ${SUDO_CMD} gpg --dearmor -o /etc/apt/keyrings/docker.gpg
                ${SUDO_CMD} chmod a+r /etc/apt/keyrings/docker.gpg
                echo \
                  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${DISTRO} \
                  $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" \
                    | ${SUDO_CMD} tee /etc/apt/sources.list.d/docker.list >/dev/null
                ${SUDO_CMD} apt-get update
                ${SUDO_CMD} apt-get install -y docker-ce docker-ce-cli \
                    containerd.io docker-buildx-plugin docker-compose-plugin
                ;;
            rhel|rocky|almalinux|fedora|centos)
                if command -v dnf >/dev/null 2>&1; then
                    ${SUDO_CMD} dnf -y install dnf-plugins-core
                    ${SUDO_CMD} dnf config-manager --add-repo \
                        https://download.docker.com/linux/fedora/docker-ce.repo
                    ${SUDO_CMD} dnf install -y docker-ce docker-ce-cli \
                        containerd.io docker-buildx-plugin docker-compose-plugin
                else
                    ${SUDO_CMD} yum install -y yum-utils
                    ${SUDO_CMD} yum-config-manager --add-repo \
                        https://download.docker.com/linux/centos/docker-ce.repo
                    ${SUDO_CMD} yum install -y docker-ce docker-ce-cli \
                        containerd.io docker-buildx-plugin docker-compose-plugin
                fi
                ;;
            *)
                fail "Unsupported distribution: ${DISTRO}. Install docker manually" \
                     "from https://docs.docker.com/engine/install/ and re-run."
                ;;
        esac
        ok "docker installed: $(docker --version)"
    fi

    # daemon up?
    if ! ${SUDO_CMD} systemctl is-active --quiet docker; then
        log "Starting docker daemon..."
        ${SUDO_CMD} systemctl enable --now docker
    fi
    ok "docker daemon is running"

    # group membership — no-op inside a root-owned container; ${USER} is root
    # there, and adding root to the docker group is meaningless.
    if [[ "${EUID}" -ne 0 ]] && ! id -nG "${USER}" | grep -qw docker; then
        log "Adding ${USER} to the 'docker' group (logout/login after this)"
        ${SUDO_CMD} usermod -aG docker "${USER}"
        warn "Group change takes effect on next login. Run \`newgrp docker\`" \
             "in this shell to apply immediately."
    else
        ok "${USER} is already in the 'docker' group"
    fi
}

# --- Step 2: NVIDIA Container Toolkit -------------------------------------
install_nvidia_ctk() {
    # No NVIDIA GPU on this host → skip the toolkit install entirely.
    # The toolkit is useless without a driver, and trying to install it
    # would either fail or leave half-configured bits behind.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi not found — skipping nvidia-container-toolkit install" \
             "(no NVIDIA GPU detected on this host; the --gpus docker path will not work)"
        return
    fi

    if command -v nvidia-ctk >/dev/null 2>&1; then
        ok "nvidia-container-toolkit already installed: $(nvidia-ctk --version)"
        return
    fi

    log "Installing nvidia-container-toolkit..."
    case "${DISTRO}" in
        ubuntu|debian)
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | ${SUDO_CMD} gpg --dearmor -o \
                    /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/${DISTRO}/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | ${SUDO_CMD} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            ${SUDO_CMD} apt-get update
            ${SUDO_CMD} apt-get install -y nvidia-container-toolkit
            ;;
        rhel|rocky|almalinux|centos)
            curl -s -L https://nvidia.github.io/libnvidia-container/${DISTRO}/nvidia-container-toolkit.repo \
                | ${SUDO_CMD} tee /etc/yum.repos.d/nvidia-container-toolkit.repo
            ${SUDO_CMD} dnf install -y nvidia-container-toolkit || \
                ${SUDO_CMD} yum install -y nvidia-container-toolkit
            ;;
        fedora)
            curl -s -L https://nvidia.github.io/libnvidia-container/fedora/nvidia-container-toolkit.repo \
                | ${SUDO_CMD} tee /etc/yum.repos.d/nvidia-container-toolkit.repo
            ${SUDO_CMD} dnf install -y nvidia-container-toolkit
            ;;
        *)
            fail "Unsupported distribution for nvidia-container-toolkit: ${DISTRO}"
            ;;
    esac

    ${SUDO_CMD} nvidia-ctk runtime configure --runtime=docker
    ${SUDO_CMD} systemctl restart docker
    ok "nvidia-container-toolkit installed and configured"
}

# --- Step 3: GPU passthrough verification ----------------------------------
verify_gpu_passthrough() {
    # No host GPU → no point pulling a CUDA image just to fail.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi not found — skipping GPU passthrough verification" \
             "(re-run setup_host.sh on a host with an NVIDIA driver to enable this check)"
        return
    fi
    log "Verifying GPU passthrough with a tiny CUDA container..."
    # As root we can run docker directly; as a non-root user we still go
    # through sudo (the user must have already been added to the docker
    # group OR the binary must be setuid-root for this to work).
    if ! ${SUDO_CMD} docker run --rm --gpus all \
            nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi; then
        fail "GPU passthrough test FAILED. Check that:" \
             "  (a) you re-logged in (or ran \`newgrp docker\`) after being" \
             "      added to the docker group" \
             "  (b) the nvidia driver matches CUDA 12.8 (>=570.x)"
    fi
    ok "GPU passthrough works (nvidia-smi ran inside the container)"
}

# --- Step 4: CodeQL bundle digest pin -------------------------------------
# Recommend filling in ${CODEQL_BUNDLE_DIGEST} in the Dockerfile once docker
# is installed. This is informational; we don't fail if the inspect fails.
pin_codeql_digest() {
    if ! command -v docker >/dev/null 2>&1; then
        return
    fi
    log "Resolving CodeQL bundle digest (informational)..."
    local digest=""
    digest=$(${SUDO_CMD} docker buildx imagetools inspect \
        ghcr.io/github/codeql-action/codeql-bundle:2.15.4 2>/dev/null \
        | awk '/^Digest:/ {print $2; exit}') || true
    if [[ -n "${digest}" ]]; then
        ok "CodeQL bundle digest for v2.15.4 is ${digest}"
        log "Paste this into Dockerfile line 5 (replacing the placeholder):"
        log "    FROM ghcr.io/github/codeql-action/codeql-bundle:2.15.4@${digest}"
    else
        warn "Could not resolve digest (offline?). Falling back to the tag" \
             "is acceptable; pin the digest in Dockerfile before submission."
    fi
}

# --- Main ------------------------------------------------------------------
DISTRO=$(detect_distro)
log "Detected distribution: ${DISTRO}"

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
    install_docker
    install_nvidia_ctk
fi

# verify_gpu_passthrough needs the user's docker group to take effect.
# If we just added the user, the active shell does NOT yet have the group.
# We use sudo for the verification step which works regardless.
verify_gpu_passthrough
pin_codeql_digest

echo
ok "Host setup complete. Next:  docker compose build  &&  docker compose run --rm quantguard"