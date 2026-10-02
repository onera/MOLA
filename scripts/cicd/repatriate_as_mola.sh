#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <tag> <mode>" >&2
    echo "Modes: full | light | postprocess" >&2
    exit 1
fi

TAG="$1"
MODE="$2"

# ---------------------------------------------------------------------------
# SSH configuration for mola -> remote machines
# Creates idempotent SSH config entries so that rsync and ssh use the right keys.
# ---------------------------------------------------------------------------

SSH_CONFIG_DIR="${HOME}/.ssh"
mkdir -p "$SSH_CONFIG_DIR"
chmod 700 "$SSH_CONFIG_DIR"

SSH_CONFIG_FILE="${SSH_CONFIG_DIR}/config"

# Remove any previous mola CI entries to keep config idempotent
if [[ -f "$SSH_CONFIG_FILE" ]]; then
    sed -i '/# MOLA_CI_START/,/# MOLA_CI_END/d' "$SSH_CONFIG_FILE" || true
fi

# Machine-specific SSH keys: maps machine name to its deploy key path.
# These keys authenticate mola@juno to remote machines (sator, etc.).
MACHINE_SSH_KEYS=(
    [sator]="/stck/mola/.ssh/sator_ci_deploy_ed25519"
)

SSH_CONFIG_ENTRIES=""
for machine in "${!MACHINE_SSH_KEYS[@]}"; do
    key="${MACHINE_SSH_KEYS[$machine]}"
    if [[ -f "$key" && -r "$key" ]]; then
        entry=$(cat << EOF
# MOLA_CI_START: ${machine}
Host ${machine}
    HostName ${machine}
    User mola
    IdentityFile ${key}
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
# MOLA_CI_END: ${machine}

EOF
)
        SSH_CONFIG_ENTRIES="${SSH_CONFIG_ENTRIES}${entry}"
        echo "Configured SSH access to ${machine} via ${key}"
    else
        echo "WARNING: SSH key not found for ${machine}: ${key}" >&2
    fi
done

if [[ -n "$SSH_CONFIG_ENTRIES" ]]; then
    printf '%s' "$SSH_CONFIG_ENTRIES" >> "$SSH_CONFIG_FILE"
fi

# Also set RSYNC_RSH so SV.rsync (which uses rsync -avz without -e) finds the right key
export RSYNC_RSH="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

# ---------------------------------------------------------------------------
# Source environment
# ---------------------------------------------------------------------------

ENV_FILE="/stck/mola/${TAG}/env_mola.sh"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9._-]+)?$ ]]; then
    echo "ERROR: Invalid version format: $TAG" >&2
    exit 1
fi

if [[ "$MODE" != "full" && "$MODE" != "light" && "$MODE" != "postprocess" ]]; then
    echo "ERROR: Invalid mode: $MODE (expected full, light, or postprocess)" >&2
    exit 1
fi

# Source environment
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
else
    echo "WARNING: Environment file not found: $ENV_FILE, using defaults" >&2
fi

echo "=========================================="
echo "MOLA Repatriate (as mola user)"
echo "=========================================="
echo "Tag: $TAG"
echo "Mode: $MODE"
echo "=========================================="

# ---------------------------------------------------------------------------
# Execute based on mode
# ---------------------------------------------------------------------------

case "$MODE" in
    full)
        echo "Running full repatriation..."
        python3 -m mola.cicd.examples_runner repatriate --full --tag "$TAG"
        ;;
    light)
        echo "Running light repatriation..."
        python3 -m mola.cicd.examples_runner repatriate --light --tag "$TAG"
        ;;
    postprocess)
        echo "Running postprocess..."
        python3 -m mola.cicd.examples_runner postprocess --tag "$TAG"
        ;;
esac

echo "=========================================="
echo "Repatriation finished successfully."
echo "=========================================="
