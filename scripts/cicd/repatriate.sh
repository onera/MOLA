#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <repo_path> <tag> <mode> [source_user]" >&2
    echo "Modes: full | light | postprocess" >&2
    exit 1
fi

REPO_PATH="$1"
TAG="$2"
MODE="$3"
SOURCE_USER="${4:-${USER:-}}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_MOLA_SSH_KEY="/stck/$USER/.ssh/mola_ci_deploy_ed25519"
REMOTE_SCRIPT="$REPO_PATH/scripts/cicd/repatriate_as_mola.sh"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

if [[ ! -d "$REPO_PATH" ]]; then
    echo "ERROR: Repository path does not exist: $REPO_PATH" >&2
    exit 1
fi

if [[ ! -f "$REMOTE_SCRIPT" ]]; then
    echo "ERROR: repatriate_as_mola.sh script not found: $REMOTE_SCRIPT" >&2
    exit 1
fi

chmod +x $REMOTE_SCRIPT

if [[ ! -f "$LOCAL_MOLA_SSH_KEY" || ! -r "$LOCAL_MOLA_SSH_KEY" ]]; then
    echo "ERROR: SSH key is not readable: $LOCAL_MOLA_SSH_KEY" >&2
    exit 1
fi

if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9._-]+)?$ ]]; then
    echo "ERROR: Invalid version format: $TAG" >&2
    exit 1
fi

if [[ "$MODE" != "full" && "$MODE" != "light" && "$MODE" != "postprocess" ]]; then
    echo "ERROR: Invalid mode: $MODE (expected full, light, or postprocess)" >&2
    exit 1
fi

if [[ -z "$SOURCE_USER" ]]; then
    echo "ERROR: Could not determine the source user. Pass it as the fourth argument." >&2
    exit 1
fi

echo "=========================================="
echo "MOLA Repatriate Script"
echo "=========================================="
echo "Repository: $REPO_PATH"
echo "Tag:        $TAG"
echo "Mode:       $MODE"
echo "=========================================="

# ---------------------------------------------------------------------------
# Execute repatriation as mola on Juno
# ---------------------------------------------------------------------------

echo "Repatriating via SSH to mola@localhost..."

if ssh \
    -4 \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no \
    -i "$LOCAL_MOLA_SSH_KEY" \
    mola@localhost \
    "cd '$REPO_PATH' && '$REMOTE_SCRIPT' '$TAG' '$MODE' '$SOURCE_USER'"
then
    echo "=========================================="
    echo "Repatriation completed successfully!"
    echo "=========================================="
else
    SSH_EXIT=$?
    echo "" >&2
    echo "ERROR: Repatriation via SSH failed (exit code: $SSH_EXIT)." >&2
    echo "Check the output above for details on the failure." >&2
    exit 1
fi

echo "=========================================="
echo "Repatriation completed successfully!"
echo "=========================================="
