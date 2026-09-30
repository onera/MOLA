#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <repository_path> <version>" >&2
    echo "Example: $0 /tmp_user/.../mola v2.0.1-rc1" >&2
    exit 1
fi

REPO_PATH="$1"
VERSION="$2"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Private key owned by $USER.
# This must not be a GitLab CI/CD variable.
LOCAL_MOLA_SSH_KEY="/stck/$USER/.ssh/mola_ci_deploy_ed25519"

# Temporary shared source directory visible to mola.
SHARED_SRC="/stck/${USER}/mola-${VERSION}"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

if [[ ! -d "$REPO_PATH" ]]; then
    echo "ERROR: Repository path does not exist: $REPO_PATH" >&2
    exit 1
fi

if [[ ! -f "$LOCAL_MOLA_SSH_KEY" || ! -r "$LOCAL_MOLA_SSH_KEY" ]]; then
    echo "ERROR: SSH key is not readable: $LOCAL_MOLA_SSH_KEY" >&2
    exit 1
fi

# Avoid accidentally deleting an unexpected directory.
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9._-]+)?$ ]]; then
    echo "ERROR: Invalid version format: $VERSION" >&2
    exit 1
fi

echo "=========================================="
echo "MOLA Deployment Script"
echo "=========================================="
echo "Repository: $REPO_PATH"
echo "Version:    $VERSION"
echo "Target:     /stck/mola/${VERSION}"
echo "Shared src: $SHARED_SRC"
echo "=========================================="

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cleanup() {
    if [[ -d "$SHARED_SRC" ]]; then
        echo "Cleaning up shared source directory: $SHARED_SRC"
        rm -rf -- "$SHARED_SRC"
    fi
}

trap cleanup EXIT

# ---------------------------------------------------------------------------
# Prepare source directory
# ---------------------------------------------------------------------------

echo "Preparing shared source directory..."

rm -rf -- "$SHARED_SRC"
mkdir -p -- "$SHARED_SRC"

# Copy repository contents, including hidden files.
cp -a -- "${REPO_PATH}/." "${SHARED_SRC}/"

# Make the source readable/traversable by mola.
chmod -R a+rX -- "$SHARED_SRC"

# Copy the deployment script explicitly.
mkdir -p -- "${SHARED_SRC}/scripts/cicd"

cp -- \
    "${REPO_PATH}/scripts/cicd/deploy_as_mola.sh" \
    "${SHARED_SRC}/scripts/cicd/deploy_as_mola.sh"

chmod +x -- "${SHARED_SRC}/scripts/cicd/deploy_as_mola.sh"

# ---------------------------------------------------------------------------
# Execute deployment as mola on Juno
# ---------------------------------------------------------------------------

echo "Deploying via SSH to mola@localhost..."

REMOTE_DEPLOY_SCRIPT="${SHARED_SRC}/scripts/cicd/deploy_as_mola.sh"

if ! ssh \
    -4 \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no \
    -i "$LOCAL_MOLA_SSH_KEY" \
    mola@localhost \
    "'${REMOTE_DEPLOY_SCRIPT}' '${SHARED_SRC}' '${VERSION}'"
then
    echo "ERROR: Deployment via SSH failed." >&2
    exit 1
fi

echo "=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo "To activate this version, run:"
echo "  source /stck/mola/${VERSION}/env_mola.sh"
echo "=========================================="