#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <doc_dir> <version>" >&2
    echo "Example: $0 doc/html v2.0.1" >&2
    exit 1
fi

DOC_DIR="$1"
VERSION="$2"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Private key owned by $USER.
# This must not be a GitLab CI/CD variable.
LOCAL_MOLA_SSH_KEY="/stck/$USER/.ssh/mola_ci_deploy_ed25519"

# Temporary staging directory for documentation.
SHARED_DOC="/stck/${USER}/mola-doc-${VERSION}"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

if [[ ! -d "$DOC_DIR" ]]; then
    echo "ERROR: Doc directory does not exist: $DOC_DIR" >&2
    exit 1
fi

if [[ ! -f "$LOCAL_MOLA_SSH_KEY" || ! -r "$LOCAL_MOLA_SSH_KEY" ]]; then
    echo "ERROR: SSH key is not readable: $LOCAL_MOLA_SSH_KEY" >&2
    exit 1
fi

if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Invalid version format: $VERSION" >&2
    exit 1
fi

echo "=========================================="
echo "MOLA Documentation Deployment Script"
echo "=========================================="
echo "Doc dir:    $DOC_DIR"
echo "Version:    $VERSION"
echo "Target:     /stck/mola/html/$VERSION"
echo "Shared doc: $SHARED_DOC"
echo "=========================================="

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cleanup() {
    if [[ -d "$SHARED_DOC" ]]; then
        echo "Cleaning up shared doc directory: $SHARED_DOC"
        rm -rf -- "$SHARED_DOC"
    fi
}

trap cleanup EXIT

# ---------------------------------------------------------------------------
# Prepare staging directory
# ---------------------------------------------------------------------------

echo "Preparing shared doc directory..."

rm -rf -- "$SHARED_DOC"
mkdir -p -- "$SHARED_DOC"

# Copy documentation contents.
cp -a -- "$DOC_DIR/." "$SHARED_DOC/"

# Make the doc readable/traversable by mola.
chmod -R a+rX -- "$SHARED_DOC"

# Copy the deployment script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p -- "${SHARED_DOC}/scripts/cicd"
cp -- \
    "${SCRIPT_DIR}/deploy_doc_as_mola.sh" \
    "${SHARED_DOC}/scripts/cicd/deploy_doc_as_mola.sh"
chmod +x -- "${SHARED_DOC}/scripts/cicd/deploy_doc_as_mola.sh"

# ---------------------------------------------------------------------------
# Execute deployment as mola on Juno
# ---------------------------------------------------------------------------

echo "Deploying via SSH to mola@localhost..."

REMOTE_DEPLOY_SCRIPT="${SHARED_DOC}/scripts/cicd/deploy_doc_as_mola.sh"

if ! ssh \
    -4 \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no \
    -i "$LOCAL_MOLA_SSH_KEY" \
    mola@localhost \
    "'${REMOTE_DEPLOY_SCRIPT}' '${SHARED_DOC}' '${VERSION}'"
then
    echo "ERROR: Documentation deployment via SSH failed." >&2
    exit 1
fi

echo "=========================================="
echo "Documentation deployment completed successfully!"
echo "=========================================="