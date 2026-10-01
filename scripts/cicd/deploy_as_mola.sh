#!/bin/bash

#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <source_path> <version>" >&2
    echo "Example: $0 /stck/lbernard/mola-v2.0.1 v2.0.1-rc1" >&2
    exit 1
fi

SOURCE_PATH="$1"
VERSION_TAG="$2"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEST_PATH="/stck/mola/${VERSION_TAG}"

JUNO_PATH="/tmp_user/juno/mola/${VERSION_TAG}"

SATOR_HOST="${SATOR_HOST:-sator}"

# Change this to lbernard if /tmp_user/sator/mola is owned by lbernard
# rather than mola.
SATOR_USER="${SATOR_USER:-mola}"

SATOR_PATH="/tmp_user/sator/mola/${VERSION_TAG}"

# This key belongs to the mola account on Juno.
# It authenticates mola@juno to Sator.
SATOR_SSH_KEY="${SATOR_SSH_KEY:-${HOME}/.ssh/sator_ci_deploy_ed25519}"

DEPLOY_SCRIPT_RELATIVE_PATH="scripts/cicd/deploy_as_mola.sh"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

if [[ ! -d "$SOURCE_PATH" ]]; then
    echo "ERROR: Source directory does not exist: $SOURCE_PATH" >&2
    exit 1
fi

if [[ ! -f "$SATOR_SSH_KEY" || ! -r "$SATOR_SSH_KEY" ]]; then
    echo "ERROR: Sator SSH key is not readable: $SATOR_SSH_KEY" >&2
    exit 1
fi

if [[ ! "$VERSION_TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9._-]+)?$ ]]; then
    echo "ERROR: Invalid version format: $VERSION_TAG" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# SSH configuration for mola -> Sator
# ---------------------------------------------------------------------------

SATOR_SSH_OPTIONS=(
    -o BatchMode=yes
    -o IdentitiesOnly=yes
    -o StrictHostKeyChecking=accept-new
    -i "$SATOR_SSH_KEY"
)

# rsync expects the SSH command as a single string.
SATOR_RSYNC_SSH="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i ${SATOR_SSH_KEY}"

SATOR_TARGET="${SATOR_USER}@${SATOR_HOST}"

# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------

RSYNC_EXCLUDE_ARGS=(
    --exclude=".git/"
    --exclude-from="${SOURCE_PATH}/.gitignore"
)

# ---------------------------------------------------------------------------
# Information
# ---------------------------------------------------------------------------

echo "=========================================="
echo "MOLA Deployment (as mola user)"
echo "=========================================="
echo "Source:       $SOURCE_PATH"
echo "Local dest:   $DEST_PATH"
echo "Juno dest:    $JUNO_PATH"
echo "Sator target: ${SATOR_TARGET}:${SATOR_PATH}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Deploy to /stck/mola
# ---------------------------------------------------------------------------

echo "Deploying to local /stck/mola..."

mkdir -p -- "$DEST_PATH"

if ! rsync -av --delete \
    "${RSYNC_EXCLUDE_ARGS[@]}" \
    "${SOURCE_PATH}/" \
    "${DEST_PATH}/"
then
    echo "ERROR: Local sync to /stck/mola failed." >&2
    exit 1
fi

echo "Local sync to /stck/mola completed."

# ---------------------------------------------------------------------------
# Deploy to Sator
# ---------------------------------------------------------------------------

echo "Preparing destination on Sator..."

if ! ssh "${SATOR_SSH_OPTIONS[@]}" \
    "$SATOR_TARGET" \
    "mkdir -p '${SATOR_PATH}'"
then
    echo "ERROR: Could not create destination directory on Sator." >&2
    exit 1
fi

echo "Syncing from Juno to Sator..."
echo "  From: ${DEST_PATH}"
echo "  To:   ${SATOR_TARGET}:${SATOR_PATH}"

# This rsync runs on Juno and transfers to Sator.
if ! rsync -av --delete \
    "${RSYNC_EXCLUDE_ARGS[@]}" \
    -e "$SATOR_RSYNC_SSH" \
    "${DEST_PATH}/" \
    "${SATOR_TARGET}:${SATOR_PATH}/"
then
    echo "ERROR: Sync to Sator failed." >&2
    exit 1
fi

echo "Sync to Sator completed."

# ---------------------------------------------------------------------------
# Deploy to Juno
# ---------------------------------------------------------------------------

echo "Syncing to Juno..."

mkdir -p -- "$JUNO_PATH"

if ! rsync -av --delete \
    "${RSYNC_EXCLUDE_ARGS[@]}" \
    "${DEST_PATH}/" \
    "${JUNO_PATH}/"
then
    echo "ERROR: Sync to Juno failed." >&2
    exit 1
fi

echo "Sync to Juno completed."

# ---------------------------------------------------------------------------
# Cleanup deployment script from deployed directories
# ---------------------------------------------------------------------------

echo "Cleaning up deployment script..."

rm -f -- \
    "${DEST_PATH}/${DEPLOY_SCRIPT_RELATIVE_PATH}"

rm -f -- \
    "${JUNO_PATH}/${DEPLOY_SCRIPT_RELATIVE_PATH}"

if ! ssh "${SATOR_SSH_OPTIONS[@]}" \
    "$SATOR_TARGET" \
    "rm -f '${SATOR_PATH}/${DEPLOY_SCRIPT_RELATIVE_PATH}'"
then
    echo "ERROR: Could not remove deployment script from Sator." >&2
    exit 1
fi

echo "Deployment script removed from deployed locations."

echo "=========================================="
echo "Deployment finished successfully."
echo "=========================================="
