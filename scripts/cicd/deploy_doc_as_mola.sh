#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <doc_source_path> <version>" >&2
    echo "Example: $0 /stck/lbernard/mola-doc-v2.0.1 v2.0.1" >&2
    exit 1
fi

DOC_SOURCE_PATH="$1"
VERSION_TAG="$2"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEST_PATH="/stck/mola/html/${VERSION_TAG}"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

if [[ ! -d "$DOC_SOURCE_PATH" ]]; then
    echo "ERROR: Doc source directory does not exist: $DOC_SOURCE_PATH" >&2
    exit 1
fi

if [[ ! "$VERSION_TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Invalid version format: $VERSION_TAG" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Information
# ---------------------------------------------------------------------------

echo "=========================================="
echo "MOLA Documentation Deployment (as mola user)"
echo "=========================================="
echo "Source:      $DOC_SOURCE_PATH"
echo "Destination: $DEST_PATH"
echo "=========================================="

# ---------------------------------------------------------------------------
# Deploy to /stck/mola/html
# ---------------------------------------------------------------------------

echo "Creating destination directory..."

mkdir -p -- "$DEST_PATH"

echo "Syncing documentation to $DEST_PATH..."

if ! rsync -a --delete \
    "$DOC_SOURCE_PATH/" \
    "$DEST_PATH/"
then
    echo "ERROR: Documentation sync failed." >&2
    exit 1
fi

echo "Documentation sync completed."

echo "=========================================="
echo "Documentation deployment finished successfully."
echo "=========================================="