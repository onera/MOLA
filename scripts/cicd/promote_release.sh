#!/usr/bin/env bash

set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <final-tag> [release-candidate-tag]" >&2
    exit 1
fi

FINAL_TAG="$1"
RC_TAG="${2:-${RELEASE_CANDIDATE_TAG:-}}"

if [[ -z "$RC_TAG" ]]; then
    mapfile -t candidates < <(
        git tag --list "${FINAL_TAG}-rc[0-9]*" --sort=-v:refname
    )
    RC_TAG="${candidates[0]:-}"
fi

if [[ -z "$RC_TAG" ]]; then
    mapfile -t candidates < <(
        git ls-remote --tags origin "refs/tags/${FINAL_TAG}-rc[0-9]*" 2>/dev/null \
        | sed -n 's#.*refs/tags/##p' \
        | sort -V -r
    )
    RC_TAG="${candidates[0]:-}"
fi

if [[ -z "$RC_TAG" ]]; then
    echo "ERROR: no release-candidate tag found for $FINAL_TAG; pass it explicitly" >&2
    exit 1
fi

if [[ "$(id -un)" == "mola" ]]; then
    exec "$(dirname "$0")/promote_release_as_mola.sh" "$FINAL_TAG" "$RC_TAG"
fi

LOCAL_MOLA_SSH_KEY="${MOLA_DEPLOY_SSH_KEY:-/stck/$USER/.ssh/mola_ci_deploy_ed25519}"
[[ -f "$LOCAL_MOLA_SSH_KEY" && -r "$LOCAL_MOLA_SSH_KEY" ]] || {
    echo "ERROR: SSH key is not readable: $LOCAL_MOLA_SSH_KEY" >&2
    exit 1
}

ssh -4 \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no \
    -i "$LOCAL_MOLA_SSH_KEY" \
    mola@localhost bash -s -- "$FINAL_TAG" "$RC_TAG" \
    < "$(dirname "$0")/promote_release_as_mola.sh"
