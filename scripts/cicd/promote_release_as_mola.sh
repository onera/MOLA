#!/usr/bin/env bash

set -Eeuo pipefail

FINAL_TAG="$1"
RC_TAG="$2"

validate_tag() {
    [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

validate_rc_tag() {
    [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+-rc[0-9]+$ ]]
}

if ! validate_tag "$FINAL_TAG" || ! validate_rc_tag "$RC_TAG"; then
    echo "ERROR: invalid final or release-candidate tag" >&2
    exit 1
fi

if [[ "${RC_TAG%-rc*}" != "$FINAL_TAG" ]]; then
    echo "ERROR: $RC_TAG is not a release candidate for $FINAL_TAG" >&2
    exit 1
fi

SATOR_HOST="${SATOR_HOST:-sator}"
SATOR_USER="${SATOR_USER:-mola}"
SATOR_SSH_KEY="${SATOR_SSH_KEY:-${HOME}/.ssh/sator_ci_deploy_ed25519}"
SATOR_SSH_OPTIONS=(
    -o BatchMode=yes
    -o IdentitiesOnly=yes
    -o StrictHostKeyChecking=accept-new
    -i "$SATOR_SSH_KEY"
)

paths=(
    "/stck/mola"
    "/tmp_user/juno/mola"
)

for root in "${paths[@]}"; do
    source_path="$root/$RC_TAG"
    target_path="$root/$FINAL_TAG"
    if [[ -d "$target_path" ]]; then
        [[ ! -e "$source_path" ]] || {
            echo "ERROR: both release directories exist: $source_path and $target_path" >&2
            exit 1
        }
    elif [[ ! -d "$source_path" ]]; then
        echo "ERROR: neither release directory exists: $source_path" >&2
        exit 1
    fi
done

if [[ ! -f "$SATOR_SSH_KEY" || ! -r "$SATOR_SSH_KEY" ]]; then
    echo "ERROR: Sator SSH key is not readable: $SATOR_SSH_KEY" >&2
    exit 1
fi

echo "Promoting $RC_TAG to $FINAL_TAG on Juno."
for root in "${paths[@]}"; do
    if [[ ! -d "$root/$FINAL_TAG" ]]; then
        mv -- "$root/$RC_TAG" "$root/$FINAL_TAG"
    fi
done

echo "Promoting $RC_TAG to $FINAL_TAG on Sator."
ssh "${SATOR_SSH_OPTIONS[@]}" "${SATOR_USER}@${SATOR_HOST}" \
    "if test -d '/tmp_user/sator/mola/$FINAL_TAG'; then test ! -e '/tmp_user/sator/mola/$RC_TAG'; elif test -d '/tmp_user/sator/mola/$RC_TAG'; then mv -- '/tmp_user/sator/mola/$RC_TAG' '/tmp_user/sator/mola/$FINAL_TAG'; else exit 1; fi"

echo "Release promotion completed: $FINAL_TAG"
