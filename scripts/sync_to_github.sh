#!/usr/bin/env bash

set -Eeuo pipefail

VERSION="${1:-$(git describe --tags --abbrev=0 2>/dev/null || echo main)}"
GITHUB_REMOTE="${2:-github}"
ORIGINAL_COMMIT="$(git rev-parse HEAD)"
ORIGINAL_REF="$(git symbolic-ref --quiet --short HEAD || true)"
TEMP_BRANCH="github-sync-$(date +%Y%m%d-%H%M%S)-$$"

cleanup() {
    local status=$?
    git reset --hard "$ORIGINAL_COMMIT" >/dev/null 2>&1 || true
    if [[ -n "$ORIGINAL_REF" ]]; then
        git switch "$ORIGINAL_REF" >/dev/null 2>&1 || true
    else
        git switch --detach "$ORIGINAL_COMMIT" >/dev/null 2>&1 || true
    fi
    git branch -D "$TEMP_BRANCH" >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup EXIT

if ! git diff-index --quiet HEAD --; then
    echo "ERROR: working directory is not clean." >&2
    exit 1
fi

if [[ "$VERSION" == "main" ]]; then
    TARGET_COMMIT="$ORIGINAL_COMMIT"
else
    TARGET_COMMIT="$(git rev-parse "$VERSION")"
fi

git switch -c "$TEMP_BRANCH" "$TARGET_COMMIT"

git rm --cached -r --ignore-unmatch \
    scripts/cicd/deploy.sh \
    scripts/cicd/deploy_as_mola.sh \
    scripts/cicd/deploy_doc.sh \
    scripts/cicd/deploy_doc_as_mola.sh \
    scripts/cicd/promote_release.sh \
    scripts/cicd/promote_release_as_mola.sh \
    scripts/cicd/repatriate_data.sh \
    scripts/cicd/submit_examples.sh \
    scripts/cicd/validate_examples.sh \
    scripts/cicd/mola_ci_examples.py \
    examples/restricted \
    .gitlab-ci.yml

if ! git diff-index --quiet HEAD --; then
    git commit -m "Remove ONERA-specific files for public release [skip ci]"
fi

git push "$GITHUB_REMOTE" "HEAD:refs/heads/$TEMP_BRANCH" --force
if [[ "$VERSION" != "main" ]]; then
    git push "$GITHUB_REMOTE" "HEAD:refs/tags/$VERSION" --force
fi

echo "GitHub synchronization completed for $VERSION."
