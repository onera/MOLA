#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

VERSION="${1:-}"
SOURCE_BRANCH="${2:-${PUBLISH_SOURCE_BRANCH:-master}}"
GITHUB_REMOTE="github"
PUBLIC_BRANCH="main"

if [[ -z "$VERSION" ]]; then
    die "Usage: $0 <tag> [source-branch]"
fi

if [[ ! "$VERSION" =~ ^(v[0-9]+\.[0-9]+\.[0-9]+|cicdtest)$ ]]; then
    die "Unsupported tag '$VERSION'. Expected vX.Y.Z or cicdtest."
fi

if [[ -n "${CI_COMMIT_TAG:-}" && "$CI_COMMIT_TAG" != "$VERSION" ]]; then
    die "Argument tag '$VERSION' does not match CI_COMMIT_TAG '$CI_COMMIT_TAG'."
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || die "Run this script from inside the Git repository."
cd "$REPO_ROOT"

git check-ref-format --branch "$SOURCE_BRANCH" >/dev/null 2>&1 \
    || die "Invalid source branch name '$SOURCE_BRANCH'."

git remote get-url origin >/dev/null 2>&1 \
    || die "Git remote 'origin' is not configured."
git remote get-url "$GITHUB_REMOTE" >/dev/null 2>&1 \
    || die "Git remote '$GITHUB_REMOTE' is not configured."

# Use the exact commit that triggered this tag pipeline.
if [[ -n "${CI_COMMIT_SHA:-}" ]]; then
    SOURCE_COMMIT="$(git rev-parse --verify "${CI_COMMIT_SHA}^{commit}" 2>/dev/null)" \
        || die "CI_COMMIT_SHA is not available as a commit in this checkout."
else
    SOURCE_COMMIT="$(git rev-parse --verify "refs/tags/${VERSION}^{commit}" 2>/dev/null)" \
        || die "Tag '$VERSION' is not available locally."
fi

# Fetch the selected internal GitLab branch for validation.
git fetch --no-tags origin \
    "+refs/heads/${SOURCE_BRANCH}:refs/remotes/origin/${SOURCE_BRANCH}" \
    || die "Could not fetch origin/$SOURCE_BRANCH."

SOURCE_BRANCH_COMMIT="$(
    git rev-parse --verify "refs/remotes/origin/${SOURCE_BRANCH}^{commit}" 2>/dev/null
)" || die "Could not resolve origin/$SOURCE_BRANCH."

if [[ "$SOURCE_BRANCH_COMMIT" != "$SOURCE_COMMIT" ]]; then
    die "Tag '$VERSION' points to $SOURCE_COMMIT, but origin/$SOURCE_BRANCH points to $SOURCE_BRANCH_COMMIT. Select the branch whose tip is the tagged commit."
fi

# These paths are deliberately omitted from every public snapshot.
RESTRICTED_PATHS=(
    examples/restricted
    scripts/cicd
    .gitlab-ci.yml
)

# Fetch GitHub's current main. The new public commit will be based on it,
# so the push to main is a fast-forward.
git fetch --no-tags "$GITHUB_REMOTE" \
    "refs/heads/${PUBLIC_BRANCH}:refs/remotes/${GITHUB_REMOTE}/${PUBLIC_BRANCH}" \
    || die "Could not fetch $GITHUB_REMOTE/$PUBLIC_BRANCH. Bootstrap the public main branch first."

GITHUB_MAIN="$(
    git rev-parse --verify "refs/remotes/${GITHUB_REMOTE}/${PUBLIC_BRANCH}^{commit}" 2>/dev/null
)" || die "Could not resolve $GITHUB_REMOTE/$PUBLIC_BRANCH."

# Refuse to extend GitHub main if its reachable history already contains
# commits that changed the paths excluded from publication.
RESTRICTED_HISTORY="$(
    git log --full-history --format=%H "$GITHUB_MAIN" -- "${RESTRICTED_PATHS[@]}"
)"
if [[ -n "$RESTRICTED_HISTORY" ]]; then
    die "$GITHUB_REMOTE/$PUBLIC_BRANCH has history involving excluded paths. Clean or recreate the public repository before publishing."
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mola-github-sync.XXXXXX")"
PUBLISH_REF="refs/heads/github-publish-${VERSION}-$$"

cleanup() {
    git update-ref -d "$PUBLISH_REF" >/dev/null 2>&1 || true
    rm -rf -- "$TMP_DIR"
}
trap cleanup EXIT

# Construct an index from the tagged GitLab tree, then remove restricted paths.
# This does not check out or modify the CI working tree.
GIT_INDEX_FILE="$TMP_DIR/index" git read-tree "${SOURCE_COMMIT}^{tree}"

GIT_INDEX_FILE="$TMP_DIR/index" \
    git rm --cached -r --force --ignore-unmatch -- \
    "${RESTRICTED_PATHS[@]}" >/dev/null

PUBLIC_TREE="$(GIT_INDEX_FILE="$TMP_DIR/index" git write-tree)"

# Verify that none of the excluded paths remain in the tree to be published.
REMAINING_RESTRICTED_PATHS="$(
    git ls-tree -r --name-only "$PUBLIC_TREE" -- "${RESTRICTED_PATHS[@]}"
)"
if [[ -n "$REMAINING_RESTRICTED_PATHS" ]]; then
    printf 'ERROR: restricted paths remain in the public tree:\n%s\n' \
        "$REMAINING_RESTRICTED_PATHS" >&2
    exit 1
fi

GITHUB_MAIN_TREE="$(git rev-parse "${GITHUB_MAIN}^{tree}")"

if [[ "$PUBLIC_TREE" == "$GITHUB_MAIN_TREE" ]]; then
    NEW_COMMIT="$GITHUB_MAIN"
    echo "Public tree is unchanged; reusing current GitHub main commit."
else
    AUTHOR_NAME="${GITLAB_USER_NAME:-GitLab CI}"
    AUTHOR_EMAIL="${GITLAB_USER_EMAIL:-gitlab-ci@localhost}"

    NEW_COMMIT="$(
        git -c user.name="$AUTHOR_NAME" \
            -c user.email="$AUTHOR_EMAIL" \
            commit-tree "$PUBLIC_TREE" \
            -p "$GITHUB_MAIN" \
            -m "Publish ${VERSION} from GitLab branch ${SOURCE_BRANCH}"
    )"
fi

# A temporary local ref makes the exact new commit available for both pushes.
git update-ref "$PUBLISH_REF" "$NEW_COMMIT"

# By default, do not move an existing GitHub tag. Set ALLOW_TAG_REPLACE=true
# only when intentionally republishing an existing tag.
TAG_EXISTS=0
if git ls-remote --exit-code --refs "$GITHUB_REMOTE" \
    "refs/tags/${VERSION}" >/dev/null 2>&1; then
    TAG_EXISTS=1
    if [[ "${ALLOW_TAG_REPLACE:-false}" != "true" ]]; then
        die "GitHub tag '$VERSION' already exists. Delete it first, or set ALLOW_TAG_REPLACE=true to move it intentionally."
    fi
fi

# This ref update is fast-forward because NEW_COMMIT's only parent is GitHub main.
git push "$GITHUB_REMOTE" "${PUBLISH_REF}:refs/heads/${PUBLIC_BRANCH}"

if [[ "$TAG_EXISTS" == "1" ]]; then
    git push --force "$GITHUB_REMOTE" "${PUBLISH_REF}:refs/tags/${VERSION}"
else
    git push "$GITHUB_REMOTE" "${PUBLISH_REF}:refs/tags/${VERSION}"
fi

echo "Published sanitized tag '$VERSION' to $GITHUB_REMOTE/$PUBLIC_BRANCH."