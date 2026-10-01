#!/usr/bin/env bash
# load_env.sh
# Usage (recommended):
#   source /path/to/load_env.sh [elsa|...]
# Works even if this file is reached via a symlink.

# ---- guard: must be sourced (not executed) ----
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed:" >&2
  echo "  source $0 [elsa|...]" >&2
  exit 2
fi

# ---- resolve real script directory (handles symlinks) ----
get_script_dir() {
  local src="${BASH_SOURCE[0]}"
  while [[ -L "$src" ]]; do
    local dir
    dir="$(builtin cd -P -- "$(dirname -- "$src")" >/dev/null 2>&1 && pwd -P)" || return 1
    src="$(readlink -- "$src")" || return 1
    [[ "$src" != /* ]] && src="$dir/$src"
  done
  builtin cd -P -- "$(dirname -- "$src")" >/dev/null 2>&1 && pwd -P
}

ENV_NAME="${1:-elsa}"

SCRIPT_DIR="$(get_script_dir)" || { echo "Error: cannot determine script directory" >&2; return 1; }

TARGET="$SCRIPT_DIR/src/mola/env/onera/env.sh"
if [[ ! -f "$TARGET" ]]; then
  echo "Error: env script not found: $TARGET" >&2
  return 1
fi

# ---- load env ----
# shellcheck source=/dev/null
source "$TARGET" "$ENV_NAME" || {
  echo "Error: sourcing failed: $TARGET $ENV_NAME" >&2
  return 1
}

