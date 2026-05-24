#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

DRY_RUN=0
RECREATE=0
INSTALL_MAIN=0
INSTALL_LIBERO=0
INSTALL_RLDS=0
FETCH_LIBERO=0

usage() {
  cat <<'EOF'
Install Python environments for this repository.

Usage:
  ./install_env.sh [--main] [--libero] [--rlds] [--all] [--fetch-libero] [--recreate] [--dry-run]

Defaults:
  With no environment selection flags, installs only the main .venv used for
  training and tests.

Options:
  --main          Install .venv from the root pyproject runtime+dev groups.
  --libero        Install .venv-libero-plus for LIBERO rerender/eval scripts.
  --rlds          Install .venv-rlds for legacy RLDS conversion utilities.
  --all           Install main, LIBERO, and RLDS environments.
  --fetch-libero  Clone the official LIBERO source into openpi/third_party/libero
                  if it is not already present.
  --recreate      Remove selected existing environments before reinstalling.
  --dry-run       Print commands without executing them.
  -h, --help      Show this message.

This public release intentionally does not create a VGGT environment.
EOF
}

log() {
  printf '[env] %s\n' "$*"
}

warn() {
  printf '[env][warn] %s\n' "$*" >&2
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run]'
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --main)
        INSTALL_MAIN=1
        ;;
      --libero)
        INSTALL_LIBERO=1
        ;;
      --rlds)
        INSTALL_RLDS=1
        ;;
      --all)
        INSTALL_MAIN=1
        INSTALL_LIBERO=1
        INSTALL_RLDS=1
        ;;
      --fetch-libero)
        FETCH_LIBERO=1
        ;;
      --recreate)
        RECREATE=1
        ;;
      --dry-run)
        DRY_RUN=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        printf 'Unknown option: %s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done

  if [[ "$INSTALL_MAIN" -eq 0 && "$INSTALL_LIBERO" -eq 0 && "$INSTALL_RLDS" -eq 0 ]]; then
    INSTALL_MAIN=1
  fi
}

require_tool() {
  local tool_name="$1"
  if ! command -v "$tool_name" >/dev/null 2>&1; then
    printf 'Error: required tool not found in PATH: %s\n' "$tool_name" >&2
    exit 1
  fi
}

maybe_remove_env() {
  local env_dir="$1"
  if [[ "$RECREATE" -eq 1 && -e "$env_dir" ]]; then
    log "Removing ${env_dir}"
    run_cmd rm -rf "$env_dir"
  fi
}

install_main_env() {
  local env_dir="${ROOT_DIR}/.venv"
  log "Installing main environment: .venv"
  maybe_remove_env "$env_dir"
  run_cmd uv venv --python 3.11 "$env_dir"
  run_cmd env UV_PROJECT_ENVIRONMENT="$env_dir" uv sync --project "$ROOT_DIR"
}

install_rlds_env() {
  local env_dir="${ROOT_DIR}/.venv-rlds"
  log "Installing RLDS environment: .venv-rlds"
  maybe_remove_env "$env_dir"
  run_cmd uv venv --python 3.11 "$env_dir"
  run_cmd env UV_PROJECT_ENVIRONMENT="$env_dir" uv sync --project "$ROOT_DIR" --no-default-groups --group rlds
}

fetch_libero_source() {
  local target="${ROOT_DIR}/openpi/third_party/libero"
  if [[ -d "$target" ]]; then
    log "LIBERO source already exists: ${target}"
    return 0
  fi
  require_tool git
  run_cmd mkdir -p "${ROOT_DIR}/openpi/third_party"
  run_cmd git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$target"
}

install_libero_env() {
  local env_dir="${ROOT_DIR}/.venv-libero-plus"
  local compat_link="${ROOT_DIR}/venv-libero-plus"

  log "Installing LIBERO rerender/eval environment: .venv-libero-plus"
  maybe_remove_env "$env_dir"
  run_cmd uv venv --python 3.11 "$env_dir"

  run_cmd env UV_PROJECT_ENVIRONMENT="$env_dir" uv sync --project "$ROOT_DIR" --no-default-groups --group runtime

  run_cmd uv pip install --python "$env_dir/bin/python" \
    requests tqdm pyarrow scipy scikit-image pillow opencv-python matplotlib \
    future cloudpickle easydict termcolor pyyaml h5py wand usd-core \
    bddl==1.0.1 robomimic==0.2.0 robosuite==1.4.0 hydra-core==1.2.0 \
    gym==0.25.2

  if [[ -d "${ROOT_DIR}/LIBERO-plus" ]]; then
    run_cmd uv pip install --python "$env_dir/bin/python" --no-deps -e "${ROOT_DIR}/LIBERO-plus"
  else
    warn "LIBERO-plus/ is not present; skipping editable install. Place a compatible LIBERO-plus checkout there before running LIBERO-plus eval scripts."
  fi

  if [[ "$FETCH_LIBERO" -eq 1 ]]; then
    fetch_libero_source
  elif [[ ! -d "${ROOT_DIR}/openpi/third_party/libero" ]]; then
    warn "openpi/third_party/libero is not present. Use --fetch-libero before rerendering original LIBERO states."
  fi

  if [[ -e "$compat_link" && ! -L "$compat_link" ]]; then
    warn "${compat_link} exists and is not a symlink; leaving it untouched."
  else
    run_cmd ln -sfn .venv-libero-plus "$compat_link"
  fi
}

main() {
  parse_args "$@"
  require_tool uv

  cd "$ROOT_DIR"

  if [[ "$INSTALL_MAIN" -eq 1 ]]; then
    install_main_env
  fi
  if [[ "$INSTALL_RLDS" -eq 1 ]]; then
    install_rlds_env
  fi
  if [[ "$INSTALL_LIBERO" -eq 1 ]]; then
    install_libero_env
  elif [[ "$FETCH_LIBERO" -eq 1 ]]; then
    fetch_libero_source
  fi

  log "Done."
}

main "$@"
