#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
SUPPORT_DIR="${PROJECT_LEAP_SUPPORT_DIR:-$HOME/Applications/Project Leap 2D Support}"
STATE_FILE="$SUPPORT_DIR/installation_state.json"

if [[ ! -r "$STATE_FILE" ]]; then
  print -u2 "Project Leap 2D is not installed for this macOS account."
  print -u2 "Run: ./Installation/macOS/install_macos.command"
  exit 1
fi

typeset -A INSTALLATION_STATE
while IFS= read -r state_line; do
  [[ "$state_line" == *\"*:*\"* ]] || continue
  state_key="${state_line#*\"}"
  state_key="${state_key%%\"*}"
  case "$state_key" in
    status|schema_version|platform|environment_contract_id|python_executable|fiji_launcher|cellpose_models_path)
      state_value="${state_line#*:}"
      state_value="${state_value#*\"}"
      state_value="${state_value%%\"*}"
      INSTALLATION_STATE[$state_key]="$state_value"
      ;;
  esac
done < "$STATE_FILE"

STATE_STATUS="${INSTALLATION_STATE[status]:-}"
STATE_SCHEMA="${INSTALLATION_STATE[schema_version]:-}"
STATE_PLATFORM="${INSTALLATION_STATE[platform]:-}"
STATE_ENVIRONMENT_CONTRACT="${INSTALLATION_STATE[environment_contract_id]:-}"
PYTHON_BIN="${INSTALLATION_STATE[python_executable]:-}"
FIJI_LAUNCHER="${INSTALLATION_STATE[fiji_launcher]:-}"
CELLPOSE_MODELS="${INSTALLATION_STATE[cellpose_models_path]:-}"
EXPECTED_ENVIRONMENT_CONTRACT="$(
  <"$PROJECT_DIR/Installation/macOS/environment_contract.txt"
)"

if [[ "$STATE_STATUS" != "ready" ||
      "$STATE_SCHEMA" != "2" ||
      "$STATE_PLATFORM" != "macos-arm64" ||
      "$STATE_ENVIRONMENT_CONTRACT" != "$EXPECTED_ENVIRONMENT_CONTRACT" ||
      ! -x "$PYTHON_BIN" ||
      ! -x "$FIJI_LAUNCHER" ||
      ! -f "$CELLPOSE_MODELS/cpsam_v2" ]]; then
  print -u2 "Project Leap 2D installation is incomplete or its environment contract changed."
  print -u2 "Run: ./Installation/macOS/install_macos.command"
  exit 1
fi

if [[ "${1:-}" == "--check-environment" ]]; then
  print "Project Leap 2D environment is ready."
  exit 0
fi

ENVIRONMENT_USAGE_LOCK="$SUPPORT_DIR/Environment Usage Lock"
if [[ ! -x /usr/bin/lockf ]]; then
  print -u2 "Project Leap 2D cannot find the macOS lock utility."
  exit 1
fi
if [[ -L "$ENVIRONMENT_USAGE_LOCK" ||
      ( -e "$ENVIRONMENT_USAGE_LOCK" && ! -f "$ENVIRONMENT_USAGE_LOCK" ) ]]; then
  print -u2 "Project Leap 2D found an unsafe environment lock path and left it unchanged."
  exit 1
fi
exec 9>>"$ENVIRONMENT_USAGE_LOCK" || {
  print -u2 "Project Leap 2D could not open the environment lock."
  exit 1
}
if ! /usr/bin/lockf -s -t 0 9; then
  print -u2 "Project Leap 2D installation or analysis is already using this environment."
  exit 1
fi

mkdir -p "$PROJECT_DIR/Runtime/matplotlib"
export MPLCONFIGDIR="$PROJECT_DIR/Runtime/matplotlib"
export CELLPOSE_LOCAL_MODELS_PATH="$CELLPOSE_MODELS"
export PYTHONDONTWRITEBYTECODE=1
export PROJECT_LEAP_USAGE_LOCK_FILE="$ENVIRONMENT_USAGE_LOCK"
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m project_leap_2d --fiji-launcher "$FIJI_LAUNCHER" "$@"
