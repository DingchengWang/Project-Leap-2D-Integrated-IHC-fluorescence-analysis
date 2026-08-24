#!/bin/sh
set -eu

MODE=${1:-}
SUPPORT_DIR=${2:-}
PROJECT_DIR=${3:-}

fail() {
  printf '%s\n' "ENVIRONMENT CHECK FAILED: $*" >&2
  exit 1
}

test "$MODE" = "--quick" ||
  fail "usage: environment_doctor.sh --quick SUPPORT_DIR PROJECT_DIR"
test -n "$SUPPORT_DIR" && test -n "$PROJECT_DIR" ||
  fail "support and project paths are required."

STATE_FILE="$SUPPORT_DIR/installation_state.json"
test -r "$STATE_FILE" || fail "installation_state.json is missing."

state_string() {
  key=$1
  /usr/bin/sed -n \
    "s/^[[:space:]]*\"${key}\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" \
    "$STATE_FILE" | /usr/bin/head -n 1
}

STATUS=$(state_string status)
SCHEMA=$(state_string schema_version)
PLATFORM=$(state_string platform)
PROJECT_VERSION=$(state_string project_version)
ENVIRONMENT_CONTRACT_ID=$(state_string environment_contract_id)
PYTHON_BIN=$(state_string python_executable)
FIJI_LAUNCHER=$(state_string fiji_launcher)
CELLPOSE_MODELS=$(state_string cellpose_models_path)
EXPECTED_CONTRACT_ID=$(
  /bin/cat "$PROJECT_DIR/Installation/macOS/environment_contract.txt"
)

test "$STATUS" = "ready" || fail "installation status is not ready."
test "$SCHEMA" = "2" || fail "installation schema is not supported."
test "$PLATFORM" = "macos-arm64" || fail "installation platform is not macos-arm64."
test -n "$PROJECT_VERSION" || fail "installation audit version is missing."
test "$ENVIRONMENT_CONTRACT_ID" = "$EXPECTED_CONTRACT_ID" ||
  fail "installation environment contract does not match this package."
test -x "$PYTHON_BIN" || fail "managed Python is missing."
test -x "$FIJI_LAUNCHER" || fail "Fiji launcher is missing."
test -f "$CELLPOSE_MODELS/cpsam_v2" || fail "Cellpose model is missing."

printf '%s\n' "Project Leap 2D environment is ready."
