#!/bin/sh
set -eu

INSTALLATION_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$INSTALLATION_DIR/../.." && pwd)
MANIFEST="$INSTALLATION_DIR/component_manifest.sh"
LOCK_FILE="$INSTALLATION_DIR/requirements_macos_arm64.lock.txt"
CONTRACT_FILE="$INSTALLATION_DIR/environment_contract.txt"
ENVIRONMENT_INSTALLER="$INSTALLATION_DIR/environment_installer.sh"
ENVIRONMENT_DOCTOR_SHELL="$INSTALLATION_DIR/environment_doctor.sh"
INSTALLER_INTEGRITY_MANIFEST="$INSTALLATION_DIR/installer_integrity_manifest.sh"
INSTALLER_INTEGRITY_MANIFEST_SHA256="2984cca1132be096369fc6637069698be648ca2b20db51df04d1dfa08089dcea"
ENVIRONMENT_DOCTOR="$INSTALLATION_DIR/environment_doctor.py"
PYTHON_INTEGRITY_FILE="$INSTALLATION_DIR/python_wheel_integrity.json"
FIJI_INTEGRITY_FILE="$INSTALLATION_DIR/fiji_tree_integrity.json"
MANAGED_PYTHON_INTEGRITY_FILE="$INSTALLATION_DIR/managed_python_integrity.json"
SUPPORT_DIR=${PROJECT_LEAP_SUPPORT_DIR:-"$HOME/Applications/Project Leap 2D Support"}
MODE=install

case "${1:-}" in
  "")
    ;;
  --dry-run)
    MODE=dry-run
    ;;
  --check)
    MODE=check
    ;;
  *)
    printf '%s\n' "Usage: ./install_macos.command [--dry-run|--check]" >&2
    exit 64
    ;;
esac

fail() {
  printf '%s\n' "INSTALLATION STOPPED SAFELY: $*" >&2
  exit 1
}

require_plain_file() {
  file_path=$1
  file_label=$2
  test -f "$file_path" &&
    test ! -L "$file_path" &&
    test -r "$file_path" ||
    fail "$file_label is missing or is not a regular installer file; no installation files were changed."
}

verify_file_sha256() {
  file_path=$1
  expected_sha256=$2
  file_label=$3
  observed_sha256=$(
    /usr/bin/shasum -a 256 "$file_path" | /usr/bin/awk '{print $1}'
  )
  test "$observed_sha256" = "$expected_sha256" ||
    fail "$file_label is damaged; no installation files were changed."
}

test -r "$MANIFEST" || fail "component_manifest.sh is missing."
test -r "$LOCK_FILE" || fail "requirements_macos_arm64.lock.txt is missing."
test -r "$CONTRACT_FILE" || fail "environment_contract.txt is missing."
require_plain_file "$INSTALLER_INTEGRITY_MANIFEST" \
  "installer_integrity_manifest.sh"
verify_file_sha256 "$INSTALLER_INTEGRITY_MANIFEST" \
  "$INSTALLER_INTEGRITY_MANIFEST_SHA256" \
  "installer_integrity_manifest.sh"

# The fixed manifest is sourced only after its own digest has been verified.
# shellcheck source=installer_integrity_manifest.sh
. "$INSTALLER_INTEGRITY_MANIFEST"

require_plain_file "$ENVIRONMENT_INSTALLER" "environment_installer.sh"
require_plain_file "$ENVIRONMENT_DOCTOR_SHELL" "environment_doctor.sh"
require_plain_file "$ENVIRONMENT_DOCTOR" "environment_doctor.py"
require_plain_file "$MANIFEST" "component_manifest.sh"
require_plain_file "$LOCK_FILE" "requirements_macos_arm64.lock.txt"
require_plain_file "$CONTRACT_FILE" "environment_contract.txt"
require_plain_file "$PYTHON_INTEGRITY_FILE" "python_wheel_integrity.json"
require_plain_file "$FIJI_INTEGRITY_FILE" "fiji_tree_integrity.json"
require_plain_file "$MANAGED_PYTHON_INTEGRITY_FILE" \
  "managed_python_integrity.json"
verify_file_sha256 "$ENVIRONMENT_INSTALLER" \
  "${PROJECT_LEAP_ENVIRONMENT_INSTALLER_SHA256:-}" \
  "environment_installer.sh"
verify_file_sha256 "$ENVIRONMENT_DOCTOR_SHELL" \
  "${PROJECT_LEAP_ENVIRONMENT_DOCTOR_SHELL_SHA256:-}" \
  "environment_doctor.sh"
verify_file_sha256 "$ENVIRONMENT_DOCTOR" \
  "${PROJECT_LEAP_ENVIRONMENT_DOCTOR_SHA256:-}" \
  "environment_doctor.py"
verify_file_sha256 "$MANIFEST" \
  "${PROJECT_LEAP_COMPONENT_MANIFEST_SHA256:-}" \
  "component_manifest.sh"
verify_file_sha256 "$LOCK_FILE" \
  "${PROJECT_LEAP_REQUIREMENTS_LOCK_SHA256:-}" \
  "requirements_macos_arm64.lock.txt"
verify_file_sha256 "$CONTRACT_FILE" \
  "${PROJECT_LEAP_ENVIRONMENT_CONTRACT_SHA256:-}" \
  "environment_contract.txt"
verify_file_sha256 "$PYTHON_INTEGRITY_FILE" \
  "${PROJECT_LEAP_PYTHON_WHEEL_INTEGRITY_SHA256:-}" \
  "python_wheel_integrity.json"
verify_file_sha256 "$MANAGED_PYTHON_INTEGRITY_FILE" \
  "${PROJECT_LEAP_MANAGED_PYTHON_INTEGRITY_SHA256:-}" \
  "managed_python_integrity.json"
verify_file_sha256 "$FIJI_INTEGRITY_FILE" \
  "${PROJECT_LEAP_FIJI_TREE_INTEGRITY_SHA256:-}" \
  "fiji_tree_integrity.json"

# shellcheck source=component_manifest.sh
. "$MANIFEST"

test "$(uname -s)" = "Darwin" || fail "this installer supports macOS only."
test "$(uname -m)" = "arm64" || fail "this installer supports Apple Silicon only."

MACOS_MAJOR=$(/usr/bin/sw_vers -productVersion | /usr/bin/cut -d. -f1)
case "$MACOS_MAJOR" in
  ''|*[!0-9]*) fail "macOS version could not be read." ;;
esac
test "$MACOS_MAJOR" -ge "$PROJECT_LEAP_MINIMUM_MACOS_MAJOR" ||
  fail "macOS $PROJECT_LEAP_MINIMUM_MACOS_MAJOR or later is required."

validate_sha() {
  value=$1
  label=$2
  test "${#value}" -eq 64 ||
    fail "$label does not have an audited SHA-256."
  case "$value" in
    *[!0-9a-f]*) fail "$label does not have an audited SHA-256." ;;
  esac
}

validate_manifest() {
  test -n "$PROJECT_LEAP_UV_URL" || fail "the fixed uv URL is missing."
  validate_sha "$PROJECT_LEAP_UV_SHA256" "uv"
  test -n "$PROJECT_LEAP_CELLPOSE_MODEL_URL" ||
    fail "the fixed Cellpose model URL is missing."
  validate_sha "$PROJECT_LEAP_CELLPOSE_MODEL_SHA256" "Cellpose model"
  test -n "$PROJECT_LEAP_FIJI_URL" ||
    fail "Fiji is intentionally blocked until an official fixed archive URL is reviewed."
  validate_sha "$PROJECT_LEAP_FIJI_SHA256" "Fiji"
}

compute_environment_contract() {
  {
    printf 'schema=%s\n' "$PROJECT_LEAP_INSTALL_SCHEMA"
    /bin/cat "$MANIFEST"
    printf '\n--dependency-lock--\n'
    /bin/cat "$LOCK_FILE"
  } | /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'
}

ENVIRONMENT_CONTRACT_ID=$(/bin/cat "$CONTRACT_FILE")
OBSERVED_CONTRACT_ID=$(compute_environment_contract)
test "$ENVIRONMENT_CONTRACT_ID" = "$OBSERVED_CONTRACT_ID" ||
  fail "environment_contract.txt does not match the component and dependency contract."

if test "$MODE" = "dry-run"; then
  printf '%s\n' "Project Leap 2D macOS installation dry run"
  printf '%s\n' "  Project: $PROJECT_DIR"
  printf '%s\n' "  Support: $SUPPORT_DIR"
  printf '%s\n' "  Platform: $PROJECT_LEAP_PLATFORM"
  printf '%s\n' "  Managed Python: $PROJECT_LEAP_PYTHON_VERSION"
  printf '%s\n' "  uv: $PROJECT_LEAP_UV_VERSION (fixed archive and SHA-256 present)"
  printf '%s\n' "  Dependencies: exact macOS arm64 lock file"
  printf '%s\n' "  Environment contract: $ENVIRONMENT_CONTRACT_ID"
  printf '%s\n' "  Cellpose model: fixed expected SHA-256 present"
  printf '%s\n' "  Fiji: fixed archive and SHA-256 present"
  printf '%s\n' "  No files were downloaded or installed."
  exit 0
fi

if test "$MODE" = "check"; then
  exec "$INSTALLATION_DIR/environment_doctor.sh" --quick "$SUPPORT_DIR" "$PROJECT_DIR"
fi

# Validate every external download contract before creating a directory or
# contacting the network. Missing provenance therefore leaves the machine
# unchanged.
validate_manifest

exec "$INSTALLATION_DIR/environment_installer.sh" \
  "$SUPPORT_DIR" "$PROJECT_DIR" "$INSTALLATION_DIR" "$ENVIRONMENT_CONTRACT_ID"
