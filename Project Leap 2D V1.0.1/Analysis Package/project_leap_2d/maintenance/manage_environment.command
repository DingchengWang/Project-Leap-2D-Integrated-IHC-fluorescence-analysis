#!/bin/zsh
set -euo pipefail

MAINTENANCE_DIR="${0:A:h}"
BOOTSTRAP="$MAINTENANCE_DIR/bootstrap_macos.sh"
INTEGRITY_MANIFEST="$MAINTENANCE_DIR/installer_integrity_manifest.sh"
EXPECTED_BOOTSTRAP_SHA256="9a15fbe0c22573ef67437ab486becdb155dd4f812fa32644810d82fadaf6e9df"
EXPECTED_INTEGRITY_MANIFEST_SHA256="43ccb31721ae9d6bc64404c9ca7eaebec4da9121772c1af2f5989d79882ea5d4"

fail() {
  print -u2 -- "MAINTENANCE STOPPED SAFELY: $1"
  exit 1
}

verify_entry_resource() {
  local resource_path=$1
  local expected_sha256=$2
  local resource_label=$3
  [[ -f "$resource_path" && ! -L "$resource_path" && -r "$resource_path" ]] ||
    fail "$resource_label is missing or is not a regular maintenance file; no environment files were changed."
  local observed_sha256
  observed_sha256=$(
    /usr/bin/shasum -a 256 "$resource_path" | /usr/bin/awk '{print $1}'
  )
  [[ "$observed_sha256" == "$expected_sha256" ]] ||
    fail "$resource_label is damaged; no environment files were changed."
}

verify_entry_resource \
  "$BOOTSTRAP" "$EXPECTED_BOOTSTRAP_SHA256" "bootstrap_macos.sh"
verify_entry_resource \
  "$INTEGRITY_MANIFEST" \
  "$EXPECTED_INTEGRITY_MANIFEST_SHA256" \
  "installer_integrity_manifest.sh"

exec /bin/sh "$BOOTSTRAP" "$@"
