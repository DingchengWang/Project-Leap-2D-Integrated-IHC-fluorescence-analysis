#!/bin/zsh
set -euo pipefail

INSTALLATION_DIR="${0:A:h}"
BOOTSTRAP="$INSTALLATION_DIR/bootstrap_macos.sh"
INTEGRITY_MANIFEST="$INSTALLATION_DIR/installer_integrity_manifest.sh"
EXPECTED_BOOTSTRAP_SHA256="0757b79210a5c351144cdaf9dbd4d1e1fab4e7befc3fa611f9d483f4650af50d"
EXPECTED_INTEGRITY_MANIFEST_SHA256="2984cca1132be096369fc6637069698be648ca2b20db51df04d1dfa08089dcea"

fail() {
  print -u2 -- "INSTALLATION STOPPED SAFELY: $1"
  exit 1
}

verify_entry_resource() {
  local resource_path=$1
  local expected_sha256=$2
  local resource_label=$3
  [[ -f "$resource_path" && ! -L "$resource_path" && -r "$resource_path" ]] ||
    fail "$resource_label is missing or is not a regular installer file; no installation files were changed."
  local observed_sha256
  observed_sha256=$(
    /usr/bin/shasum -a 256 "$resource_path" | /usr/bin/awk '{print $1}'
  )
  [[ "$observed_sha256" == "$expected_sha256" ]] ||
    fail "$resource_label is damaged; no installation files were changed."
}

verify_entry_resource \
  "$BOOTSTRAP" "$EXPECTED_BOOTSTRAP_SHA256" "bootstrap_macos.sh"
verify_entry_resource \
  "$INTEGRITY_MANIFEST" \
  "$EXPECTED_INTEGRITY_MANIFEST_SHA256" \
  "installer_integrity_manifest.sh"

exec /bin/sh "$BOOTSTRAP" "$@"
