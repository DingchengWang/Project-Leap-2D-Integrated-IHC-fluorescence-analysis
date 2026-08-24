#!/bin/zsh

set -euo pipefail

readonly SCRIPT_DIR="${0:A:h}"
readonly PACKAGE_DIR="${SCRIPT_DIR}/Project Leap 2D (8-23-26)"

if [[ -L "${PACKAGE_DIR}" || ! -d "${PACKAGE_DIR}" ]]; then
  print -u2 -- "Error: expected a real package directory at: ${PACKAGE_DIR}"
  exit 1
fi

readonly -a REQUIRED_DIRECTORIES=(
  "${PACKAGE_DIR}/Original Image"
  "${PACKAGE_DIR}/Result"
  "${PACKAGE_DIR}/Runtime"
  "${PACKAGE_DIR}/Runtime/locks"
  "${PACKAGE_DIR}/Runtime/recovery"
  "${PACKAGE_DIR}/Runtime/staging"
  "${PACKAGE_DIR}/Runtime/matplotlib"
)

# Validate every existing path before creating anything. This prevents a
# partially prepared workspace when a target is a symlink or a non-directory.
for target in "${REQUIRED_DIRECTORIES[@]}"; do
  if [[ -L "${target}" || ( -e "${target}" && ! -d "${target}" ) ]]; then
    print -u2 -- "Error: refusing unsafe workspace path: ${target}"
    exit 1
  fi
done

/bin/mkdir -p "${REQUIRED_DIRECTORIES[@]}"

print -- "Project Leap 2D workspace is ready:"
print -- "${PACKAGE_DIR}"
