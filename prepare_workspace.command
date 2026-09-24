#!/bin/zsh

set -euo pipefail

readonly SCRIPT_DIR="${0:A:h}"
readonly PACKAGE_DIR="${SCRIPT_DIR}/Project Leap 2D V1.0.1"

if [[ -L "${PACKAGE_DIR}" || ! -d "${PACKAGE_DIR}" ]]; then
  print -u2 -- "Error: expected a real package directory at: ${PACKAGE_DIR}"
  exit 1
fi

readonly ANALYSIS_DIR="${PACKAGE_DIR}/Analysis Package"
if [[ -L "${ANALYSIS_DIR}" || ! -d "${ANALYSIS_DIR}" ]]; then
  print -u2 -- "Error: expected a real program directory at: ${ANALYSIS_DIR}"
  exit 1
fi

readonly -a REQUIRED_DIRECTORIES=(
  "${PACKAGE_DIR}/Sample Image"
  "${PACKAGE_DIR}/Result"
  "${ANALYSIS_DIR}/Run State"
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
