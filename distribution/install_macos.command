#!/bin/zsh
set -euo pipefail

# This distribution entry copies the complete working package before invoking
# its shared environment-maintenance entry. It does not download software.
PACKAGE_NAME="Project Leap 2D V1.0.1"
DISTRIBUTION_DIR="${0:A:h}"
SOURCE_PACKAGE="$DISTRIBUTION_DIR/$PACKAGE_NAME"
PAYLOAD_MANIFEST="$DISTRIBUTION_DIR/payload_sha256.txt"
MAINTENANCE_RELATIVE="Analysis Package/project_leap_2d/maintenance/manage_environment.command"
destination="$HOME/Desktop/$PACKAGE_NAME"
dry_run=0
staging_container=""
typeset -A expected_hashes

fail() {
  print -u2 -r -- "INSTALLATION STOPPED: $1"
  exit 1
}

cleanup() {
  if [[ -n "$staging_container" && -d "$staging_container" && ! -L "$staging_container" ]]; then
    /bin/rm -rf -- "$staging_container"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

usage() {
  print -r -- "Usage: install_macos.command [--dry-run] [--destination /absolute/package/path]"
  print -r -- "Default destination: $HOME/Desktop/$PACKAGE_NAME"
  print -r -- "The destination must not exist; its parent directory must already exist."
}

while (( $# > 0 )); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --destination)
      (( $# >= 2 )) || fail "--destination requires an absolute path."
      destination="$2"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown argument: $1. Use --help for usage." ;;
  esac
done

[[ "$destination" == /* ]] || fail "The destination must be an absolute path."
destination="${destination:a}"
destination_parent="${destination:h}"
destination_name="${destination:t}"
[[ ! -e "$destination" && ! -L "$destination" ]] ||
  fail "Destination already exists; nothing was overwritten: $destination"
[[ -d "$destination_parent" && ! -L "$destination_parent" && -w "$destination_parent" ]] ||
  fail "The destination parent must be an existing, writable directory, not a symbolic link: $destination_parent"
[[ -d "$SOURCE_PACKAGE" && ! -L "$SOURCE_PACKAGE" ]] ||
  fail "The accompanying working package is missing or is a symbolic link: $SOURCE_PACKAGE"
[[ "${destination_parent:A}/" != "${SOURCE_PACKAGE:A}/"* ]] ||
  fail "The destination cannot be inside the accompanying working package."
[[ -f "$SOURCE_PACKAGE/$MAINTENANCE_RELATIVE" && ! -L "$SOURCE_PACKAGE/$MAINTENANCE_RELATIVE" && -r "$SOURCE_PACKAGE/$MAINTENANCE_RELATIVE" ]] ||
  fail "The working package's environment-maintenance entry is missing or unreadable."
[[ -f "$PAYLOAD_MANIFEST" && ! -L "$PAYLOAD_MANIFEST" && -r "$PAYLOAD_MANIFEST" ]] ||
  fail "payload_sha256.txt is missing, unreadable, or a symbolic link."

# Manifest format: lowercase SHA256, two spaces, then the path relative to the
# working package. Reject ambiguous paths and duplicates before copying.
while IFS= read -r manifest_line || [[ -n "$manifest_line" ]]; do
  (( ${#manifest_line} > 66 )) || fail "Malformed payload checksum manifest."
  file_digest="${manifest_line[1,64]}"
  relative_file="${manifest_line[67,-1]}"
  [[ "$file_digest" != *[^0-9a-f]* && "${manifest_line[65,66]}" == '  ' ]] ||
    fail "Malformed payload checksum manifest."
  [[ "$relative_file" != /* && "$relative_file" != *'//'*
     && "/$relative_file/" != *'/../'* && "/$relative_file/" != *'/./'*
     && "$relative_file" != *$'\r'* && "$relative_file" != *$'\n'*
     && "$relative_file" != *'\'* ]] || fail "Unsafe path in payload checksum manifest."
  [[ -z "${expected_hashes[$relative_file]-}" ]] ||
    fail "Duplicate file in payload checksum manifest: $relative_file"
  expected_hashes[$relative_file]="$file_digest"
done < "$PAYLOAD_MANIFEST"
(( ${#expected_hashes} > 0 )) || fail "The payload checksum manifest is empty."

verify_package() {
  local package_root="$1"
  local allow_finder_metadata="${2:-0}"
  local resource relative_file checksum_output observed_digest required_directory
  local file_count=0
  [[ -d "$package_root" && ! -L "$package_root" ]] ||
    fail "The working package must be a regular directory: $package_root"
  for required_directory in "Sample Image" "Result" "Analysis Package" \
      "Analysis Package/Run State" "README"; do
    [[ -d "$package_root/$required_directory" && ! -L "$package_root/$required_directory" ]] ||
      fail "Required working-package directory is missing or is a symbolic link: $required_directory"
  done
  # D includes dotfiles, N permits an empty directory. Recursive globbing does
  # not follow symbolic-link directories; the link itself is rejected below.
  for resource in "$package_root"/**/*(DN); do
    [[ ! -L "$resource" ]] || fail "Symbolic links are not allowed in the working package: $resource"
    [[ -d "$resource" || -f "$resource" ]] || fail "Unsupported resource in the working package: $resource"
    [[ -f "$resource" ]] || continue
    relative_file="${resource#"$package_root"/}"
    if (( allow_finder_metadata )) && [[ "${resource:t}" == ".DS_Store" && -z "${expected_hashes[$relative_file]-}" ]]; then
      continue
    fi
    [[ -n "${expected_hashes[$relative_file]-}" ]] ||
      fail "Unlisted file in the working package: $relative_file"
    checksum_output=$(/usr/bin/shasum -a 256 "$resource") ||
      fail "Could not read the working package file: $relative_file"
    observed_digest="${checksum_output%% *}"
    [[ "$observed_digest" == "${expected_hashes[$relative_file]}" ]] ||
      fail "Payload checksum mismatch: $relative_file"
    (( file_count += 1 ))
  done
  (( file_count == ${#expected_hashes} )) || fail "Files listed in the payload checksum manifest are missing."
}

verify_package "$SOURCE_PACKAGE" 1
if (( dry_run )); then
  print -r -- "DRY RUN PASSED: the accompanying working package and destination passed validation."
  print -r -- "Destination: $destination"
  print -r -- "No files were written and environment setup was not run."
  exit 0
fi

staging_container=$(/usr/bin/mktemp -d "$destination_parent/.project-leap-v1.0.1.XXXXXX") ||
  fail "Could not create a private staging directory beside the destination."
staged_package="$staging_container/$destination_name"
print -r -- "Copying the working package to: $destination"
/usr/bin/ditto "$SOURCE_PACKAGE" "$staged_package" || fail "The working package could not be copied."
# Finder may add ordinary .DS_Store files after the download is extracted.
# Remove only unlisted Finder metadata from our private copy, never the source.
for finder_metadata in "$staged_package"/**/.DS_Store(DN); do
  relative_file="${finder_metadata#"$staged_package"/}"
  if [[ -f "$finder_metadata" && ! -L "$finder_metadata" && -z "${expected_hashes[$relative_file]-}" ]]; then
    /bin/rm -f -- "$finder_metadata" || fail "Could not remove Finder metadata from the staged copy."
  fi
done
verify_package "$staged_package"
[[ ! -e "$destination" && ! -L "$destination" ]] ||
  fail "The destination appeared during copying; it was not overwritten: $destination"
# The staged child has the final basename. Moving it into the existing parent
# avoids interpreting an already-created destination directory as a container.
/bin/mv -n "$staged_package" "$destination_parent/" || fail "The working package could not be published."
[[ ! -e "$staged_package" && ! -L "$staged_package" && -d "$destination" && ! -L "$destination" ]] ||
  fail "The destination was not published; an existing destination was not overwritten."
cleanup
staging_container=""

print -r -- "Working package deployed. Setting up its environment..."
if /bin/zsh "$destination/$MAINTENANCE_RELATIVE"; then
  print -r -- "INSTALLATION COMPLETE: $destination"
  print -r -- "Open Run Analysis.command in that working package to start Project Leap 2D."
else
  maintenance_status=$?
  print -u2 -r -- "ENVIRONMENT SETUP FAILED (exit $maintenance_status). Installation is not complete."
  print -u2 -r -- "The deployed working package has been retained: $destination"
  print -u2 -r -- "To retry environment repair, run:"
  print -u2 -r -- "  cd ${(q)destination} && ./Repair.command"
  exit "$maintenance_status"
fi
