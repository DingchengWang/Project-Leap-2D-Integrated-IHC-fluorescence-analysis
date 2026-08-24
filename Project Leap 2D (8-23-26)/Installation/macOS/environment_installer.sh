#!/bin/sh
set -eu

SUPPORT_DIR=$1
PROJECT_DIR=$2
INSTALLATION_DIR=$3
ENVIRONMENT_CONTRACT_ID=$4
MANIFEST="$INSTALLATION_DIR/component_manifest.sh"
LOCK_FILE="$INSTALLATION_DIR/requirements_macos_arm64.lock.txt"

# shellcheck source=component_manifest.sh
. "$MANIFEST"

fail() {
  printf '%s\n' "INSTALLATION STOPPED SAFELY: $*" >&2
  exit 1
}

download_verified() {
  url=$1
  expected_sha=$2
  destination=$3
  label=$4
  /usr/bin/curl \
    --fail --location --progress-bar --show-error --proto '=https' --tlsv1.2 \
    --output "$destination" "$url" ||
    fail "$label download failed."
  observed_sha=$(/usr/bin/shasum -a 256 "$destination" | /usr/bin/awk '{print $1}')
  test "$observed_sha" = "$expected_sha" ||
    fail "$label integrity check failed; the downloaded file was not used."
}

PROJECT_VERSION=$(/bin/cat "$PROJECT_DIR/VERSION")
LOCK_SHA=$(/usr/bin/shasum -a 256 "$LOCK_FILE" | /usr/bin/awk '{print $1}')
RELEASE_ID="environment-$(printf '%s' "$ENVIRONMENT_CONTRACT_ID" | /usr/bin/cut -c1-16)"
RELEASES_DIR="$SUPPORT_DIR/Releases"
RELEASE_DIR="$RELEASES_DIR/$RELEASE_ID"
USAGE_LOCK_FILE="$SUPPORT_DIR/Environment Usage Lock"
STATE_FILE="$SUPPORT_DIR/installation_state.json"
STATE_NEXT=
STATE_STAGE_DIR=
REPAIR_BACKUP_DIR="$RELEASES_DIR/$RELEASE_ID Repair Backup"
REPAIR_STATE_DIR="$SUPPORT_DIR/Repair State $RELEASE_ID"
REPAIR_STATE_STAGE_DIR=
INSTALLING_MARKER_STAGE=
DOCTOR_CACHE_DIR="$PROJECT_DIR/Runtime/matplotlib"
INSTALL_COMPLETE=0
REPAIR_PREPARING=0
REPAIR_ACTIVE=0
REPAIR_ATTEMPTED=0
LOCK_ACQUIRED=0
PRESERVE_UNCERTAIN=0

state_file_string() {
  file=$1
  key=$2
  /usr/bin/sed -n \
    "s/^[[:space:]]*\"${key}\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" \
    "$file" | /usr/bin/head -n 1
}

create_state_next() {
  test -z "$STATE_STAGE_DIR" && test -z "$STATE_NEXT" ||
    return 1
  old_umask=$(umask)
  umask 077
  STATE_STAGE_DIR=$(
    /usr/bin/mktemp -d "$SUPPORT_DIR/State Publication.XXXXXX"
  ) || {
    umask "$old_umask"
    return 1
  }
  umask "$old_umask"
  STATE_NEXT="$STATE_STAGE_DIR/installation_state.json"
}

state_file_claims_release() {
  file=$1
  test -r "$file" &&
    test "$(state_file_string "$file" status)" = "ready" &&
    test "$(state_file_string "$file" schema_version)" = "$PROJECT_LEAP_INSTALL_SCHEMA" &&
    test "$(state_file_string "$file" platform)" = "$PROJECT_LEAP_PLATFORM" &&
    test "$(state_file_string "$file" environment_contract_id)" = "$ENVIRONMENT_CONTRACT_ID" &&
    test "$(state_file_string "$file" python_executable)" = "$RELEASE_DIR/Environment/bin/python3" &&
    test "$(state_file_string "$file" fiji_launcher)" = "$RELEASE_DIR/$PROJECT_LEAP_FIJI_LAUNCHER_RELATIVE" &&
    test "$(state_file_string "$file" cellpose_models_path)" = "$RELEASE_DIR/Models"
}

state_claims_release() {
  state_file_claims_release "$STATE_FILE"
}

installing_marker_is_owned() {
  marker="$RELEASE_DIR/INSTALLING"
  test -f "$marker" &&
    test ! -L "$marker" &&
    test "$(/usr/bin/sed -n 's/^environment_contract_id=//p' "$marker" | /usr/bin/head -n 1)" = "$ENVIRONMENT_CONTRACT_ID" &&
    test "$(/usr/bin/sed -n 's/^release_dir=//p' "$marker" | /usr/bin/head -n 1)" = "$RELEASE_DIR"
}

write_installing_marker() {
  old_umask=$(umask)
  umask 077
  INSTALLING_MARKER_STAGE=$(
    /usr/bin/mktemp "$RELEASES_DIR/Installing Marker.XXXXXX"
  ) || {
    umask "$old_umask"
    return 1
  }
  umask "$old_umask"
  {
    printf 'environment_contract_id=%s\n' "$ENVIRONMENT_CONTRACT_ID"
    printf 'release_dir=%s\n' "$RELEASE_DIR"
  } > "$INSTALLING_MARKER_STAGE" || return 1
  /bin/mv "$INSTALLING_MARKER_STAGE" "$RELEASE_DIR/INSTALLING" || return 1
  INSTALLING_MARKER_STAGE=
}

acquire_environment_lock() {
  test -x /usr/bin/lockf ||
    fail "the macOS lock utility is unavailable; no installation files were changed."
  test ! -L "$USAGE_LOCK_FILE" ||
    fail "the environment lock path is a symbolic link and was not changed."
  if test -e "$USAGE_LOCK_FILE"; then
    test -f "$USAGE_LOCK_FILE" ||
      fail "the environment lock path is not a regular file."
  fi
  # The file remains as fixed infrastructure. The kernel-held lock, rather
  # than file contents, owns the critical section and is released on every
  # normal exit, signal, exec, or process crash.
  exec 9>>"$USAGE_LOCK_FILE" ||
    fail "the environment lock could not be opened."
  /usr/bin/lockf -s -t 0 9 ||
    fail "another installation or analysis is already using $SUPPORT_DIR."
  LOCK_ACQUIRED=1
}

analysis_is_running() {
  analysis_lock="$PROJECT_DIR/Runtime/locks/workspace.lock"
  test -r "$analysis_lock" || return 1
  analysis_pid=$(
    /usr/bin/sed -n 's/^pid=\([0-9][0-9]*\)[[:space:]].*/\1/p' \
      "$analysis_lock" | /usr/bin/head -n 1
  )
  test -n "$analysis_pid" || return 1
  /bin/kill -0 "$analysis_pid" 2>/dev/null
}

require_analysis_idle() {
  analysis_is_running &&
    fail "an analysis is currently using this Project Leap 2D package; finish it before installation or repair."
  return 0
}

repair_state_is_owned() {
  test -d "$REPAIR_STATE_DIR" &&
    test ! -L "$REPAIR_STATE_DIR" &&
    test ! -L "$REPAIR_BACKUP_DIR" &&
    test -r "$REPAIR_STATE_DIR/environment_contract_id" &&
    test "$(/bin/cat "$REPAIR_STATE_DIR/environment_contract_id")" = "$ENVIRONMENT_CONTRACT_ID" &&
    test -r "$REPAIR_STATE_DIR/release_dir" &&
    test "$(/bin/cat "$REPAIR_STATE_DIR/release_dir")" = "$RELEASE_DIR" &&
    test -r "$REPAIR_STATE_DIR/backup_dir" &&
    test "$(/bin/cat "$REPAIR_STATE_DIR/backup_dir")" = "$REPAIR_BACKUP_DIR" &&
    state_file_claims_release "$REPAIR_STATE_DIR/installation_state.json"
}

run_full_check() {
  check_release=$1
  test -d "$check_release" || return 21
  test ! -L "$check_release" || return 22
  check_python="$check_release/Environment/bin/python3"
  check_managed_python="$check_release/Managed Python/cpython-3.9.25-macos-aarch64-none/bin/python3.9"
  check_libpython="$check_release/Managed Python/cpython-3.9.25-macos-aarch64-none/lib/libpython3.9.dylib"
  check_fiji="$check_release/$PROJECT_LEAP_FIJI_LAUNCHER_RELATIVE"
  test -x "$check_python" || return 21
  test -x "$check_managed_python" || return 21
  test -f "$check_libpython" || return 21
  test -x "$check_fiji" || return 21

  cache_found=0
  cache_root="$check_release/Managed Python"
  if test -n "$(
    /usr/bin/find "$cache_root" \
      -type d -name __pycache__ -print -quit 2>/dev/null
  )"; then
    cache_found=1
    /usr/bin/find "$cache_root" \
      -type d -name __pycache__ -prune \
      -exec /bin/rm -rf {} + ||
      return 22
  fi
  if test "$cache_found" -eq 1; then
    printf '%s\n' \
      "Removed rebuildable Python bytecode caches before verification."
  fi

  /bin/mkdir -p "$DOCTOR_CACHE_DIR" || return 22
  check_status=0
  /usr/bin/env \
    PYTHONDONTWRITEBYTECODE=1 \
    "$check_managed_python" \
    -B -I -S \
    "$INSTALLATION_DIR/environment_doctor.py" \
    --mode content \
    --project-dir "$PROJECT_DIR" \
    --release-dir "$check_release" \
    --lock-file "$LOCK_FILE" \
    --cellpose-sha256 "$PROJECT_LEAP_CELLPOSE_MODEL_SHA256" \
    --fiji-launcher "$check_fiji" \
    --fiji-zip-sha256 "$PROJECT_LEAP_FIJI_SHA256" ||
    check_status=$?
  test "$check_status" -eq 0 || return "$check_status"

  /usr/bin/env \
    CELLPOSE_LOCAL_MODELS_PATH="$check_release/Models" \
    MPLCONFIGDIR="$DOCTOR_CACHE_DIR" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$check_python" \
    -B -I \
    "$INSTALLATION_DIR/environment_doctor.py" \
    --mode smoke \
    --project-dir "$PROJECT_DIR" \
    --release-dir "$check_release" \
    --lock-file "$LOCK_FILE" \
    --cellpose-sha256 "$PROJECT_LEAP_CELLPOSE_MODEL_SHA256" \
    --fiji-launcher "$check_fiji" \
    --fiji-zip-sha256 "$PROJECT_LEAP_FIJI_SHA256" ||
    check_status=$?
  return "$check_status"
}

restore_previous_release() {
  test -d "$REPAIR_BACKUP_DIR" || return 1
  test ! -L "$REPAIR_BACKUP_DIR" || return 1
  if test -e "$RELEASE_DIR"; then
    test ! -L "$RELEASE_DIR" || return 1
    installing_marker_is_owned || return 1
    /bin/rm -rf "$RELEASE_DIR" || return 1
  fi
  /bin/mv "$REPAIR_BACKUP_DIR" "$RELEASE_DIR" || return 1
  create_state_next || return 1
  /bin/cp "$REPAIR_STATE_DIR/installation_state.json" "$STATE_NEXT" ||
    return 1
  /bin/mv -f "$STATE_NEXT" "$STATE_FILE" || return 1
  STATE_NEXT=
  /bin/rmdir "$STATE_STAGE_DIR" || return 1
  STATE_STAGE_DIR=
  /bin/rm -rf "$REPAIR_STATE_DIR" || return 1
  return 0
}

recover_interrupted_repair() {
  if test ! -e "$REPAIR_STATE_DIR" && test ! -e "$REPAIR_BACKUP_DIR"; then
    return 0
  fi
  # Until the repair journal proves ownership and recovery completes, the EXIT
  # trap must preserve every related path exactly as found.
  PRESERVE_UNCERTAIN=1
  repair_state_is_owned ||
    fail "an unverified automatic-repair record exists; no files were changed."

  if test ! -e "$REPAIR_BACKUP_DIR"; then
    if test -d "$RELEASE_DIR" &&
      state_claims_release &&
      /usr/bin/cmp -s \
        "$REPAIR_STATE_DIR/installation_state.json" "$STATE_FILE"; then
      # The repair record was written but the old release was never moved.
      /bin/rm -rf "$REPAIR_STATE_DIR"
      PRESERVE_UNCERTAIN=0
      return 0
    fi
    if test -d "$RELEASE_DIR" &&
      state_claims_release &&
      "$INSTALLATION_DIR/environment_doctor.sh" \
        --quick "$SUPPORT_DIR" "$PROJECT_DIR" >/dev/null 2>&1 &&
      run_full_check "$RELEASE_DIR"; then
      # A verified replacement was committed and only journal cleanup was
      # interrupted.
      /bin/rm -rf "$REPAIR_STATE_DIR"
      PRESERVE_UNCERTAIN=0
      return 0
    fi
    fail "automatic-repair recovery is incomplete; no uncertain files were removed."
  fi

  if test ! -e "$RELEASE_DIR" || installing_marker_is_owned; then
    restore_previous_release ||
      fail "the previous environment could not be restored automatically."
    PRESERVE_UNCERTAIN=0
    printf '%s\n' "The previous Project Leap 2D environment was restored after an interrupted repair."
    return 0
  fi

  if state_claims_release &&
    "$INSTALLATION_DIR/environment_doctor.sh" \
      --quick "$SUPPORT_DIR" "$PROJECT_DIR" >/dev/null 2>&1 &&
    run_full_check "$RELEASE_DIR"; then
    /bin/rm -rf "$REPAIR_BACKUP_DIR" "$REPAIR_STATE_DIR"
    PRESERVE_UNCERTAIN=0
    printf '%s\n' "A completed Project Leap 2D repair was recovered and verified."
    return 0
  fi

  fail "automatic-repair recovery found two uncertain environments; neither was removed."
}

begin_owned_repair() {
  test "$REPAIR_ATTEMPTED" -eq 0 ||
    fail "automatic repair is limited to one attempt per installer run."
  REPAIR_ATTEMPTED=1
  state_claims_release ||
    fail "the damaged environment is not owned by this installation contract."
  test -d "$RELEASE_DIR" ||
    fail "the owned release directory is missing."
  test ! -L "$RELEASE_DIR" ||
    fail "the owned release path is a symbolic link and was not changed."
  test ! -e "$REPAIR_BACKUP_DIR" && test ! -e "$REPAIR_STATE_DIR" ||
    fail "a previous automatic-repair record must be recovered first."
  require_analysis_idle

  REPAIR_PREPARING=1
  old_umask=$(umask)
  umask 077
  REPAIR_STATE_STAGE_DIR=$(
    /usr/bin/mktemp -d "$SUPPORT_DIR/Repair State Preparing.XXXXXX"
  ) || {
    umask "$old_umask"
    fail "a private automatic-repair record could not be created."
  }
  umask "$old_umask"
  printf '%s\n' "$ENVIRONMENT_CONTRACT_ID" \
    > "$REPAIR_STATE_STAGE_DIR/environment_contract_id"
  printf '%s\n' "$RELEASE_DIR" > "$REPAIR_STATE_STAGE_DIR/release_dir"
  printf '%s\n' "$REPAIR_BACKUP_DIR" > "$REPAIR_STATE_STAGE_DIR/backup_dir"
  /bin/cp "$STATE_FILE" "$REPAIR_STATE_STAGE_DIR/installation_state.json"
  /bin/mv "$REPAIR_STATE_STAGE_DIR" "$REPAIR_STATE_DIR"
  REPAIR_STATE_STAGE_DIR=
  /bin/rm -f "$RELEASE_DIR/INSTALLING"
  /bin/mv "$RELEASE_DIR" "$REPAIR_BACKUP_DIR"
  REPAIR_ACTIVE=1
  REPAIR_PREPARING=0
}

cleanup() {
  status=$?
  trap - EXIT HUP INT TERM
  # Cleanup must continue through every rollback and lock-release step even if
  # one filesystem operation fails. The original installer status is retained.
  set +e
  if test "$INSTALL_COMPLETE" -ne 1; then
    if test "$LOCK_ACQUIRED" -ne 1; then
      :
    elif test "$PRESERVE_UNCERTAIN" -eq 1; then
      :
    elif test "$REPAIR_ACTIVE" -eq 1; then
      if repair_state_is_owned && test -d "$REPAIR_BACKUP_DIR"; then
        if test -e "$RELEASE_DIR" && installing_marker_is_owned; then
          /bin/rm -rf "$RELEASE_DIR"
        fi
        if restore_previous_release; then
          printf '%s\n' "The previous Project Leap 2D environment was restored after the repair failed." >&2
        else
          printf '%s\n' "INSTALLATION STOPPED SAFELY: the previous environment could not be restored automatically." >&2
        fi
      else
        printf '%s\n' "INSTALLATION STOPPED SAFELY: repair ownership could not be verified; uncertain files were preserved." >&2
      fi
    elif test "$REPAIR_PREPARING" -eq 1; then
      if repair_state_is_owned && test -d "$REPAIR_BACKUP_DIR"; then
        if test -e "$RELEASE_DIR" && installing_marker_is_owned; then
          /bin/rm -rf "$RELEASE_DIR"
        fi
        restore_previous_release || true
      elif repair_state_is_owned && test ! -e "$REPAIR_BACKUP_DIR"; then
        /bin/rm -rf "$REPAIR_STATE_DIR"
      fi
    # A signal can arrive immediately after the small ready-state file is
    # published during a clean installation. That state is created only after
    # the new release has passed the full integrity check.
    elif "$INSTALLATION_DIR/environment_doctor.sh" \
      --quick "$SUPPORT_DIR" "$PROJECT_DIR" >/dev/null 2>&1 &&
      state_claims_release; then
      INSTALL_COMPLETE=1
    elif test -d "$RELEASE_DIR" && installing_marker_is_owned; then
      /bin/rm -rf "$RELEASE_DIR"
    elif test -d "$RELEASE_DIR" && test ! -L "$RELEASE_DIR"; then
      /bin/rmdir "$RELEASE_DIR" 2>/dev/null || true
    fi
    if test -n "$STATE_NEXT" &&
      test -f "$STATE_NEXT" &&
      test ! -L "$STATE_NEXT"; then
      /bin/rm -f "$STATE_NEXT"
    fi
    if test -n "$STATE_STAGE_DIR" &&
      test -d "$STATE_STAGE_DIR" &&
      test ! -L "$STATE_STAGE_DIR"; then
      /bin/rmdir "$STATE_STAGE_DIR" 2>/dev/null || true
    fi
    if test -n "$REPAIR_STATE_STAGE_DIR" &&
      test -d "$REPAIR_STATE_STAGE_DIR" &&
      test ! -L "$REPAIR_STATE_STAGE_DIR"; then
      /bin/rm -f \
        "$REPAIR_STATE_STAGE_DIR/environment_contract_id" \
        "$REPAIR_STATE_STAGE_DIR/release_dir" \
        "$REPAIR_STATE_STAGE_DIR/backup_dir" \
        "$REPAIR_STATE_STAGE_DIR/installation_state.json"
      /bin/rmdir "$REPAIR_STATE_STAGE_DIR" 2>/dev/null || true
    fi
    if test -n "$INSTALLING_MARKER_STAGE" &&
      test -f "$INSTALLING_MARKER_STAGE" &&
      test ! -L "$INSTALLING_MARKER_STAGE"; then
      /bin/rm -f "$INSTALLING_MARKER_STAGE"
    fi
  fi
  exit "$status"
}
trap cleanup EXIT HUP INT TERM

test ! -L "$SUPPORT_DIR" ||
  fail "the support directory must not be a symbolic link."
/bin/mkdir -p "$SUPPORT_DIR"
test -d "$SUPPORT_DIR" && test ! -L "$SUPPORT_DIR" ||
  fail "the support path is not a real directory."
if test -e "$STATE_FILE" || test -L "$STATE_FILE"; then
  test -f "$STATE_FILE" && test ! -L "$STATE_FILE" ||
    fail "the installation state path is not a regular file."
fi
if test -e "$RELEASES_DIR"; then
  test -d "$RELEASES_DIR" && test ! -L "$RELEASES_DIR" ||
    fail "the Releases path is not a real directory."
else
  /bin/mkdir "$RELEASES_DIR"
fi
test ! -L "$RELEASE_DIR" ||
  fail "the release path is a symbolic link and was not changed."
test ! -L "$REPAIR_BACKUP_DIR" && test ! -L "$REPAIR_STATE_DIR" ||
  fail "an automatic-repair path is a symbolic link and was not changed."
acquire_environment_lock

if test -e "$REPAIR_STATE_DIR" || test -e "$REPAIR_BACKUP_DIR"; then
  require_analysis_idle
fi
recover_interrupted_repair

if "$INSTALLATION_DIR/environment_doctor.sh" \
  --quick "$SUPPORT_DIR" "$PROJECT_DIR" >/dev/null 2>&1; then
  printf '%s\n' "Checking the installed Project Leap 2D environment offline..."
  full_check_status=0
  run_full_check "$RELEASE_DIR" || full_check_status=$?
  if test "$full_check_status" -eq 0; then
    # A power loss may leave INSTALLING behind after the atomic ready-state
    # publication. Remove it only when that state names this exact release.
    if state_claims_release; then
      /bin/rm -f "$RELEASE_DIR/INSTALLING" 2>/dev/null || true
    fi
    printf '%s\n' "Project Leap 2D is already installed and passed the full integrity check."
    INSTALL_COMPLETE=1
    exit 0
  fi
  test "$full_check_status" -ne 20 ||
    fail "the Project Leap 2D package resources are damaged; reinstalling dependencies cannot repair the package."
  test "$full_check_status" -ne 22 ||
    fail "the full check encountered a non-repairable system or filesystem error."
  state_claims_release ||
    fail "the damaged environment is not owned by this installation contract."
  printf '%s\n' "Installed environment damage was detected. Starting one automatic repair."
  begin_owned_repair
elif state_claims_release; then
  if test -e "$RELEASE_DIR"; then
    printf '%s\n' "Installed environment damage was detected. Starting one automatic repair."
    begin_owned_repair
  else
    REPAIR_ATTEMPTED=1
    printf '%s\n' "The owned environment is missing. Starting one clean automatic repair."
  fi
fi

if test -e "$RELEASE_DIR"; then
  if installing_marker_is_owned; then
    # No ready state owns this exact installer-created release. It is an
    # interrupted inactive build and is safe to rebuild under the held lock.
    /bin/rm -rf "$RELEASE_DIR"
  elif test -d "$RELEASE_DIR" &&
    test ! -L "$RELEASE_DIR" &&
    /bin/rmdir "$RELEASE_DIR" 2>/dev/null; then
    :
  else
    fail "an unverified release directory already exists: $RELEASE_DIR"
  fi
fi

require_analysis_idle
/bin/mkdir "$RELEASE_DIR"
write_installing_marker ||
  fail "the installation ownership marker could not be published."
/bin/mkdir \
  "$RELEASE_DIR/Downloads" \
  "$RELEASE_DIR/Tools" \
  "$RELEASE_DIR/Models"

UV_ARCHIVE="$RELEASE_DIR/Downloads/uv.tar.gz"
download_verified \
  "$PROJECT_LEAP_UV_URL" \
  "$PROJECT_LEAP_UV_SHA256" \
  "$UV_ARCHIVE" \
  "uv"
/usr/bin/tar -xzf "$UV_ARCHIVE" -C "$RELEASE_DIR/Downloads"
UV_SOURCE="$RELEASE_DIR/Downloads/$PROJECT_LEAP_UV_ARCHIVE_ROOT/uv"
test -x "$UV_SOURCE" || fail "the verified uv archive has an unexpected layout."
/bin/cp "$UV_SOURCE" "$RELEASE_DIR/Tools/uv"
/bin/chmod 755 "$RELEASE_DIR/Tools/uv"
UV_BIN="$RELEASE_DIR/Tools/uv"

export UV_CACHE_DIR="$RELEASE_DIR/Cache"
export UV_PYTHON_INSTALL_DIR="$RELEASE_DIR/Managed Python"
"$UV_BIN" --no-progress python install --no-bin "$PROJECT_LEAP_PYTHON_VERSION"
MANAGED_PYTHON=$(
  "$UV_BIN" python find \
    --python-preference only-managed \
    "$PROJECT_LEAP_PYTHON_VERSION"
)
test -x "$MANAGED_PYTHON" || fail "uv did not create the managed Python runtime."

"$UV_BIN" --no-progress venv \
  --python "$MANAGED_PYTHON" \
  "$RELEASE_DIR/Environment"
ENVIRONMENT_PYTHON="$RELEASE_DIR/Environment/bin/python3"
test -x "$ENVIRONMENT_PYTHON" ||
  fail "the isolated Project Leap 2D environment was not created."

"$UV_BIN" --no-progress pip install \
  --python "$ENVIRONMENT_PYTHON" \
  --no-deps --strict \
  --requirement "$LOCK_FILE"
"$UV_BIN" pip check --python "$ENVIRONMENT_PYTHON"

download_verified \
  "$PROJECT_LEAP_CELLPOSE_MODEL_URL" \
  "$PROJECT_LEAP_CELLPOSE_MODEL_SHA256" \
  "$RELEASE_DIR/Models/cpsam_v2" \
  "Cellpose cpsam_v2 model"

FIJI_ARCHIVE="$RELEASE_DIR/Downloads/Fiji.$PROJECT_LEAP_FIJI_ARCHIVE_KIND"
download_verified \
  "$PROJECT_LEAP_FIJI_URL" \
  "$PROJECT_LEAP_FIJI_SHA256" \
  "$FIJI_ARCHIVE" \
  "Fiji"
# The audited archive owns its top-level Fiji/ directory. Extracting into a
# second Fiji/ directory would create Fiji/Fiji and invalidate every path.
case "$PROJECT_LEAP_FIJI_ARCHIVE_KIND" in
  zip)
    /usr/bin/ditto -x -k "$FIJI_ARCHIVE" "$RELEASE_DIR"
    ;;
  tar.gz)
    /usr/bin/tar -xzf "$FIJI_ARCHIVE" -C "$RELEASE_DIR"
    ;;
  *)
    fail "unsupported Fiji archive kind: $PROJECT_LEAP_FIJI_ARCHIVE_KIND"
    ;;
esac
FIJI_LAUNCHER="$RELEASE_DIR/$PROJECT_LEAP_FIJI_LAUNCHER_RELATIVE"
test -x "$FIJI_LAUNCHER" ||
  fail "the verified Fiji archive does not contain the expected launcher."

run_full_check "$RELEASE_DIR" ||
  fail "the replacement environment did not pass the full integrity check."

json_escape() {
  printf '%s' "$1" |
    /usr/bin/sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

CREATED_AT=$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')
create_state_next ||
  fail "a private state-publication directory could not be created."
{
  printf '{\n'
  printf '  "schema_version": "%s",\n' "$PROJECT_LEAP_INSTALL_SCHEMA"
  printf '  "status": "ready",\n'
  printf '  "platform": "%s",\n' "$PROJECT_LEAP_PLATFORM"
  printf '  "environment_contract_id": "%s",\n' "$ENVIRONMENT_CONTRACT_ID"
  printf '  "project_version": "%s",\n' "$(json_escape "$PROJECT_VERSION")"
  printf '  "python_executable": "%s",\n' "$(json_escape "$ENVIRONMENT_PYTHON")"
  printf '  "fiji_launcher": "%s",\n' "$(json_escape "$FIJI_LAUNCHER")"
  printf '  "cellpose_models_path": "%s",\n' "$(json_escape "$RELEASE_DIR/Models")"
  printf '  "environment_lock_sha256": "%s",\n' "$LOCK_SHA"
  printf '  "cellpose_model_sha256": "%s",\n' "$PROJECT_LEAP_CELLPOSE_MODEL_SHA256"
  printf '  "created_at": "%s"\n' "$CREATED_AT"
  printf '}\n'
} > "$STATE_NEXT"

/bin/rm -rf "$RELEASE_DIR/Downloads" "$RELEASE_DIR/Cache"
/bin/mv -f "$STATE_NEXT" "$STATE_FILE"
STATE_NEXT=
/bin/rmdir "$STATE_STAGE_DIR"
STATE_STAGE_DIR=
/bin/rm -f "$RELEASE_DIR/INSTALLING" 2>/dev/null || true
INSTALL_COMPLETE=1
if test "$REPAIR_ACTIVE" -eq 1; then
  /bin/rm -rf "$REPAIR_BACKUP_DIR" "$REPAIR_STATE_DIR"
  REPAIR_ACTIVE=0
fi

printf '%s\n' "Project Leap 2D installation completed."
printf '%s\n' "Support directory: $SUPPORT_DIR"
