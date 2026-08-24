# Project Leap 2D

## First use

This release supports Apple Silicon Macs running macOS 11 or later. A new user
does not need to preinstall Homebrew, pip, Python, Cellpose, or Fiji. Open the
`Project Leap 2D (8-23-26)` package directory in Terminal, then run:

```bash
./Installation/macOS/install_macos.command
```

The installer places checksum-pinned Python 3.9, scientific dependencies,
Cellpose model, and Fiji components in the visible
`~/Applications/Project Leap 2D Support` directory. Administrator access is
not required. A clean installation of this release used about 2.1 GB of
network downloads and 2.8 GB of installed support data; filesystem reporting
may vary slightly. Later Project Leap 2D releases reuse this environment
when the dependency contract has not changed.

Normal startup reads one small installation-state file and checks that Python,
Fiji, and the Cellpose model still exist at their recorded paths. It does not
contact the network, import a model, or hash a large file. Running the installer
without an option explicitly performs an offline full integrity check of locked
dependency versions, the managed Python interpreter, runtime libraries and
standard library, about 19,000 registered files from 39 Python wheels,
Cellpose and bundled InstanSeg model hashes, the fixed Fiji/Java tree, and
critical Python imports. The integrity baselines themselves have fixed
SHA-256 values. Runtime-generated `__pycache__` directories in the managed
standard library are removed before comparison, and both checks use `-B` to
prevent regeneration; precompiled `.pyc` files registered by a wheel remain
and are verified file by file. Fiji's runtime-updated `.checksums` and
`db.xml.gz` are excluded from fixed-tree comparison. macOS code signatures
for the Python executable and runtime library are verified separately. For 19
uv-generated command entry points, only the virtual-environment absolute
prefix that varies by Mac username is normalized; every other byte remains
covered. The installation entry chain also authenticates the installer,
checker, contracts, and all three integrity baselines before any network
access or installation change. A healthy environment exits
without downloading or reinstalling anything. If an environment demonstrably
owned by this installation is damaged, the installer makes one automatic
repair attempt. It holds the old environment as a rollback copy, installs and
verifies a clean replacement, and commits the replacement only after every
check passes. A failed repair restores the old environment and stops. Unowned
directories are never removed.

For only a quick installation-state and path check, run:

```bash
./Installation/macOS/install_macos.command --check
```

An `INSTALLING` marker records the current environment contract and exact
release path, and means the previous installation stopped before full
verification. When that identity matches, the installer does not resume the
unverifiable partial environment; it removes the incomplete release and starts
cleanly. An unidentified marker or directory is preserved and causes a safe
stop. If an automatic repair itself was interrupted, the next run first
restores the old environment or verifies that the replacement was fully
committed. Analysis and installation share one macOS kernel-held environment
lock. The system releases it automatically after normal exit, failure, or
power loss, so no stale PID record needs to be deleted. Installation or repair
stops safely instead of moving Python, models, or Fiji while an analysis is
using them.

## Run

1. Put one batch of split single-channel Z-stack TIFF files in
   `Original Image`.
2. Open the `Project Leap 2D (8-23-26)` package directory in Terminal, then
   run:

```bash
./run_project_leap_2d.command
```

The same relative commands are stored in `RUN_COMMAND.txt`. Production outputs
are written directly to the root of `Result`.

## Input routes

- Each batch must contain exactly one DAPI stack, at least one eGFP/GFAP stack,
  and exactly one KCNN1/KCNN2/KCNN3/KCNJ10 measurement stack. All inputs must
  be split single-channel ZYX TIFF files with matching geometry and physical
  calibration.
- Do not place the original unsplit multichannel acquisition in
  `Original Image`. A filename containing several channel identities such as
  DAPI, GFAP, and KCNN2 is rejected safely as ambiguous.
- Any recognized eGFP stack selects the validated eGFP route.
  GFAP-only is not enabled when eGFP is present, even if GFAP is also present.
- If the DAPI/eGFP filenames contain no age marker, the eGFP route retains its
  morphology-based automatic mature/neonatal profile selection. An explicit,
  non-conflicting age marker takes priority.
- DAPI + GFAP without a recognized eGFP stack uses the independent GFAP-only
  route. A checksum-pinned InstanSeg CPU TorchScript model is bundled, so no
  network access or full InstanSeg installation is required. It generates DAPI
  nucleus candidates only; this program still performs 3D linking, GFAP
  association, exclusive ownership validation, and Whole/Soma/Processes
  construction.
- The measurement channel is used only for final untouched-grayscale
  measurement and never for ROI definition.
- This release supports GFAP-only analysis only for mature astrocytes. A
  missing age marker or an explicit `mature` marker in the DAPI/GFAP filenames
  uses the mature configuration. A recognized `neonatal` marker stops before
  analysis; conflicting age markers also stop as ambiguous.
- The author tested the program with mature GFAP-only samples held outside the
  package. Those samples and their results are not distributed.

## Fiji review and Cell Edit

- Whole Cell ROI Manager: Delete and Split.
- Soma ROI Manager: Delete, Merge, and Enlarge.
- Processes ROI Manager: Delete.
- Revert is a LIFO undo stack. Repeated clicks step backward through committed
  Delete, Merge, Split, and Enlarge edits.

Each Split click searches the selected Whole Cell and its immediate local
neighborhood for exactly one additional DAPI nucleus, then recalculates the
cell as two children. The original Whole is the trusted primary region.
Processes outside it are recovered only when they remain connected to the
corresponding Soma, have clear structural support, and do not enter another
nucleus's competing territory. An unaccepted or implausibly small second
nucleus requires local DAPI-model confirmation, and a marginal nucleus outside
the selected ROI cannot trigger Split directly. Enlarge retains the
established local-evidence rule for eGFP samples. In GFAP-only samples it is
driven primarily by the
complete DAPI nucleus and a physically calibrated perinuclear range; GFAP is
secondary evidence for the outer boundary and neighboring-cell exclusion. A
validated Soma may extend beyond the old Whole; the Whole expands with it and
Processes is recomputed as `Whole - Soma`. Every action synchronizes all three
compartments and restores continuous IDs. Local calculations run in a bounded
worker process with timeout and cancellation. Insufficient evidence produces
a short English refusal instead of a forced result.

## Workspace rules

- `Original Image` and `Result` are permanent folders.
- `Runtime` is a visible runtime-state folder for locks, recovery data, and the
  Matplotlib cache. The launcher does not create a hidden `.runtime` folder.
- Files left in the root of `Result` are archived together into `Pending`,
  `Pending 1`, `Pending 2`, and so on before a new run.
- No cross-run image cache is created.
- Accepted TIFFs move to macOS Trash only after Fiji completes, all validation
  passes, and the production bundle is published.
- Five-file publication and the final move to Trash use small overwrite-only
  recovery journals. After forced termination or power loss, the next launch
  first restores a consistent state; completed journals are then deleted and
  do not accumulate per run.
- Inputs remain after a safety stop, Fiji Cancel, exception, `Ctrl-C`,
  `--skip-fiji`, or publication failure. Unrecognized extra files are never
  moved.

## Production outputs

- `IHC_2D_Whole_Astrocyte_Overlay.png`
- `IHC_2D_Astrocyte_Soma_Overlay.png`
- `IHC_2D_Astrocyte_Processes_Overlay.png`
- `IHC_2D_Analysis_Report.txt`
- `IHC_2D_Fluorescence_Results.xlsx`

The program validates synchronized IDs, exact compartment partitioning, Fiji
raw-grayscale measurements, overlay dimensions, and workbook structure before
publishing the five-file bundle with rollback protection.
The analysis report retains overall status and inference timing for each major
stage without creating a separate candidate-by-candidate debug report.

See [MODULE_MAP_ENGLISH.md](MODULE_MAP_ENGLISH.md) for module responsibilities.
The emergency `fallback/single_file_fallback.py` is limited to eGFP recovery
and internal parity checks; GFAP-only analysis must use the standard launcher.
The fallback is not the normal entry point.
