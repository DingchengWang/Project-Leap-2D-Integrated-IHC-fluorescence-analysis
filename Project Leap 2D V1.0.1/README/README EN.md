# Project Leap 2D

## Daily use

The default working package folder is `~/Desktop/Project Leap 2D V1.0.1`.

```text
Project Leap 2D V1.0.1/
├── Sample Image/
├── Result/
├── Analysis Package/
├── README/
├── Run Analysis.command
├── Repair.command
└── Manual Command.txt
```

`Analysis Package` holds program files. Its `Run State` folder is managed
automatically and contains locks and recovery records; do not clear it manually.
The `README` folder contains `README CN.md` and `README EN.md`.

1. Put one batch of split single-channel Z-stack TIFF files in the working
   package's `Sample Image` folder.
2. Double-click `Run Analysis.command` inside the working package. macOS opens
   it in Terminal to show progress; follow the prompts to complete Fiji review.

Final output files are saved directly in the root of `Result`. You do not need
to type commands for daily use. If you prefer Terminal, `Manual Command.txt`
contains equivalent commands for both launchers.

At startup, the program reads the small installation status file and the
configuration file that records the environment requirements. It then
checks that Python, Fiji, and the Cellpose model still exist at their recorded
paths. This quick check does not access the network, import a model, or hash
large files; models are loaded when the analysis needs them. Passing this
check does not verify the integrity of every program or dependency file.

## Check and repair

If program files are missing or damaged, or startup checks fail, double-click
`Repair.command` inside the working package. It takes no arguments and does
not ask you to select a ZIP file. It performs these steps in order:

1. Using macOS system tools, it retrieves the release checksum manifest for
   this version online and checks the program files. If any are missing or
   damaged, it automatically downloads and verifies the release package for
   the same version, then restores the affected files.
2. Once the program check and any required restoration succeed, it thoroughly
   checks the Python installation managed by the program, dependencies at
   their locked versions, models, and Fiji/Java. It also checks that critical
   dependencies can be imported. The dependency environment is rebuilt only
   if needed; a healthy environment is not reinstalled.

Restoring program files does not overwrite or clear the contents of
`Sample Image`, `Result`, or `Analysis Package/Run State`, and does not upgrade
the version. Repair can restore missing or damaged files from the same
version; it cannot fix bugs already present in that version. If
`Repair.command` itself is missing or cannot start, obtain that version's
repair launcher again.

Online repair requires this version's GitHub Release and its assets to be
published and accessible. If the release information cannot be retrieved or
verified, repair stops while checking the program files, without
checking or rebuilding the environment. Daily analysis does not access the
network; every `Repair.command` run needs network access to check release
information.

The full environment check removes `__pycache__` directories generated during
runtime from the managed Python standard library, so the check modifies these
cache directories. If a damaged environment is confirmed to belong to this
program, repair makes at most one attempt to rebuild it. It retains the old
environment as a rollback copy, then installs and verifies a replacement. The
replacement is put into use only after all checks pass. If repair fails, it
attempts to restore the old environment and then stops. Directories that
cannot be confirmed to belong to this program are left untouched.

Analysis and repair share an environment lock held by the macOS kernel.
Repair stops while analysis is using the environment and does not move Python, models, or
Fiji in use. Repair reports completion only after the program and environment
checks pass. Then double-click `Run Analysis.command` to start an analysis.

## Inputs and analysis routes

- Each batch must contain exactly one DAPI stack, at least one eGFP/GFAP stack,
  and exactly one KCNN1/KCNN2/KCNN3/KCNJ10 measurement stack. All inputs must
  be split single-channel ZYX TIFF files with matching geometry and physical
  calibration.
- Do not place the original unsplit multichannel acquisition in
  `Sample Image`. A filename containing several channel identities such as
  DAPI, GFAP, and KCNN2 is rejected because the program cannot identify a
  unique channel.
- Any recognized eGFP stack selects the eGFP route.
  GFAP-only is not enabled when eGFP is present, even if GFAP is also present.
- For both eGFP-only inputs and inputs containing eGFP and GFAP, the program
  first excludes 5 µm from each end of the Z-stack, converts the 9.5 µm target
  width to the nearest plane count from the actual Z-step, and places five
  equally spaced windows in the usable depth. Each window runs six Morphology
  Baseline, six Structural Refinement, and six Distributional Threshold
  candidates. The program first selects the best Whole within each window,
  then compares the best candidates from all five windows. To compare positions,
  the program identifies axial runs from the per-plane normalized P85 (85th
  percentile) profile of each DAPI nucleus region. An analyzable cell requires
  one axial nucleus and one Whole component to correspond uniquely in both
  directions, and axial retention within the window is scored at the same time.
  The measurement channel has zero weight and cannot select Z or Whole.
  Insufficient depth prints `z_step_limit` and stops; an exact tie for the
  highest final score prints `z_selection_ambiguous` and stops.
- If the DAPI/eGFP filenames contain no age marker, the eGFP route selects the
  mature or neonatal profile from morphology. An eGFP-only input uses the
  eGFP-only age calibration; an input containing both eGFP and GFAP retains the
  established age parameters. The program applies the age configuration only
  after the final Z window and initial Whole have been selected. The eGFP-only
  parameters were calibrated specifically against two known-Mature and three
  P3 Neonatal reference samples; other ages, regions, or acquisition conditions
  require validation with known-age samples. An explicit, non-conflicting
  filename age marker takes priority.
- DAPI + GFAP without a recognized eGFP stack uses the independent GFAP-only
  route. An InstanSeg CPU TorchScript model is included and verified against
  a fixed checksum, so no network access or full InstanSeg installation is required. It generates DAPI
  nucleus candidates only; this program still performs 3D linking, GFAP
  association, exclusive ownership validation, and Whole/Soma/Processes
  construction.
- The program checks only the technical suitability of the measurement
  channel, without changing its intensity values. This channel has zero
  weight in the five-position score and does not define Z, Whole, or any ROI.
  Its original grayscale values are measured only at the final best position.
- This release supports GFAP-only analysis only for mature astrocytes. A
  missing age marker or an explicit `mature` marker in the DAPI/GFAP filenames
  uses the mature configuration. A recognized `neonatal` marker stops before
  analysis; conflicting age markers also stop as ambiguous.

## Fiji review and Cell Edit

- Whole Cell ROI Manager: Delete and Split.
- Soma ROI Manager: Delete, Merge, and Enlarge.
- Processes ROI Manager: Delete and Trim Processes.
- Revert undoes the most recently committed edit first. Repeated clicks step
  backward through committed Delete, Merge, Split, Enlarge, and Trim Processes
  edits.

Trim Processes opens a modeless panel that lets you continue working in the
image windows. It contains Preview Trim, Clear Cell Draft, Undo, Confirm Trim,
and Cancel. No cell needs to be selected before opening it. Draw freehand
area selections in any of the three composite windows;
before Preview Trim, the draft shows only affected cell IDs and stroke
outlines. The first preview checks all affected cells in parallel and shows
predicted Processes removal in yellow with a black outline for contrast.
The formal Whole/Soma/Processes outlines remain unchanged until confirmation.
After the first preview, draft changes automatically recheck the affected
cells.

Clear Cell Draft removes only the selected cell ID's drafts, including that
cell's part of a stroke spanning multiple cells. Undo reverses strokes and
Clear Cell Draft actions within this panel. Confirm Trim cannot proceed if
the preview is out of date, the draft removes no Processes pixels, the
resulting Processes would be empty, or the trim would disconnect retained
branches from their Soma. A successful confirmation
removes the same pixels from Whole and Processes, keeps Soma unchanged, and
adds one edit to the global Revert stack. Cancel discards the draft. Trim
does not add an exclusion constraint to later Split or Enlarge and writes no
runtime history files.

Each Split click searches the selected Whole Cell and its immediate local
neighborhood for exactly one additional DAPI nucleus, then recalculates the
cell as two cells. The original Whole is the trusted primary region.
Processes outside it are recovered only when they remain connected to the
corresponding Soma, have clear structural support, and do not enter another
nucleus's competing territory. An unaccepted or implausibly small second
nucleus requires local DAPI-model confirmation, and a marginal nucleus outside
the selected ROI cannot trigger Split directly. Enlarge retains the
established local-evidence rule for eGFP samples. In GFAP-only samples it is
driven primarily by the complete DAPI nucleus and a physically calibrated
perinuclear range; GFAP is secondary evidence for the outer boundary and
neighboring-cell exclusion. A
validated Soma may extend beyond the old Whole; the Whole expands with it and
Processes is recomputed as `Whole - Soma`. Every action updates all three
compartments together and restores continuous IDs. Local calculations run in
a separate, bounded worker process with timeout and cancellation. If the
evidence is insufficient, the program declines the edit with a short English
message rather than forcing a result.

## Workspace rules

- `Sample Image` and `Result` are permanent folders.
- `Analysis Package/Run State` is a visible runtime-state folder for locks, recovery data, and the
  Matplotlib cache. The launcher does not create a hidden `.runtime` folder.
- Files left in the root of `Result` are archived together into `Pending`,
  `Pending 1`, `Pending 2`, and so on before a new run.
- No cross-run image cache is created.
- TIFFs used in the analysis move to macOS Trash only after Fiji completes,
  all validation passes, and the final output files have been saved successfully.
- Saving the five final output files and moving the source TIFFs to Trash
  both maintain small recovery records that are updated in place.
  After forced termination or power loss, the next launch first restores a
  consistent state. Completed records are then deleted and do not accumulate
  across runs.
- Inputs remain after a safety stop, Fiji Cancel, exception, `Ctrl-C`,
  `--skip-fiji`, or a failure while saving the final outputs. Unrecognized
  extra files are never moved.

## Output files

- `IHC_2D_Whole_Astrocyte_Overlay.png`
- `IHC_2D_Astrocyte_Soma_Overlay.png`
- `IHC_2D_Astrocyte_Processes_Overlay.png`
- `IHC_2D_Analysis_Report.txt`
- `IHC_2D_Fluorescence_Results.xlsx`

The program validates synchronized IDs, exact compartment partitioning, Fiji
raw-grayscale measurements, overlay dimensions, and workbook structure before
saving the five output files together with rollback protection.
The analysis report records the overall status and inference time for each
major stage. It lists the actual range and width of all five Z positions, the
best candidate at each position, DAPI axial-nucleus correspondence and
retention, the five component scores, the total score, and the final best position.
The candidate list records each candidate's position and whether it is the best
candidate at that position. No separate debug report is created for each candidate.
