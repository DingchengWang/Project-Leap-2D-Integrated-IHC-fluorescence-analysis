# Project Leap 2D Module Map

This file describes the package layout, module responsibilities, and runtime
order. Normal use requires only `run_project_leap_2d.command`; individual
Python modules should not be launched directly.

## 1. Package layout

```text
Project Leap 2D (8-23-26)/
├── run_project_leap_2d.command
├── RUN_COMMAND.txt
├── README_中文.md
├── README_English.md
├── MODULE_MAP_中文.md
├── MODULE_MAP_ENGLISH.md
├── Installation/
│   └── macOS/
├── Original Image/
├── Result/
├── Runtime/
├── project_leap_2d/
├── fallback/
├── tests/
└── validation/
```

- `Original Image/`: holds one batch of split single-channel Z-stack TIFFs.
- `Result/`: holds the current five-file production bundle. Existing outputs
  move together into sequentially named `Pending` folders before a new run.
- `Runtime/`: visible run locks, Cell Edit transactions, and temporary state;
  it is not part of the production result.
- `fallback/single_file_fallback.py`: emergency single-file entry point limited
  to eGFP recovery and internal parity checks. GFAP-only must use the standard
  launcher; the fallback is not the normal runtime path.

## 2. Entry point, routing, and shared runtime state

```text
project_leap_2d/
├── __main__.py
├── workspace_launcher.py
├── runtime_loader.py
├── runtime_manifest.py
├── runtime_attributes.py
├── startup.py
├── analysis_workflow.py
├── analysis_controller.py
├── command_line.py
├── settings.py
├── data_structures.py
└── run_state.py
```

- `workspace_launcher.py`: uses the package-local `Original Image`, `Result`,
  and `Runtime` folders and controls run locks, Pending archiving, safe
  publication, and post-success Trash cleanup.
- `startup.py`: fixes parallel thread settings before NumPy, SciPy, OpenCV, or
  model code is imported.
- `runtime_loader.py`: loads modules into one shared state in declared order,
  avoiding duplicate models, caches, and global runtime state.
- `runtime_manifest.py`: declares load order, required symbols, and Fiji
  resources.
- `runtime_attributes.py`: temporarily replaces shared-runtime attributes for
  a local task, then restores the exact prior objects in reverse order after
  either success or failure. This prevents one operation from changing later
  analysis.
- `analysis_workflow.py`: routes inputs to the validated eGFP path or independent
  GFAP-only path, then coordinates measurement, Fiji review, and publication.
- `analysis_controller.py`: preserves the validated eGFP scientific order.
- The remaining modules provide settings, data structures, command-line
  control, parallel state, and runtime diagnostics.

## 3. Image handling and eGFP candidate analysis

```text
project_leap_2d/images/
├── channel_files.py
├── image_loading.py
└── image_processing.py

project_leap_2d/segmentation/
├── cellpose_segmentation.py
├── candidate_features.py
├── candidate_evaluation.py
├── candidate_selection.py
├── standard_morphology_candidates.py
├── structural_refinement_candidates.py
├── distributional_threshold_candidates.py
└── candidate_catalog.py
```

- `images/`: channel discovery, physical calibration, ZYX loading,
  projection, and display rendering.
- `cellpose_segmentation.py`: runs and reuses Cellpose-SAM over five Z
  intervals.
- `candidate_features.py`: builds reusable DAPI, Sato, top-hat, and
  distribution features.
- Three candidate modules form 30 Morphology Baseline, 30 Structural
  Refinement, and 30 Distributional Threshold candidates in declared order.
- Candidate producers assign these functional module names directly. The
  report reads the same names from candidate metadata; no display translation
  layer is involved.
- `candidate_evaluation.py` and `candidate_selection.py`: parallel candidate
  computation, ranking, near-duplicate removal, and baseline-preserving
  challenger safeguards.

Any recognized valid eGFP stack keeps this main path, even when GFAP is also
present. Measurement-channel pixels never define an ROI.

## 4. DAPI, Soma/Processes, and manual recalculation

```text
project_leap_2d/nuclei/
├── dapi_nuclei.py
├── dapi_3d_inventory.py
├── nucleus_ownership.py
└── instanseg_nucleus_detection.py

project_leap_2d/compartments/
├── cell_separation.py
├── compartment_validation.py
├── soma_completion.py
├── soma_and_processes.py
├── selected_cell_split.py
└── selected_soma_enlargement.py
```

- `dapi_3d_inventory.py`: builds the 3D DAPI nucleus inventory and stops safely
  when abnormal fragmentation would create excessive work.
- `command_line.py` exposes
  `--dapi-fragment-workload-preflight-only` for a validation-only workload
  count and `--dapi-fragment-workload-json` for its required JSON destination.
  A workload safety stop uses
  `IHC_2D_DAPI_Fragment_Workload_Failure.json` when no explicit destination is
  supplied for the normal analysis route.
- `nucleus_ownership.py`: coordinates unique owner nuclei on the eGFP path.
- `instanseg_nucleus_detection.py`: lazily loads the bundled pinned CPU
  TorchScript model and generates DAPI nucleus candidates only; it does not
  decide astrocyte identity.
- `soma_completion.py` and `soma_and_processes.py`: complete nucleus extent,
  connect same-Soma islands, and enforce the final
  `Whole = Soma union Processes` partition.
- `selected_cell_split.py`: confirms one additional DAPI nucleus per click and
  rebuilds two cells around the two nuclei. The original Whole is the trusted
  primary region. External Processes are recovered only when connected to the
  corresponding Soma, structurally supported, and outside every competing
  nucleus territory. A weak nucleus outside the original Whole cannot trigger
  Split directly; an unaccepted or implausibly small nucleus first requires
  local DAPI-model confirmation. Both children must retain real Processes or
  the entire edit is rejected without commit.
- `selected_soma_enlargement.py`: recalculates the selected Soma. eGFP samples
  retain the established local-evidence rule. GFAP-only samples are driven
  primarily by the complete DAPI nucleus and a physically calibrated
  perinuclear range; GFAP is secondary evidence for the outer boundary and
  neighboring-cell exclusion.

Split, Enlarge, Merge, Delete, and Revert must update Whole, Soma, and
Processes together while preserving continuous IDs and an exact partition
without overlap or gaps.

## 5. Independent GFAP-only analysis

```text
project_leap_2d/analysis_modes/
├── structural_fluorescence.py
└── gfap_only/
    ├── gfap_only_pipeline.py
    ├── gfap_only_analysis.py
    ├── gfap_structure.py
    ├── gfap_nucleus_ownership.py
    ├── gfap_compartments.py
    └── gfap_post_compartment_quality.py
```

- `structural_fluorescence.py`: eGFP structural-fluorescence interface.
- `gfap_only_pipeline.py`: GFAP-only Z selection, DAPI detection, structural
  analysis, compartment construction, measurement/Fiji handoff, and memory
  release.
- `gfap_only_analysis.py`: composes the GFAP-only stages, performs final
  consistency checks, preserves an accepted owner across 2D nucleus
  collisions, and retains each neighboring nucleus in its exclusive pixels
  for Cell Edit exclusion.
- `gfap_structure.py`: GFAP background correction and multiscale fiber
  evidence.
- `gfap_nucleus_ownership.py`: links DAPI nuclei in 3D, keeps all qualified
  neighboring nuclei in competition, and uses Z position, continuous
  structural paths, and local GFAP association to resolve cell identity.
- `gfap_compartments.py`: makes DAPI nuclei the primary Soma basis. Internal
  ownership keeps a `2.10 um` search range while the final visible mature Soma
  uses an independent `1.25 um` limit. Processes are assigned exclusively
  along continuous structural paths, and ambiguous structures remain
  unassigned.
- `gfap_post_compartment_quality.py`: rejects clearly unreliable GFAP-only
  nucleus matches or morphologies before publication.

GFAP-only is enabled only for DAPI + GFAP without a recognized eGFP stack. The
bundled InstanSeg model requires neither network access nor a full InstanSeg
installation. This release supports GFAP-only analysis only for mature
astrocytes: no age marker or an explicit `mature` marker uses the mature
configuration, while a recognized `neonatal` marker stops before analysis.
Conflicting age markers also stop as ambiguous. The author tested the program
with mature GFAP-only samples held outside the package; those samples and their
results are not distributed.

## 6. Fiji review and Cell Edit

```text
project_leap_2d/fiji_review/
├── fiji_launcher.py
├── review_protocol.py
├── cell_editing.py
├── cell_edit_context.py
├── cell_edit_worker.py
├── cell_edit_transactions.py
├── cell_edit_fiji_bridge.py
├── failed_run_retention.py
├── review_validation.py
├── measurement_result_validation.py
└── resources/
    └── astrocyte_roi_reviewer.groovy
```

- `review_protocol.py`: prepares compartment labels, displays, and the Fiji
  manifest.
- `cell_edit_context.py`: stores DAPI, structural evidence, nucleus identity,
  analysis mode, physical calibration, and synchronized labels for local
  recalculation.
- `cell_edit_worker.py`: runs Split/Enlarge in an isolated bounded process with
  cancellation, timeout, input hashes, and result validation.
- `cell_edit_transactions.py`: for production Split and Enlarge requests,
  validates the proposed three-compartment state, maintains stable Cell UIDs,
  and commits the proposal only when it still matches the current Fiji state.
- `cell_editing.py` and `cell_edit_fiji_bridge.py`: connect Fiji Split and
  Enlarge requests to Python background tasks.
- `failed_run_retention.py`: after a Fiji failure, keeps only the latest
  verified failure workspace. It removes older failure workspaces only from
  the controlled cache location and only when their names and contents match
  the safety rules.
- `review_validation.py`: verifies identity, pixel-change boundaries, and the
  exact compartment partition after every edit.
- `measurement_result_validation.py`: validates final Fiji raw-grayscale
  measurements, continuous IDs, area partitioning, integrated density, and
  overlays.
- `astrocyte_roi_reviewer.groovy`: performs Delete, Merge, and Revert directly
  in Fiji; it sends Split and Enlarge requests to Python and provides Cancel
  and the three ROI Managers.

Merge, Split, and Enlarge remain available after GFAP-only reaches Fiji. They
use GFAP-only DAPI/structural evidence and do not switch to eGFP rules.

## 7. Measurement, reporting, and workspace publication

```text
project_leap_2d/reporting/
├── analysis_report.py
└── excel_results.py

project_leap_2d/workspace/
├── folder_checks.py
├── workspace_preflight.py
├── pending_results.py
├── result_publishing.py
├── publication_recovery.py
├── input_cleanup.py
└── input_cleanup_recovery.py
```

- `analysis_controller.py` reads the measurement channel as untouched
  grayscale over the selected inclusive Z interval. The resulting projection
  is passed to Fiji for final ROI measurements and never participates in ROI
  definition.
- `analysis_report.py`: records inputs, channels, physical calibration,
  inference timing, overall status for each major stage, final compartments,
  and Fiji state.
- Candidate and workload producers emit their functional names directly;
  reporting writes those names without a translation module.
- `excel_results.py`: creates final Whole, Processes, and Soma measurement
  sheets.
- `result_publishing.py`: performs the validated five-production-file
  publication steps.
- `publication_recovery.py`: writes one small overwrite-only transaction
  record and prepares recovery copies of existing formal outputs. The five
  files are not replaced at the same instant; after a power loss or process
  interruption, the next launch restores the complete prior result instead of
  leaving a mixture of old and new files.
- `input_cleanup.py`: moves accepted source TIFFs to macOS Trash only after the
  formal result has passed validation and publication.
- `input_cleanup_recovery.py`: records input identity and location before the
  Trash move. After an interrupted move, the next launch either restores the
  files to `Original Image` or confirms the complete Trash destination rather
  than silently losing inputs.
- The remaining `workspace/` modules perform preflight checks and Pending
  archiving. Successful completion removes transaction records and recovery
  copies, so they do not accumulate as user reports.

## 8. First-time macOS installation

```text
Installation/macOS/
├── install_macos.command
├── bootstrap_macos.sh
├── environment_installer.sh
├── environment_doctor.sh
├── environment_doctor.py
├── installer_integrity_manifest.sh
├── component_manifest.sh
├── requirements_macos_arm64.lock.txt
├── python_wheel_integrity.json
├── managed_python_integrity.json
├── fiji_tree_integrity.json
└── environment_contract.txt
```

- `install_macos.command`: the first-use entry point that a new Mac user can
  double-click or run in Terminal. Running it without an option performs an
  offline full integrity check; `--check` performs only the normal quick state
  and path check.
- `bootstrap_macos.sh` and `environment_installer.sh`: install a pinned Python
  environment, dependencies, Cellpose model, and Fiji in a visible
  user-support directory. They do not require Homebrew or pip to be installed
  beforehand. When an environment proven to belong to the current contract is
  damaged, they make one transactional automatic repair attempt: the old
  environment remains available for rollback, the clean replacement is
  committed only after a full check, a failed repair restores the old
  environment, and unowned directories remain untouched.
- `component_manifest.sh`, the dependency lock, and the environment contract
  pin download sources, versions, and SHA-256 values so different Macs receive
  the same environment.
- `environment_doctor.sh/.py`: verifies the managed Python interpreter,
  runtime libraries and standard library, dependency versions, every
  RECORD-hashed file in 39 Python wheels, models, the fixed Fiji/Java tree,
  and bundled scientific resources. `python_wheel_integrity.json` protects
  each wheel RECORD, `managed_python_integrity.json` protects the Python
  runtime obtained by the pinned uv version, and `fiji_tree_integrity.json`
  was derived from the official checksum-pinned Fiji ZIP.
  `installer_integrity_manifest.sh` lets the bootstrap authenticate the
  installer, checker, contracts, and all three baselines before network access
  or installation changes. Runtime standard-library caches are removed first
  and `-B` prevents regeneration; wheel-registered precompiled `.pyc` files
  remain covered by their RECORDs. Only the username-dependent virtual-
  environment prefix in 19 uv-generated command entry points is normalized.
  Normal analysis startup reads only a small
  installation-state file; it does not reinstall, contact the network, or
  import the full scientific environment.
- `INSTALLING` records the environment contract and exact release path, so it
  identifies an unverified environment demonstrably created by this installer.
  The next run discards that incomplete release and starts cleanly instead of
  resuming unknown state; an unidentified directory is preserved. An
  interrupted automatic repair first restores the old environment or fully
  verifies that the replacement was committed. The launcher and installer
  share one macOS kernel-held environment lock. The system releases it on
  process exit or failure without a stale PID record, preventing repair from
  moving Python, models, or Fiji while an analysis is using them.

## 9. Overall runtime order

```text
Workspace and channel checks
→ eGFP-priority route or independent GFAP-only route
→ Whole/Soma/Processes construction and validation
→ untouched-grayscale measurement projection
→ Fiji manual review and optional Cell Edit
→ three-compartment measurement and result validation
→ recoverable, rollback-protected transaction for all five production files
→ used source TIFFs move to macOS Trash
```

Here, “transactional publication” means the final state can be recovered to
either a complete new result or a complete prior result. Because five
independent files must be replaced in sequence, this is not a single
instantaneous filesystem-atomic operation. If publication or the Trash move is
interrupted, the next launch recovers it before starting another analysis.

The validated eGFP execution order is: Cellpose-SAM → 90 candidates →
candidate ranking → 3D DAPI inventory → nucleus ownership → Soma/Processes.

## 10. Validation and release gates

- `validation/source_manifest.json`: scientific core and shared-runtime load order.
- `validation/release_contract.json`: runtime interfaces, resources, and
  documentation contract.
- `validation/release_package_files.json`: final release-package file hashes.
- `tests/`: module loading, eGFP baseline, GFAP-only, Split, Enlarge, Fiji
  transactions, measurement, and publication safety.
- The optional official InstanSeg reference comparison reads
  `PROJECT_LEAP_INSTANSEG_FIXTURE_DIR`, an absolute directory containing
  `test-input.npy` and `test-output_instance_segmentation.npy`. The test verifies
  their fixed SHA-256 values; the fixture files are not distributed.
- A version is released only after source, resources, documentation, tests,
  and final package hashes all agree.
