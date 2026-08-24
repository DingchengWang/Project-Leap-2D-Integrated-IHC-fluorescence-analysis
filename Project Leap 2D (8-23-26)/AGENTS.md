# Project Leap 2D development safety rules

- Keep the validated eGFP analysis order, 90-candidate catalog, selected Z,
  Whole geometry, Soma/Processes partition, and raw-grayscale KCNN1/KCNN2
  measurement unchanged unless a separately approved scientific change is
  supported by targeted regression.
- Keep the candidate catalog ordered as 30 Morphology Baseline, 30 Structural
  Refinement, and 30 Distributional Threshold candidates. Candidate producers,
  runtime metadata, and reports must use these functional names directly.
- Whole, Soma, and Processes must always use synchronized cell IDs and satisfy
  `Whole = Soma union Processes` without overlap. Delete, Merge, Split, Enlarge,
  Revert, and renumbering must update all three compartments together.
- DAPI + GFAP-only is an independent route. Any recognized eGFP stack keeps the
  established eGFP route, and measurement-channel pixels must never define an
  ROI.
- GFAP-only is mature-astrocyte only. No age marker or an explicit `mature`
  marker uses the mature configuration; a recognized `neonatal` marker must
  stop before analysis. Do not add neonatal GFAP-only execution without a
  separately approved scientific change supported by real-sample validation.
  This restriction must not change the eGFP route's automatic age
  classification when filenames provide no age decision.
- Set BLAS/OpenMP thread variables before importing NumPy, SciPy, OpenCV, or
  model code. Optional InstanSeg/GFAP modules must remain lazily loaded.
- Treat `validation/source_manifest.json`,
  `validation/release_contract.json`, model/resource hashes, and the full test
  suite as release gates. Do not refresh final hashes before code, documents,
  and package cleanup are complete.
- Publish only a complete validated Result bundle. Move accepted source TIFFs
  to macOS Trash only after successful publication; retain them after safety
  stops, cancellation, exceptions, or validation failure.
- Preserve failed-run diagnostics and Cell Edit context. Never expose an
  unfinished Fiji action or silently accept an ambiguous biological result.
- Installer repair may change only a release whose ready state, contract, and
  canonical paths prove ownership. Attempt repair once, retain the old release
  until the replacement passes the full integrity check, restore it on failure,
  and never delete an unknown release or repair record. Installation and
  analysis must hold the same macOS kernel-managed support-side environment
  lock while either can use or replace Python, models, or Fiji. State,
  `INSTALLING`, and repair records must be published from private temporary
  paths by same-volume rename; never follow a pre-existing symbolic link.
- Keep the trusted offline integrity baselines for managed Python, locked
  wheels, and Fiji/Java synchronized with the checker and bootstrap integrity
  manifest. Build a new baseline only from an independently verified fixed
  source; never learn a replacement baseline from an environment merely
  because it is installed.
