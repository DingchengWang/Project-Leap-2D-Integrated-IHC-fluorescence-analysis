# Module and standalone-package tests

Run the tests from the repository root on macOS with Python 3.10 or later. The current module tests also require scientific Python packages, including NumPy, SciPy, scikit-image, and tifffile.

```sh
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
```

The suite covers two parts of this repository:

- **Current module:** desktop worker requests and paths, shared runtime and model identity checks, required Fiji interaction, GFAP pipeline ordering, DAPI workload limits, and final ROI archive consistency and recovery. These tests use controlled fixtures and mocked analysis stages.
- **Standalone v1.0.1 package:** fixed-tag release identity and digests, program path restrictions, recovery of corrupted or missing files, executable permissions, transaction rollback and interrupted recovery, environment and workspace locks, installation without overwriting existing targets, and retention of inputs, results, and run state.

Standalone maintenance and installer fixtures use temporary working packages and Support directories, synthetic input/result sentinels, local release transport, and mock Python/Fiji/model components. Their child processes run with a macOS network-deny sandbox. Fixture shells use temporary `HOME` and `ZDOTDIR` values to avoid user startup files, plus a fixed system `PATH`.

## Test the standalone release archive

Build the v1.0.1 release assets, then run the archive integration script:

```sh
python3 tools/build_release_zip.py
python3 -B tests/test_final_archive_integration.py
```

This script has a standalone `main()` and is **not executed by unittest discovery**. It restores program files from the built release ZIP using a local transport fixture and reaches the real maintenance shell chain. Both test commands report in the terminal and do not create repository audit logs.

The suite does not run real model inference, a live Fiji GUI, full image analysis, dependency installation, or live GitHub release downloads. Published asset availability and anonymous download access require a separate post-publication check.
