# Project Leap 2D — Integrated IHC fluorescence analysis

[中文说明](README_中文.md)

Project Leap 2D analyzes split, single-channel immunohistochemistry (IHC)
fluorescence Z-stacks on Apple Silicon Macs. The program creates regions of
interest (ROIs) for the whole cell (Whole Cell), cell body (Soma), and cell
processes (Processes), which you review in Fiji. It measures the selected
fluorescence channel from untouched grayscale data and produces overlays, an
analysis report, and an Excel workbook.

The application is in
[`Project Leap 2D V1.0.1/`](<Project Leap 2D V1.0.1/>).
Distribution tools and documentation are in the repository root.

## First installation

Download `Project-Leap-2D-V1.0.1.zip` from the published `v1.0.1`
[GitHub Release](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases).

Extract the ZIP, keeping `install_macos.command`, `payload_sha256.txt`, and
the `Project Leap 2D V1.0.1` folder together at the outer level. Double-click
that `install_macos.command` and follow the Terminal prompts. The installer
checks the accompanying package, installs it by default in
`~/Desktop/Project Leap 2D V1.0.1`, and then sets up or checks the shared
dependency environment. It stops if the destination already exists.

First-time environment setup requires network access. The dependencies are
stored in `~/Applications/Project Leap 2D Support`. Installation details and
alternative destinations are described in the Release ZIP's
`INSTALL_English.md`.

## Daily use and repair

Inside the installed working package:

1. Place one batch of backed-up, split single-channel Z-stack TIFF files in
   `Sample Image`.
2. Double-click `Run Analysis.command`, then follow the prompts for Fiji review.
   Results are written to `Result`.
3. If files are missing or damaged, or startup checks fail, double-click
   `Repair.command`. It checks this version's program files, restores them if
   needed, and then checks the dependency environment and rebuilds it if needed.

Before each analysis, the program runs a quick check of the local environment
without accessing the network. Every repair needs network access to retrieve
and verify `Project-Leap-2D-V1.0.1.manifest.json` from the published `v1.0.1`
Release. It downloads and verifies `Project-Leap-2D-V1.0.1.zip` only when
program files need restoration. If repair cannot retrieve or verify the
required release information or files, it stops before checking or rebuilding
the environment.

Repair restores damaged or missing program files from the same version and preserves
`Sample Image`, `Result`, and `Analysis Package/Run State`. It does not upgrade
the program or fix bugs already present in that version. If the repair script
itself is missing or cannot start, download it again from the same release.

Input requirements, analysis routes, Fiji editing, and output descriptions are
in the [English application manual](<Project Leap 2D V1.0.1/README/README EN.md>).
`Manual Command.txt` inside the working package provides Terminal commands for
running the analysis and repair scripts.

## Input backups and run state

Work from a copy of each input batch and keep an independent backup. TIFF files
used by the analysis move to macOS Trash only after Fiji completes, all
validation passes, and the five output files have been successfully saved
together in `Result`. The input files stay in place if a safety check stops the
run, you cancel, an exception occurs, or the output files cannot be saved.

The program stores run locks and recovery records in
`Analysis Package/Run State`; do not clear this folder manually. Git excludes
these runtime files, input images, and analysis results.

## Repository clone, Code ZIP, and Release ZIP

A Git clone contains tracked source files and Git history. GitHub's
**Code → Download ZIP** and automatically generated **Source code** archives
contain source snapshots. They do not include the complete
installation package or empty working directories that Git does not track.
Use the versioned Release ZIP for installation.

For source inspection or development, run `./prepare_workspace.command` from
the repository root after cloning or extracting a source archive. It creates
`Sample Image`, `Result`, and `Analysis Package/Run State` inside the working
package, preserves existing contents, and rejects redirected or invalid target
directories. This command prepares the working folders; it does not install
dependencies.

## Supported system and scientific scope

V1.0.1 is designed for Apple Silicon Macs running macOS 11 or later. See the
application manual for the required channels and calibration. GFAP-only
analysis supports mature astrocytes: a missing age marker or an explicit
`mature` marker uses the mature configuration; a recognized `neonatal` marker
or conflicting age markers stops analysis.

Passing software integrity and automated checks does not establish biological
validity across tissues, ages, staining protocols, disease models, microscopes,
or laboratories. Use appropriate reference samples to validate the analysis
for your imaging conditions.

## License and third-party software

Original project code is licensed under the [Apache License 2.0](LICENSE).
Third-party software and models remain subject to their own terms. Their
licenses, sources, training-data notices, and requested citations are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).
