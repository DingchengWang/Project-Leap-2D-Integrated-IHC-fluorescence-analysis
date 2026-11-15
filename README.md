# Project Leap 2D — Integrated IHC fluorescence analysis

[中文](README_中文.md) · [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/) · [User guide](docs/guide.en.md) · [Methods](docs/methods.en.md)

Project Leap 2D measures a target fluorescence channel within astrocyte whole-cell, soma, and process regions. Version **1.1.0** runs as the **Astrocyte Fluorescence Intensity Measurement** module in Project Leap. Import a multichannel TIFF, choose its image series and time point when prompted, and assign the channel roles. The application prepares single-channel working images, runs the established analysis, opens Fiji for ROI review, and saves six result files. The original TIFF stays unchanged in its original location.

This repository contains the module source and documentation. The Project Leap application provides installation and maintenance. The v1.1.0 public release is being prepared; the new application download and module catalog are not yet published. Release availability will be announced through the [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/).

## Use the module

The current Project Leap application targets **Apple Silicon Macs running macOS 27 or later**. In **Modules & Maintenance**, install **Astrocyte Fluorescence Intensity Measurement** and wait for **Module ready**. The application manages the module, scientific runtime, models, and Fiji; there is no separate manual module installer.

1. Import a supported `.tif` or `.tiff` with DAPI, eGFP and/or GFAP, and one target: KCNN1, KCNN2, KCNN3, or KCNJ10.
2. Select the image series/time point when required, confirm the channel mapping, and choose **Sample Age**.
3. Click **Start analysis**. The application splits the selected channels into working copies and runs the analysis.
4. Review the proposed regions in Fiji, make any needed corrections, and finish the review.
5. Open **View results** after **Completed** to inspect the overlays, report, measurements, and final ROI archive.

The [user guide](docs/guide.en.md) explains each control, the expected feedback, and how to handle rejected edits. Image metadata is checked automatically; channel identity and cell boundaries require your judgment. GFAP-only analysis supports mature astrocytes only.

## What changes in v1.1.0

| Area | v1.1.0 module |
|---|---|
| Input preparation | Import one multichannel TIFF; select a series/time point and map source channels in the application. Single-channel working copies are generated automatically. |
| Original images | Read-only source TIFFs remain in place; working copies live in a separate task directory. |
| Installation and results | Project Leap manages components, task status, result files, and history. |
| Final ROI retention | Adds `IHC_2D_ROI_Archive.zip` to the five existing outputs, preserving final geometry and its correspondence to measurements. |

The scientific rules remain version **1.0.1**. Z selection, automatic segmentation, age-profile selection, Fiji editing including Trim Processes, and raw-grayscale measurement were already part of the earlier workflow. See [Methods](docs/methods.en.md) and [CHANGELOG](CHANGELOG.md).

## Legacy v1.0.1 script package

The standalone script package remains available for its existing users. Its installation, input handling, and system requirements are described in its own documentation:

- [v1.0.1 Release](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/tag/v1.0.1), [installation ZIP](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.zip), and [repair manifest](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.manifest.json).
- [Installation instructions](distribution/INSTALL_English.md) and [legacy application manual](<Project Leap 2D V1.0.1/README/README EN.md>).
- [`Project Leap 2D V1.0.1/`](<Project Leap 2D V1.0.1/>) retains `Run Analysis.command`, `Repair.command`, and `Manual Command.txt`. The legacy `Repair.command` uses the exact v1.0.1 Release and its matching assets.

The legacy package requires backed-up, presplit input copies in `Sample Image` and moves successfully processed inputs to macOS Trash. Follow that package's manual when using it. GitHub's **Code → Download ZIP** is a source snapshot, not an application installer.

## Source and attribution

Current module code is in [`engine/project_leap_2d/`](engine/project_leap_2d/), with module tests in [`tests/`](tests/). The [Project Leap repository](https://github.com/DingchengWang/Project-Leap) contains the desktop interface and component management. The preserved legacy package and its distribution tools remain separate from the current module.

Dingcheng Wang defined the scientific criteria, workflow decisions, and review requirements. AI assisted software implementation, debugging, and packaging.

Original project code uses the [Apache License 2.0](LICENSE). Third-party software and models retain their own terms; sources, licenses, and requested citations are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).
