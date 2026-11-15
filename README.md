# Project Leap 2D — Integrated IHC fluorescence analysis

[中文](README_中文.md) · [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/) · [User guide](docs/guide.en.md) · [Methods](docs/methods.en.md)

Project Leap 2D measures a target fluorescence channel within astrocyte whole-cell, soma, and process regions. The module uses DAPI plus eGFP and/or GFAP to select analysis depth and cell regions. After ROI review in Fiji, the module measures the target channel's raw grayscale values in a two-dimensional maximum-intensity projection of the selected Z range.

This repository contains source and documentation for **v1.1.1** of the **Astrocyte Fluorescence Intensity Measurement** module. Install it through **Project Leap v1.0.0**, available from the [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/). Version changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Prepare an image and install the module

Use an **Apple Silicon Mac running macOS 27 or later**. In Project Leap, open **Modules & Maintenance**, install **Astrocyte Fluorescence Intensity Measurement**, and wait for **Module ready**. The application installs the scientific runtime, models, and Fiji with the module.

Prepare one multichannel `.tif` or `.tiff` with separate DAPI, eGFP and/or GFAP channels, plus one target: KCNN1, KCNN2, KCNN3, or KCNJ10. The image must retain its Z axis and physical pixel and slice spacing. Export microscope acquisition files as calibrated TIFFs; merged RGB pictures are unsuitable. GFAP-only analysis supports mature astrocytes only.

## Analyze and review

Automatic analysis produces linked whole-cell, soma, and process regions for review in Fiji. The [user guide](docs/guide.en.md) explains how to inspect and adjust these regions, complete measurement, and open the results. [Methods](docs/methods.en.md) describes how the regions and measurements are calculated.

## Read and retain the results

Each completed run saves three PNG overlays, a TXT report, an XLSX workbook, and `IHC_2D_ROI_Archive.zip`. The workbook reports area and fluorescence measurements for Whole Cell, Soma, and Processes, together with cell IDs, the selected Z range, and the review record. The target intensities retain the source grayscale scale, without background subtraction or intensity normalization.

The ROI archive preserves final label masks, ImageJ ROIs, and their correspondence to the measurements for inspection or extraction. Keep the result folder with the original TIFF.

## Legacy v1.0.1 script package

The standalone script package remains available for its existing users. Its installation, input handling, and system requirements are described in its own documentation:

- [v1.0.1 Release](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/tag/v1.0.1), [installation ZIP](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.zip), and [repair manifest](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.manifest.json).
- [Installation instructions](distribution/INSTALL_English.md) and [legacy application manual](<Project Leap 2D V1.0.1/README/README EN.md>).
- [`Project Leap 2D V1.0.1/`](<Project Leap 2D V1.0.1/>) retains `Run Analysis.command`, `Repair.command`, and `Manual Command.txt`. The legacy `Repair.command` uses the exact v1.0.1 Release and its matching assets.

The legacy package requires backed-up, presplit input copies in `Sample Image` and moves successfully processed inputs to macOS Trash. Follow that package's manual when using it. GitHub's **Code → Download ZIP** provides a source snapshot; application installers are available from the website.

## Source and attribution

Current module code is in [`engine/project_leap_2d/`](engine/project_leap_2d/), with module tests in [`tests/`](tests/). The [Project Leap repository](https://github.com/DingchengWang/Project-Leap) contains the desktop interface and component management. The preserved legacy package and its distribution tools remain separate from the current module.

Dingcheng Wang defined the scientific criteria, workflow decisions, and review requirements. AI assisted software implementation, debugging, and packaging.

Original project code uses the [Apache License 2.0](LICENSE). Third-party software and models retain their own terms; sources, licenses, and requested citations are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).
