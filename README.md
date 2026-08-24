# Project Leap 2D — Integrated IHC fluorescence analysis

[中文说明](README_中文.md)

Project Leap 2D is a macOS application package for two-dimensional analysis of
split, single-channel immunohistochemistry fluorescence Z-stacks. It builds and
reviews Whole Cell, Soma, and Processes regions of interest, measures the
selected fluorescence channel from untouched grayscale data, and publishes
validated overlays, a report, and an Excel workbook.

The runnable application package is located in
[`Project Leap 2D (8-23-26)/`](<Project Leap 2D (8-23-26)/>). The files at the
repository root prepare and document the GitHub distribution; they are not a
second implementation of the analysis.

## Prepare a clone before first use

After cloning the repository, open Terminal in the repository root and run:

```bash
./prepare_workspace.command
```

This creates the empty working directories required by the application. Then
enter the runnable package and follow its installation and operating
instructions:

```bash
cd "Project Leap 2D (8-23-26)"
./Installation/macOS/install_macos.command
```

See the inner [English application manual](<Project Leap 2D (8-23-26)/README_English.md>)
for input requirements, installation checks, Fiji review, outputs, and normal
startup.

## Protect source images

Use only a backed-up working copy of each input batch. After Fiji has completed,
all validation has passed, and the production output bundle has been published
successfully, TIFF files actually used by the analysis are moved to macOS
Trash. Safety stops, cancellation, exceptions, and publication failures retain
the input files, but they do not replace the need for an independent backup.

## Repository clone, source archive, and release ZIP

A Git clone contains the tracked files and Git history. Git does not preserve
empty directories, so a new clone must be prepared with
`./prepare_workspace.command` before the application is used.

GitHub's automatically generated **Source code** archives are generic snapshots
of tracked repository files. They do not preserve the complete, prepared
application-package layout. Use the separately published, verified **Release
ZIP** when a ready-to-unpack application package is required. Do not treat the
automatic source archive as equivalent to that Release ZIP.

## Supported system

This release is designed for Apple Silicon Macs running macOS. It is not
presented as a Windows, Linux, or Intel Mac release. Detailed minimum-version
and installation requirements are maintained in the inner application manual.

## Verification boundary

The engineering checks cover the frozen package files, installation and release
contracts, analysis invariants, and repeatable test outputs. This release
supports GFAP-only analysis only for mature astrocytes. A missing age marker or
an explicit `mature` marker uses the mature GFAP-only configuration; a
recognized `neonatal` marker stops before analysis. The author tested the
program with mature-astrocyte GFAP-only samples held outside this repository;
the samples and their results are not distributed. Automated tests also cover
synthetic GFAP-only cases. The current validation scope does not establish
broad biological validation across tissues, developmental stages, staining
protocols, disease models, microscopes, or laboratories.

## License and third-party software

Code written for this project is released under the
[Apache License 2.0](LICENSE). Third-party software, models, training-data
notices, licenses, sources, and requested citations remain governed by their
own terms and are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [`LICENSES/`](LICENSES/).
