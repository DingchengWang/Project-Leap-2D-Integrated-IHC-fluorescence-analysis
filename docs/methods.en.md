# Analysis methods and measurement scope

[中文](methods.zh.md) · [Overview](../README.md) · [User guide](guide.en.md)

Project Leap 2D uses a calibrated Z-stack to select analysis depth and associate astrocyte structures with nuclei, then measures one target channel within two-dimensional cell regions. This document covers module **v1.1.1**, with scientific rules version **1.0.1**. Software changes are listed in [CHANGELOG.md](../CHANGELOG.md).

## Input data and channel roles

The desktop importer accepts TIFF data with explicit Z, Y, and X axes, optional C and T axes, and a complete scalar YX image on each TIFF page. It accepts `uint8`, `uint16`, and `float32` without converting the pixel values. Each pixel must have one scalar sample; supported photometric representations are MINISBLACK and PALETTE. Export native microscope acquisition files as calibrated TIFFs before import. Merged RGB images are not supported.

Positive physical spacing in X, Y, and Z must be readable from supported OME, ImageJ/SCIFIO, or TIFF calibration fields. The application checks the axes, pages, data type, and calibration. Multiple image series or time points require an explicit choice; one run processes one chosen series/time point. Exact, unique OME channel-name matches may prefill roles, which the user confirms.

The analysis needs one DAPI channel, at least one eGFP/GFAP channel, and one measurement channel from KCNN1, KCNN2, KCNN3, or KCNJ10. Each role has a separate source channel. The importer keeps the original TIFF read-only and writes calibrated ZYX working TIFFs in a separate task directory, preserving the selected pixels and data type. The engine then checks that channel geometry and calibration agree.

DAPI supplies nuclear evidence; eGFP and/or GFAP supply structural evidence for cell regions. These channels determine automatic ROIs and the selected Z range. The target channel supplies the fluorescence values for measurement. Its technical quality checks can stop a run, but it does not define automatic ROIs or choose analysis depth.

## Select depth and construct cell regions

| Selected structural channels | Analysis route and age configuration |
|---|---|
| eGFP, with or without GFAP | The eGFP route compares five Z windows and 18 candidate profiles per window. Automatic selects a Mature or Neonatal processing profile from morphology; an explicit user selection takes precedence. |
| GFAP without eGFP | The GFAP-only route uses DAPI and GFAP evidence and supports mature astrocytes only. Automatic uses the mature configuration; Neonatal is rejected. |

### eGFP route

The program excludes at least **5.0 µm** from each end of the stack and places **five** equally spaced windows in the remaining depth. The target window width is **9.5 µm**. It rounds `9.5 / Z-step` to the nearest interval count, with half steps rounded upward, then includes both endpoint planes. The actual width therefore depends on the acquired Z spacing. The end guard is rounded upward to whole slices. If five distinct window starts cannot fit, analysis stops with `z_step_limit`.

Within each window, six Morphology Baseline, six Structural Refinement, and six Distributional Threshold profiles propose Whole Cell regions. The 18 candidates share a projection set, Cellpose-SAM mask, and prepared structural features. The program selects a local best candidate, then compares the five position-best candidates using nuclear association and axial retention, Whole Cell quality, centering, and optical evidence. The target channel has zero weight in this position score. An exact tie at the highest final position score stops the run with `z_selection_ambiguous`.

An analyzable cell requires a unique correspondence between one axial DAPI nucleus and one Whole component in both directions. Axial nuclear evidence is derived from normalized per-plane DAPI P85 profiles, where P85 is the 85th percentile within a nuclear region. This evidence helps determine cell identity across the selected depth.

The age configuration is applied after the final window and initial Whole Cell selection and governs subsequent processing. Automatic uses morphology to select a processing profile, not to establish the animal's actual age. The eGFP-only age calibration was developed from **two known-Mature and three P3 Neonatal reference samples**.

### GFAP-only route

The program selects a continuous active Z range from GFAP evidence. An included InstanSeg CPU TorchScript model generates DAPI nuclear candidates. The program links nuclei across depth, checks their association with GFAP structure, resolves exclusive cell ownership, and constructs Whole Cell, Soma, and Processes. InstanSeg supplies the nuclear candidates; the subsequent analysis constructs and validates the cell regions. The target projection is loaded for measurement after the regions have been defined. The five-window, 90-candidate comparison belongs to the eGFP route only.

## Review the regions in Fiji

Whole Cell is the complete retained cell region; Soma is its cell-body region; Processes is the part of Whole Cell outside Soma. The three label sets share cell identities and satisfy:

```text
Whole Cell = Soma ∪ Processes
Soma ∩ Processes = ∅
Processes = Whole Cell − Soma
```

Fiji shows these regions over composite structural displays and separate raw target-channel projections. The user checks the proposed boundaries and applies linked Delete, Merge, Split, Enlarge, or Trim Processes edits where needed. Editing updates the three regions together. Split and Enlarge use local evidence and can reject an unsupported change. Merge requires the selected Soma boundaries to form a proximity chain with gaps no greater than **1.0 µm**. Trim preserves Soma and checks that retained processes are not newly disconnected. The [user guide](guide.en.md) explains each control and its conditions.

The initial Fiji dialog offers **Review ROIs** and **Measure Now**. The guide uses Review ROIs to inspect the proposed regions before measurement. `Manual_Review_Used` records whether this review workflow was entered; **Review Audit** separately records applied and reverted edits. Boundary acceptance remains the user's decision during review.

## Measure fluorescence and save the results

Fiji forms a **two-dimensional maximum-intensity projection** of the target channel over the selected Z range. At each XY position, the projection retains the maximum source intensity within that range; the final ROI determines which projected pixels are measured. Target intensities retain their original grayscale scale, with **no background subtraction or intensity normalization**. Composite colors and display contrast affect viewing only. Processing of the structural channels for region construction is separate from target-intensity measurement.

The workbook has Whole Cell, Soma, and Processes sheets containing area, mean, median, minimum, maximum, integrated density, and raw integrated density. It also records the target channel, projection, Z bounds, age profile, and review history. Area describes the region in the selected 2D projection, and intensity statistics describe its projected pixels. Z-stack evidence supports cell identification and region selection; these outputs quantify projected regions rather than 3D cell volume.

Before saving, the program checks synchronized identities, compartment partitioning, Fiji measurement records, overlay dimensions, and workbook structure. It saves three PNG overlays, a TXT report, an XLSX workbook, and `IHC_2D_ROI_Archive.zip` together. The archive contains final Whole/Soma/Processes label TIFFs, native ImageJ ROIs, identity mappings, calibration, selected Z/projection, and runtime provenance. The exporter retains the completed ROIs and checks their correspondence to the Fiji measurements. The archive preserves these shapes for inspection or extraction; the application currently has no one-click route to reopen a completed fluorescence task for editing.

Use known-age references appropriate to the experiment and inspect cell boundaries when applying the workflow to other ages, tissues, staining protocols, microscopes, or laboratories. These checks address the biological suitability of the selected regions. Meaningful comparisons of the reported raw intensities also depend on the experiment's acquisition settings and sampling design.

## Source map

- [Depth-window construction](../engine/project_leap_2d/images/image_processing.py), [eGFP orchestration](../engine/project_leap_2d/analysis_controller.py), and [candidate selection](../engine/project_leap_2d/segmentation/candidate_selection.py).
- [GFAP-only orchestration](../engine/project_leap_2d/analysis_modes/gfap_only/gfap_only_pipeline.py) and [cell ownership and compartments](../engine/project_leap_2d/analysis_modes/gfap_only/gfap_only_analysis.py).
- [Fiji review and measurement](../engine/project_leap_2d/fiji_review/resources/astrocyte_roi_reviewer.groovy), [workbook output](../engine/project_leap_2d/reporting/excel_results.py), [final ROI archive](../engine/project_leap_2d/final_roi_archive.py), and [desktop adapter](../engine/project_leap_2d/desktop_adapter.py).
- The [Project Leap host repository](https://github.com/DingchengWang/Project-Leap) owns TIFF import, channel mapping, component management, and the desktop interface. Third-party sources and licensing are recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
