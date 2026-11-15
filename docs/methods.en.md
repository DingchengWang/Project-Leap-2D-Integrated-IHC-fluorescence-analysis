# Analysis methods and measurement scope

[中文](methods.zh.md) · [Overview](../README.md) · [User guide](guide.en.md)

Project Leap 2D produces two-dimensional astrocyte regions and measures one target channel within them. Module **1.1.0** adds the Project Leap input, task, and output workflow while retaining scientific rules version **1.0.1**. The original multichannel TIFF is read-only. Its selected image series, time point, and channel assignments determine the single-channel working images passed to the analysis engine.

## Input data

The desktop importer accepts TIFF data with explicit Z, Y, and X axes, optional C and T axes, and a complete scalar YX image on each TIFF page. It accepts `uint8`, `uint16`, and `float32` without converting the pixel values. Each pixel must have one scalar sample; supported photometric representations are MINISBLACK and PALETTE. A merged RGB image and a native microscope acquisition file are not supported inputs.

Positive physical spacing in X, Y, and Z must be readable from supported OME, ImageJ/SCIFIO, or TIFF calibration fields. The application checks the axes, pages, data type, and calibration. Multiple image series or time points require an explicit choice; one run processes one chosen series/time point. Exact, unique OME channel-name matches may prefill roles, which the user confirms.

The analysis needs one DAPI channel, at least one eGFP/GFAP channel, and one measurement channel from KCNN1, KCNN2, KCNN3, or KCNJ10. Each role has a separate source channel. The importer writes calibrated ZYX working TIFFs outside the source directory, preserving the selected pixels and data type. The engine then checks that channel geometry and calibration agree.

## Structural channels determine the analysis route

DAPI supplies nuclear evidence. eGFP and/or GFAP supply structural evidence for the cell regions. The target channel supplies the fluorescence values to be measured; it does not define the automatic ROIs or select the analysis depth.

| Selected structural channels | Route and age configuration |
|---|---|
| eGFP, with or without GFAP | The eGFP route compares five Z windows and 18 candidate profiles per window. Automatic selects a Mature or Neonatal processing profile from morphology; an explicit user selection takes precedence. |
| GFAP without eGFP | The independent GFAP-only route uses DAPI and GFAP evidence and supports mature astrocytes only. Automatic uses the mature configuration; Neonatal is rejected. |

### eGFP route: choose depth, then construct compartments

The program excludes at least **5.0 µm** from each physical end of the stack and places **five** equally spaced windows in the remaining depth. The target window width is **9.5 µm**. It rounds `9.5 / Z-step` to the nearest interval count, with half steps rounded upward, then includes both endpoint planes. The actual width therefore depends on the acquired Z spacing. The end guard is rounded upward to whole slices. If five distinct window starts cannot fit, analysis stops with `z_step_limit`.

Within each window, six Morphology Baseline, six Structural Refinement, and six Distributional Threshold profiles propose Whole Cell regions. Each window shares a projection set, Cellpose-SAM mask, and prepared structural features across its 18 candidates. The program selects a local best candidate, then compares the five position-best candidates using nuclear association and axial retention, Whole Cell quality, centering, and optical evidence. An analyzable cell requires a unique correspondence between one axial DAPI nucleus and one Whole component in both directions. Axial nuclear evidence is derived from normalized per-plane DAPI P85 profiles, where P85 is the 85th percentile within a nuclear region. An exact tie at the highest final position score stops the run with `z_selection_ambiguous`.

The measurement channel undergoes technical quality checks but has zero weight in this position score and does not choose Z or Whole Cell. The age configuration is applied after the final window and initial Whole Cell selection. It governs later processing; it does not identify an animal's actual age. The eGFP-only age calibration was developed from two known-Mature and three P3 Neonatal reference samples. Other ages or imaging conditions require appropriate known-age references.

### GFAP-only route: associate nuclei with GFAP structure

The program selects a continuous active Z range from GFAP evidence. An included InstanSeg CPU TorchScript model generates DAPI nuclear candidates. The program links nuclei across depth, checks their association with GFAP structure, resolves exclusive cell ownership, and constructs Whole Cell, Soma, and Processes. InstanSeg supplies nuclear candidates; the surrounding analysis defines and validates the cell regions. The target projection is loaded for measurement after the regions have been defined. This route does not run the eGFP route's five-window, 90-candidate comparison.

## Compartments and Fiji review

Whole Cell is the complete retained cell region; Soma is its cell-body region; Processes is the part of Whole Cell outside Soma. The three label sets share cell identities and satisfy:

```text
Whole Cell = Soma ∪ Processes
Soma ∩ Processes = ∅
Processes = Whole Cell − Soma
```

Fiji shows these regions over composite structural displays and separate raw target-channel projections. The user checks the proposed boundaries and applies linked Delete, Merge, Split, Enlarge, or Trim Processes edits where appropriate. Editing updates the three regions together. Split and Enlarge use local evidence and can decline an unsupported change. Merge applies a **1.0 µm** Soma-boundary proximity condition. Trim preserves Soma and checks that retained processes are not newly disconnected. The [user guide](guide.en.md) explains the controls and their limits at the point of use.

The initial Fiji dialog includes both **Review ROIs** and **Measure Now**. The guide uses Review ROIs so the user inspects the proposed regions before accepting measurements. Entering the editing workflow is recorded separately from whether edits were made; `Manual_Review_Used` is a workflow record, not proof that the boundaries are biologically correct.

## Raw-grayscale measurement and saved results

The selected Z range and projection method determine which source pixels form the measurement image. The final ROI determines which part of that image is measured. Fiji measures the raw target-channel projection; composite colors and display contrast are not measurement data. The pipeline applies no background subtraction or intensity normalization to the measured target intensities. Structural image processing used to construct regions is separate from this measurement path.

The workbook contains Whole Cell, Soma, and Processes sheets with area, mean, median, minimum, maximum, integrated density, and raw integrated density, plus the channel, projection, Z bounds, age profile, and review record. These are measurements of a selected **2D projection**, not a cell's 3D volume or an automatic background-corrected expression estimate. The experiment's acquisition settings and sampling design still determine which comparisons are meaningful.

Before saving results, the program checks synchronized identities, compartment partitioning, Fiji measurement records, overlay dimensions, and workbook structure. Module v1.1.0 adds `IHC_2D_ROI_Archive.zip` to the existing three PNGs, TXT report, and XLSX workbook. The archive includes final Whole/Soma/Processes label TIFFs, native ImageJ ROIs, identity mappings, calibration, selected Z/projection, and runtime provenance. The exporter validates correspondence to the completed Fiji measurements; it does not reconstruct or redefine the ROIs. All six outputs enter the same result-publication transaction.

Technical consistency checks do not establish biological validity across tissues, ages, staining protocols, microscopes, or laboratories. Use reference samples and boundary review appropriate to the intended experiment. The v1.1.0 workflow changes do not establish an accuracy or speed improvement.

## Source map

- [Depth-window construction](../engine/project_leap_2d/images/image_processing.py), [eGFP orchestration](../engine/project_leap_2d/analysis_controller.py), and [candidate selection](../engine/project_leap_2d/segmentation/candidate_selection.py).
- [GFAP-only orchestration](../engine/project_leap_2d/analysis_modes/gfap_only/gfap_only_pipeline.py) and [cell ownership and compartments](../engine/project_leap_2d/analysis_modes/gfap_only/gfap_only_analysis.py).
- [Fiji review and measurement](../engine/project_leap_2d/fiji_review/resources/astrocyte_roi_reviewer.groovy), [workbook output](../engine/project_leap_2d/reporting/excel_results.py), [final ROI archive](../engine/project_leap_2d/final_roi_archive.py), and [desktop adapter](../engine/project_leap_2d/desktop_adapter.py).
- The [Project Leap host repository](https://github.com/DingchengWang/Project-Leap) owns TIFF import, channel mapping, component management, and the desktop interface. Third-party sources and licensing are recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
