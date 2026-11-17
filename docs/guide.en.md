# Fluorescence measurement guide

[中文](guide.zh.md) · [Overview](../README.md) · [Methods](methods.en.md)

This guide covers image import, automatic generation of regions of interest (ROIs), the choice to measure directly or review regions in Fiji, and the resulting fluorescence measurements. It is written for laboratory users familiar with fluorescence images and describes module **v1.1.1** in Project Leap.

## Before starting

Download **Project Leap v1.0.0** from the [Project Leap website](https://dingchengwang.github.io/Project-Leap/en/) and install it on an Apple Silicon Mac running macOS 27 or later. Open **Modules & Maintenance**, install **Astrocyte Fluorescence Intensity Measurement**, and wait for **Module ready**. The application installs the required scientific runtime, models, and Fiji. The control names below follow the English interface; choose English under **Settings → Interface language** if needed.

Have one multichannel TIFF containing separate DAPI, eGFP and/or GFAP channels, plus one target: KCNN1, KCNN2, KCNN3, or KCNJ10. Project Leap splits these channels into working images automatically; use the multichannel TIFF as input. A merged RGB picture is unsuitable. The TIFF must retain its depth axis and physical pixel/slice spacing; the program reads these automatically. Exact supported layouts are listed under [Input data](methods.en.md#input-data-and-channel-roles).

## 1. Import the image

Select **Astrocyte Fluorescence Intensity Measurement** in the sidebar. In **Image import**, click **Choose image…**, or drag one `.tif`/`.tiff` into the import area. If prompted, select the **Image series** and **Time point** you intend to analyze. This run processes that selection.

The application displays image dimensions, channel and Z-plane counts, data type, and voxel calibration. It checks this metadata automatically. If the file has missing calibration or an unsupported layout, read the message, obtain a correctly exported TIFF from the image provider, and import it again.

## 2. Confirm channels and sample age

Under **Channel mapping**, confirm any prefilled choices or assign C1, C2, and the other source channels to **DAPI**, **eGFP**, and **GFAP**. Choose **Not included** for an absent marker. Include DAPI and at least one of eGFP/GFAP. For **Target channel**, choose both its marker and its source channel. Assign each selected source channel to only one marker; unassigned channels are ignored.

Channel names can prefill a choice, but you must confirm what was stained. Display colors do not identify markers. Consult the image provider if a channel's identity is unknown.

With eGFP, **Automatic** under **Sample Age** selects a processing profile from image morphology. If the sample category is known, you can choose **Mature** or **Neonatal**. The route with GFAP as the sole structural channel is for **mature samples only**, using either Automatic or Mature. Importing another image resets the selection to Automatic.

## 3. Start the analysis

Click **Start analysis** in the analysis-status panel. The application creates calibrated single-channel working TIFFs in a separate task directory, then selects the analysis depth and proposes Whole Cell, Soma, and Processes ROIs. It preserves the original TIFF, its pixel values, and its location.

If Start analysis is unavailable, check the displayed module status, required series/time selection, channel assignments, and GFAP-only age setting. During analysis, keep Project Leap and Fiji open. When the status becomes **Awaiting Fiji review**, click **Review in Fiji** to bring Fiji forward.

## 4. Choose direct measurement or ROI review in Fiji

In **Astrocyte ROI Reviewer**, choose how to proceed:

- **Measure Now** measures the automatically generated ROIs without entering the editing workflow. The program then validates and saves the results; continue to step 6 to read them.
- **Review ROIs** opens the linked views and editing controls described below.
- **Cancel Analysis**, or closing this initial dialog, cancels the run.

With **Review ROIs**, three managers open: **Whole Astrocyte Cell ROI Manager**, **Astrocyte Soma ROI Manager**, and **Astrocyte Processes ROI Manager**. These control whole-cell regions, cell bodies, and processes. Each has a Composite view and a Raw Grayscale view, giving six views of the same image. **Astrocyte ROI Reviewer-Waiting** contains the completion and recovery controls.

Click a cell's row in any manager to highlight it across the views. **Show ROI Images** brings that manager's paired images forward. In an active image, press **+** or **−** to zoom; hold **Space** and drag to pan. For multiple selection, click the first row and Command-click additional rows.

Check that the outlines follow the intended cell structures and exclude unrelated cells and background. Use the raw grayscale views to inspect the target signal. Recheck the selected IDs after editing because numbering changes.

## 5. Correct only the regions that need changes

| What you see | Where to act and what follows |
|---|---|
| An unwanted cell | Select its row in any manager and click **Delete Selected ID(s)**. All three regions belonging to that cell are removed. The last remaining cell cannot be deleted; if no valid cells remain, use **Cancel Analysis**. |
| One cell represented by separate IDs | In the Soma manager, select at least two rows and click **Merge Selected Soma IDs**. The program checks whether the selected Soma boundaries form a connected proximity chain with gaps no greater than **1.0 µm**. You do not measure the gaps manually. A distance rejection leaves the ROIs unchanged; inspect all three regions after a successful merge. |
| Multiple cells in one whole-cell ROI | Select exactly one row in the Whole Cell manager and click **Split Selected Whole Cell**, when available. The program searches for one additional supported DAPI nucleus and recalculates two cells. Inspect the result; insufficient evidence can cause rejection. |
| A Soma ROI that is too small | Select exactly one row in the Soma manager and click **Enlarge Selected Soma**, when available. The program recalculates the boundary from local evidence and updates the other regions with it. |
| Unwanted process areas | Open **Trim Processes** in the Processes manager and follow the draft-and-preview steps below. |

To trim processes:

1. In any Composite view, draw a freehand area around the unwanted region and release. You do not need to preselect a cell; a stroke can affect several cells.
2. Click **Preview Trim**. Inspect the affected IDs and the yellow area proposed for deletion. Formal ROI outlines have not changed yet. After this first preview, further drawing and draft changes update the preview automatically.
3. Use **Clear Cell Draft** to remove the selected affected ID's draft, or **Undo** to reverse a draft action. **Cancel** discards the entire open draft.
4. Wait for the current preview, then click **Confirm Trim**. The program removes the accepted pixels from Whole Cell and Processes together, leaving Soma unchanged. It blocks a stale preview, a draft that removes no process pixels, removal of all Processes from an affected cell, or a new disconnection of retained processes from Soma. Adjust the draft if it is rejected.

When **Cell Edit: ready** appears, processing has ended. Read the outcome message and inspect the regions; rejected attempts retain the prior ROIs. To stop Split or Enlarge processing, click **Cancel Cell Edit** and wait for cancellation to finish. To undo an applied edit, close Trim, wait for any processing to end, and click **Revert** in the Waiting window. Repeated clicks undo applied edits in reverse order; one confirmed Trim session is reverted as a whole. Trim's Undo operates only on its open draft.

## 6. Complete measurement and read the results

If you entered **Review ROIs**, finish any editing first. If Trim is open, apply a valid draft with **Confirm Trim** or close it with **Cancel**; use Cancel for an empty draft. Wait for processing or cancellation to finish, then click **Finish Review and Measure** in **Astrocyte ROI Reviewer-Waiting**.

After either **Measure Now** or **Finish Review and Measure**, Project Leap continues automatically. Wait for **Completed**, then click **View results**.

Open `IHC_2D_Fluorescence_Results.xlsx` and choose **Whole Cell**, **Soma**, or **Processes**. Each row contains one cell's measurements in that region. `Astrocyte_ID` matches the overlay label. `Mean` is the average pixel intensity in the ROI; `Median` is its middle pixel intensity. `Manual_Review_Used` records whether **Review ROIs** was entered. **Review Audit** records the edits applied and reverted during that workflow.

Measurements use the target channel's raw grayscale values, including background signal, in a two-dimensional maximum-intensity projection of the selected Z range. The target intensities receive no background subtraction or intensity normalization; display colors do not alter them. The workbook records the projection and Z range.

The program validates and saves these six files together:

| File | Contents |
|---|---|
| `IHC_2D_Whole_Astrocyte_Overlay.png` | Whole-cell outlines and IDs |
| `IHC_2D_Astrocyte_Soma_Overlay.png` | Soma outlines and IDs |
| `IHC_2D_Astrocyte_Processes_Overlay.png` | Process outlines and IDs |
| `IHC_2D_Analysis_Report.txt` | Analysis settings, selection details, and run report |
| `IHC_2D_Fluorescence_Results.xlsx` | Measurements for all three regions and the edit record |
| `IHC_2D_ROI_Archive.zip` | Final label masks, ImageJ ROIs, calibration, identities, and their correspondence to the measurements |

Keep the result folder and original TIFF for future reference. Extract the ROI archive to inspect the final label masks or ImageJ ROIs. Completed fluorescence tasks currently have no one-click route back into ROI editing.

## 7. Handle a stopped run or maintain the module

After cancellation or failure, return to the module, resolve the reported error, and click **Start analysis** to begin a new run. For the current task, scroll to **Run details** below the analysis controls to inspect its status and run log. For a saved task, open **Jobs & Results**, select the task, and click **Expand** beside **Run details** to read its saved summary. A `z_step_limit` message means the eGFP route cannot fit its required depth windows in this stack. Use an acquisition with sufficient depth and retain its recorded slice spacing. `z_selection_ambiguous` means the highest position scores tie and no position was selected.

If Project Leap closes while its background process remains alive, reopening the application can reconnect to that task. If the process has exited, the task may be marked interrupted; start a new run after addressing the cause.

When the application requests module repair, open **Modules & Maintenance**. Use the application's cleanup controls for managed temporary data. Deleting a task removes its managed data and results; export files you need before deleting it. External original images and exported copies remain outside that deletion.
