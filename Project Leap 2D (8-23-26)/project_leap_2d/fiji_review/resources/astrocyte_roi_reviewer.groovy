#@ String manifest

import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import ij.CompositeImage
import ij.IJ
import ij.ImagePlus
import ij.WindowManager
import ij.gui.Overlay
import ij.gui.Roi
import ij.gui.ShapeRoi
import ij.gui.TextRoi
import ij.io.FileSaver
import ij.measure.Measurements
import ij.measure.ResultsTable
import ij.plugin.RGBStackMerge
import ij.plugin.Duplicator
import ij.plugin.ZProjector
import ij.plugin.filter.Analyzer
import ij.plugin.filter.ThresholdToSelection
import ij.plugin.frame.RoiManager
import ij.process.ImageProcessor
import ij.process.LUT
import ij.process.ShortProcessor
import java.awt.BorderLayout
import java.awt.Button
import java.awt.Color
import java.awt.Dialog
import java.awt.EventQueue
import java.awt.Frame
import java.awt.Font
import java.awt.BasicStroke
import java.awt.GridLayout
import java.awt.Label
import java.awt.Panel
import java.awt.Rectangle
import java.awt.TextArea
import java.awt.event.ActionListener
import java.awt.event.ItemEvent
import java.awt.event.ItemListener
import java.awt.event.KeyAdapter
import java.awt.event.KeyEvent
import java.awt.event.WindowAdapter
import java.awt.event.WindowEvent
import java.awt.geom.AffineTransform
import java.awt.geom.Area
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.ArrayList
import java.util.LinkedHashMap
import java.util.LinkedHashSet
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

def manifestFile = new File(manifest)
def cfg = new JsonSlurper().parse(manifestFile)
def readyFile = new File(cfg.ready_marker as String)
def doneFile = new File(cfg.done_marker as String)
def errorFile = new File(cfg.error_marker as String)

try {
    def expected = cfg.expected_shape.collect { (it as Number).intValue() }
    int zStart = (cfg.z_start_1based as Number).intValue()
    int zEnd = (cfg.z_end_1based_inclusive as Number).intValue()
    def projectionMethod = (cfg.projection as String).toLowerCase()
    def fijiProjectionMethod = projectionMethod == "mean" ? "avg" : projectionMethod
    def projections = [:]

    cfg.channel_order.each { channelValue ->
        String channel = channelValue as String
        String imagePath = cfg.channels[channel] as String
        ImagePlus source = IJ.openImage(imagePath)
        if (source == null) {
            throw new IllegalStateException("Fiji could not open " + imagePath)
        }
        if (source.getNChannels() != 1 || source.getNFrames() != 1 ||
                source.getNSlices() != expected[0] ||
                source.getHeight() != expected[1] || source.getWidth() != expected[2]) {
            def observed = [
                source.getNSlices(), source.getHeight(), source.getWidth(),
                source.getNChannels(), source.getNFrames()
            ]
            source.close()
            throw new IllegalStateException(
                "Unexpected dimensions for " + channel + ": " + observed + "; expected ZYX=" + expected
            )
        }
        ImagePlus projected = ZProjector.run(source, fijiProjectionMethod, zStart, zEnd)
        source.changes = false
        source.close()
        if (projected == null) {
            throw new IllegalStateException("Z projection failed for " + channel)
        }
        projected.setTitle(channel + "_raw_" + projectionMethod.toUpperCase() + "_Z" + zStart + "-" + zEnd)
        def limits = cfg.display_ranges[channel]
        projected.setDisplayRange(
            (limits[0] as Number).doubleValue(),
            (limits[1] as Number).doubleValue()
        )
        projections[channel] = projected
    }

    String measurementChannel = cfg.measurement_channel as String
    def displayOrder = cfg.channel_order.collect { it as String }
    ImagePlus[] mergeInputs = displayOrder.collect { projections[it].duplicate() } as ImagePlus[]
    ImagePlus merged = RGBStackMerge.mergeChannels(mergeInputs, false)
    if (merged == null) {
        throw new IllegalStateException("Fiji channel merge failed")
    }
    CompositeImage composite = merged instanceof CompositeImage ?
        (CompositeImage) merged : new CompositeImage(merged, CompositeImage.COMPOSITE)
    composite.setMode(CompositeImage.COMPOSITE)
    composite.setTitle("IHC 2D Composite - Automatic Astrocyte ROIs")

    def colors = [:]
    colors["DAPI"] = Color.BLUE
    colors[measurementChannel] = Color.RED
    def structural = cfg.structural_channels.collect { it as String }
    if (structural.size() == 2) {
        colors["eGFP"] = Color.GREEN
        colors["GFAP"] = Color.YELLOW
    } else {
        colors[structural[0]] = Color.GREEN
    }
    displayOrder.eachWithIndex { channel, index ->
        composite.setPosition(index + 1, 1, 1)
        composite.setChannelLut(LUT.createLutFromColor(colors[channel] as Color))
        def limits = cfg.display_ranges[channel]
        composite.setDisplayRange(
            (limits[0] as Number).doubleValue(),
            (limits[1] as Number).doubleValue()
        )
    }
    composite.setActiveChannels("1" * displayOrder.size())
    composite.setPosition(1, 1, 1)
    composite.show()

    RoiManager roiManager = RoiManager.getRoiManager()
    roiManager.reset()
    roiManager.setVisible(false)
    int expectedRois = (cfg.roi_count as Number).intValue()
    boolean requireCompleteSomaIds = true
    Color wholeColor = new Color(255, 0, 255)
    Color somaColor = Color.CYAN
    Color processColor = Color.WHITE

    def validateAreaRoi = { Roi roi, String description ->
        def bounds = roi.getBounds()
        if (!roi.isArea() || bounds.width <= 0 || bounds.height <= 0) {
            throw new IllegalStateException(description + " is not a valid non-empty area ROI")
        }
    }
    def roiId = { Roi roi ->
        String name = roi.getName() ?: ""
        def matcher = name =~ /Astrocyte_(\d+)/
        if (!matcher.find()) {
            throw new IllegalStateException(
                "ROI name must retain its Astrocyte ID; found '" + name + "'"
            )
        }
        return Integer.parseInt(matcher.group(1))
    }
    def originalRoiId = { Roi roi ->
        String stored = roi.getProperty("IHC_ORIGINAL_ID")
        return stored == null ? roiId(roi) : Integer.parseInt(stored)
    }
    def cellUid = { Roi roi ->
        String stored = roi.getProperty("IHC_CELL_UID")
        return stored == null || stored.trim().isEmpty() ?
            String.format("cell-%06d", originalRoiId(roi)) : stored
    }
    def parentCellUid = { Roi roi ->
        String stored = roi.getProperty("IHC_PARENT_UID")
        return stored == null ? "" : stored
    }
    def ownerNucleusId = { Roi roi ->
        String stored = roi.getProperty("IHC_OWNER_NUCLEUS_ID")
        return stored == null ? "" : stored
    }
    def roiLineage = { Roi roi ->
        String stored = roi.getProperty("IHC_LINEAGE")
        if (stored == null || stored.trim().isEmpty()) {
            return [originalRoiId(roi) as Integer]
        }
        return stored.split(",").collect { Integer.parseInt(it.trim()) }.sort()
    }
    def loadLabelRois = { String path, String compartment, Color color, boolean allowMissing ->
        ImagePlus labelImage = IJ.openImage(path)
        if (labelImage == null) {
            throw new IllegalStateException("Fiji could not open ROI labels: " + path)
        }
        if (labelImage.getWidth() != expected[2] || labelImage.getHeight() != expected[1]) {
            labelImage.close()
            throw new IllegalStateException("ROI label dimensions do not match the projections")
        }
        def loaded = []
        def processor = labelImage.getProcessor()
        for (int label = 1; label <= expectedRois; label++) {
            processor.setThreshold(label, label, ImageProcessor.NO_LUT_UPDATE)
            Roi roi = new ThresholdToSelection().convert(processor)
            if (roi == null) {
                if (allowMissing) {
                    continue
                }
                labelImage.close()
                throw new IllegalStateException("Could not convert " + compartment + " ROI label " + label)
            }
            roi.setName(String.format("Astrocyte_%03d_%s", label, compartment))
            roi.setProperty("IHC_ORIGINAL_ID", Integer.toString(label))
            roi.setProperty("IHC_LINEAGE", Integer.toString(label))
            roi.setProperty("IHC_CELL_UID", String.format("cell-%06d", label))
            roi.setProperty("IHC_PARENT_UID", "")
            roi.setProperty("IHC_OWNER_NUCLEUS_ID", "")
            roi.setStrokeColor(color)
            roi.setStrokeWidth(2.0d)
            validateAreaRoi(roi, roi.getName())
            loaded.add(roi)
        }
        processor.resetThreshold()
        labelImage.changes = false
        labelImage.close()
        return loaded
    }
    def labelAnchors = new LinkedHashMap<Integer, double[]>()
    def highlightedOriginalIds = new LinkedHashSet<Integer>()
    AtomicBoolean highlightRefreshPending = new AtomicBoolean(false)
    Color highlightColor = new Color(255, 170, 0)
    Font roiLabelFont = new Font("SansSerif", Font.BOLD, 16)
    def exportOverlayFor = { rois, Color color ->
        Overlay overlay = new Overlay()
        overlay.drawLabels(false)
        overlay.drawNames(false)
        overlay.drawBackgrounds(false)
        overlay.selectable(false)
        overlay.setDraggable(false)
        rois.each { sourceRoi ->
            Roi displayRoi = (Roi) sourceRoi.clone()
            displayRoi.setStrokeColor(color)
            displayRoi.setStrokeWidth(2.0d)
            displayRoi.setFillColor(null)
            overlay.add(displayRoi)
            int currentId = roiId(sourceRoi)
            int originalId = originalRoiId(sourceRoi)
            double[] anchor = labelAnchors[originalId]
            if (anchor == null) {
                def bounds = sourceRoi.getBounds()
                anchor = [
                    bounds.x + bounds.width / 2.0d,
                    bounds.y + bounds.height / 2.0d
                ] as double[]
            }
            TextRoi labelRoi = new TextRoi(
                anchor[0],
                anchor[1] - 8.0d,
                Integer.toString(currentId),
                roiLabelFont
            )
            labelRoi.setJustification(TextRoi.CENTER)
            labelRoi.setAntialiased(true)
            labelRoi.setStrokeColor(Color.WHITE)
            labelRoi.setFillColor(new Color(0, 0, 0, 170))
            labelRoi.setName("IHC_Display_ID_" + currentId)
            overlay.add(labelRoi)
        }
        return overlay
    }
    def displayOverlayFor = { rois, Color color ->
        Overlay overlay = new Overlay()
        overlay.drawLabels(false)
        overlay.drawNames(false)
        overlay.drawBackgrounds(false)
        overlay.selectable(false)
        overlay.setDraggable(false)
        rois.each { sourceRoi ->
            Roi displayRoi = (Roi) sourceRoi.clone()
            displayRoi.setStrokeColor(color)
            displayRoi.setStrokeWidth(2.0d)
            displayRoi.setFillColor(null)
            overlay.add(displayRoi)
            int currentId = roiId(sourceRoi)
            int originalId = originalRoiId(sourceRoi)
            if (highlightedOriginalIds.contains(originalId)) {
                Roi highlightHalo = (Roi) sourceRoi.clone()
                highlightHalo.setStrokeColor(Color.BLACK)
                highlightHalo.setStrokeWidth(6.0d)
                highlightHalo.setFillColor(null)
                highlightHalo.setName("IHC_Selection_Halo_" + currentId)
                overlay.add(highlightHalo)
                Roi highlightOutline = (Roi) sourceRoi.clone()
                highlightOutline.setStrokeColor(highlightColor)
                highlightOutline.setStrokeWidth(3.0d)
                highlightOutline.setFillColor(null)
                highlightOutline.setName("IHC_Selection_" + currentId)
                overlay.add(highlightOutline)
            }
            double[] anchor = labelAnchors[originalId]
            if (anchor == null) {
                def bounds = sourceRoi.getBounds()
                anchor = [
                    bounds.x + bounds.width / 2.0d,
                    bounds.y + bounds.height / 2.0d
                ] as double[]
            }
            TextRoi labelRoi = new TextRoi(
                anchor[0],
                anchor[1] - 8.0d,
                Integer.toString(currentId),
                roiLabelFont
            )
            labelRoi.setJustification(TextRoi.CENTER)
            labelRoi.setAntialiased(true)
            labelRoi.setStrokeColor(Color.WHITE)
            labelRoi.setFillColor(new Color(0, 0, 0, 170))
            labelRoi.setName("IHC_Display_ID_" + currentId)
            overlay.add(labelRoi)
        }
        return overlay
    }
    def initialWhole = loadLabelRois(
        cfg.label_mask_paths["whole"] as String,
        "Whole",
        wholeColor,
        false
    )
    def initialSoma = loadLabelRois(
        cfg.label_mask_paths["soma"] as String,
        "Soma",
        somaColor,
        false
    )
    def initialProcesses = loadLabelRois(
        cfg.label_mask_paths["processes"] as String,
        "Processes",
        processColor,
        false
    )
    def recomputeLabelAnchors = { somaRois ->
        labelAnchors.clear()
        somaRois.each { Roi roi ->
            double[] centroid = roi.getContourCentroid()
            labelAnchors[originalRoiId(roi) as Integer] = centroid
        }
    }
    recomputeLabelAnchors(initialSoma)
    def wholeIds = initialWhole.collect { roiId(it) }.toSet()
    if (initialProcesses.collect { roiId(it) }.toSet() != wholeIds) {
        throw new IllegalStateException("Processes IDs do not match Whole Astrocyte IDs")
    }
    def initialSomaIds = initialSoma.collect { roiId(it) }.toSet()
    if (initialSomaIds != wholeIds) {
        throw new IllegalStateException("Soma IDs must match Whole Astrocyte IDs")
    }
    def roiSets = [
        whole: initialWhole,
        processes: initialProcesses,
        soma: initialSoma
    ]
    def roiColors = [
        whole: wholeColor,
        processes: processColor,
        soma: somaColor
    ]
    def displayNames = [
        whole: "Whole Astrocyte Cell",
        processes: "Astrocyte Processes",
        soma: "Astrocyte Soma"
    ]
    def displayPrefixes = [whole: "01", soma: "02", processes: "03"]

    def configureComposite = { ImagePlus source, String title, rois, Color color ->
        CompositeImage view = source instanceof CompositeImage ?
            (CompositeImage) source : new CompositeImage(source, CompositeImage.COMPOSITE)
        view.setMode(CompositeImage.COMPOSITE)
        displayOrder.eachWithIndex { channel, index ->
            view.setPosition(index + 1, 1, 1)
            view.setChannelLut(LUT.createLutFromColor(colors[channel] as Color))
            def limits = cfg.display_ranges[channel]
            view.setDisplayRange(
                (limits[0] as Number).doubleValue(),
                (limits[1] as Number).doubleValue()
            )
        }
        view.setActiveChannels("1" * displayOrder.size())
        view.setPosition(1, 1, 1)
        view.setTitle(title)
        view.killRoi()
        view.setOverlay(displayOverlayFor(rois, color))
        return view
    }

    def compositeViews = [:]
    ["whole", "soma", "processes"].eachWithIndex { key, index ->
        ImagePlus source = index == 0 ? composite : new Duplicator().run(composite)
        CompositeImage view = configureComposite(
            source,
            displayPrefixes[key] + " Composite - " + displayNames[key] + " ROI",
            roiSets[key],
            roiColors[key] as Color
        )
        if (index > 0 || view.getWindow() == null) {
            view.show()
        }
        view.updateAndDraw()
        compositeViews[key] = view
    }

    ImagePlus measurementProjection = projections[measurementChannel] as ImagePlus
    def rawViews = [:]
    ["whole", "soma", "processes"].eachWithIndex { key, index ->
        ImagePlus view = index == 0 ? measurementProjection : measurementProjection.duplicate()
        view.setTitle(
            displayPrefixes[key] + " " + measurementChannel + " Raw Grayscale - " +
            displayNames[key] + " ROI - " + projectionMethod.toUpperCase() +
            " Z" + zStart + "-" + zEnd
        )
        view.killRoi()
        view.setOverlay(displayOverlayFor(roiSets[key], roiColors[key] as Color))
        view.show()
        view.updateAndDraw()
        rawViews[key] = view
    }

    def atomicWriteText = { File target, String content ->
        File temporary = new File(
            target.getParentFile(),
            "." + target.getName() + "." + UUID.randomUUID().toString() + ".tmp"
        )
        temporary.setText(content, "UTF-8")
        try {
            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING
            )
        } catch (Throwable ignored) {
            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.REPLACE_EXISTING
            )
        }
    }
    def saveOverlay = { ImagePlus view, rois, Color color, String outputPath, String title ->
        Overlay displayOverlay = view.getOverlay()
        view.setOverlay(exportOverlayFor(rois, color))
        ImagePlus flattened = view.flatten()
        view.setOverlay(displayOverlay)
        flattened.setTitle(title)
        boolean saved = new FileSaver(flattened).saveAsPng(outputPath)
        flattened.changes = false
        flattened.close()
        if (!saved) {
            throw new IllegalStateException("Fiji could not save " + title)
        }
    }
    def deepCloneRoi = { Roi source ->
        Roi cloned = (Roi) source.clone()
        cloned.setProperty("IHC_ORIGINAL_ID", Integer.toString(originalRoiId(source)))
        cloned.setProperty("IHC_LINEAGE", roiLineage(source).join(","))
        cloned.setProperty("IHC_CELL_UID", cellUid(source))
        cloned.setProperty("IHC_PARENT_UID", parentCellUid(source))
        cloned.setProperty("IHC_OWNER_NUCLEUS_ID", ownerNucleusId(source))
        return cloned
    }
    def cloneRoiSets = { source ->
        def cloned = [:]
        ["whole", "processes", "soma"].each { key ->
            cloned[key] = source[key].collect { deepCloneRoi(it as Roi) }
        }
        return cloned
    }
    def originalWholeIds = new LinkedHashSet<Integer>(
        roiSets["whole"].collect { originalRoiId(it) as Integer }
    )
    def validateTripletRoiSets = { candidateSets ->
        def wholeOriginalIds = candidateSets["whole"].collect {
            originalRoiId(it) as Integer
        }
        if (wholeOriginalIds.size() != wholeOriginalIds.toSet().size()) {
            throw new IllegalStateException("Whole ROI set repeats an Original Astrocyte ID")
        }
        ["processes", "soma"].each { key ->
            def compartmentIds = candidateSets[key].collect {
                originalRoiId(it) as Integer
            }
            if (compartmentIds != wholeOriginalIds) {
                throw new IllegalStateException(
                    "Linked ROI records drifted before measurement: Whole=" +
                    wholeOriginalIds + ", " + key + "=" + compartmentIds
                )
            }
            def compartmentLineages = candidateSets[key].collect { roiLineage(it).join(",") }
            def wholeLineages = candidateSets["whole"].collect { roiLineage(it).join(",") }
            if (compartmentLineages != wholeLineages) {
                throw new IllegalStateException(
                    "Linked ROI lineages drifted: Whole=" + wholeLineages +
                    ", " + key + "=" + compartmentLineages
                )
            }
            def compartmentUids = candidateSets[key].collect { cellUid(it) }
            def wholeUids = candidateSets["whole"].collect { cellUid(it) }
            if (compartmentUids != wholeUids) {
                throw new IllegalStateException(
                    "Linked Cell UIDs drifted: Whole=" + wholeUids +
                    ", " + key + "=" + compartmentUids
                )
            }
        }
        def wholeUids = candidateSets["whole"].collect { cellUid(it) }
        if (wholeUids.size() != wholeUids.toSet().size()) {
            throw new IllegalStateException("Whole ROI set repeats a Cell UID")
        }
    }
    def renumberRoiSets = {
        def originToFinal = new LinkedHashMap<Integer, Integer>()
        def sortedWhole = roiSets["whole"].sort {
            left, right -> originalRoiId(left) <=> originalRoiId(right)
        }
        sortedWhole.eachWithIndex { roi, index ->
            originToFinal[originalRoiId(roi) as Integer] = index + 1
        }
        def suffixes = [whole: "Whole", processes: "Processes", soma: "Soma"]
        ["whole", "processes", "soma"].each { key ->
            roiSets[key] = roiSets[key].sort {
                left, right -> originalRoiId(left) <=> originalRoiId(right)
            }
            roiSets[key].each { Roi roi ->
                Integer finalId = originToFinal[originalRoiId(roi) as Integer]
                if (finalId == null) {
                    throw new IllegalStateException("A " + key + " ROI has no Whole Astrocyte ID")
                }
                roi.setName(String.format("Astrocyte_%03d_%s", finalId, suffixes[key]))
                roi.setStrokeColor(roiColors[key] as Color)
                roi.setStrokeWidth(2.0d)
                roi.setFillColor(null)
            }
        }
        validateTripletRoiSets(roiSets)
        return originToFinal
    }
    def refreshPersistentViews = {
        ["whole", "soma", "processes"].each { key ->
            ImagePlus compositeView = compositeViews[key] as ImagePlus
            ImagePlus rawView = rawViews[key] as ImagePlus
            if (compositeView.getWindow() == null) compositeView.show()
            if (rawView.getWindow() == null) rawView.show()
            compositeView.killRoi()
            rawView.killRoi()
            compositeView.setOverlay(displayOverlayFor(roiSets[key], roiColors[key] as Color))
            rawView.setOverlay(displayOverlayFor(roiSets[key], roiColors[key] as Color))
            compositeView.updateAndDraw()
            rawView.updateAndDraw()
        }
    }
    def refreshSelectionHighlights = {
        ["whole", "soma", "processes"].each { key ->
            ImagePlus compositeView = compositeViews[key] as ImagePlus
            ImagePlus rawView = rawViews[key] as ImagePlus
            compositeView.setOverlay(displayOverlayFor(roiSets[key], roiColors[key] as Color))
            rawView.setOverlay(displayOverlayFor(roiSets[key], roiColors[key] as Color))
        }
    }
    def managerLists = [:]
    def managerFrames = []
    def deleteButtons = []
    def mergeButtons = []
    def splitButtons = []
    def enlargeButtons = []
    def editActionButtons = []
    def undoStack = new ArrayList()
    def reviewAudit = new ArrayList()
    AtomicBoolean reviewActive = new AtomicBoolean(false)
    AtomicBoolean editBusy = new AtomicBoolean(false)
    AtomicReference<String> activeEditRequestId = new AtomicReference<String>(null)
    AtomicBoolean activeEditCancelRequested = new AtomicBoolean(false)
    int cellEditStateRevision = 0
    String cellEditStateToken = UUID.randomUUID().toString().replace("-", "")
    def cellEditConfig = cfg.cell_edit instanceof Map ? cfg.cell_edit : [:]
    def enabledPythonEdits = new LinkedHashSet<String>(
        (cellEditConfig.enabled_actions ?: []).collect { it.toString().toLowerCase() }
    )
    boolean splitEnabled = enabledPythonEdits.contains("split")
    boolean enlargeEnabled = enabledPythonEdits.contains("enlarge")
    Label cellEditStatus = new Label("Cell Edit: ready")
    Button cancelEditButton = new Button("Cancel Cell Edit")
    cancelEditButton.setEnabled(false)
    Button revertButton = new Button("Revert")
    revertButton.setEnabled(false)
    int revertedActions = 0
    def refreshManagerLists = {
        managerLists.each { key, java.awt.List listWidget ->
            listWidget.removeAll()
            roiSets[key].each { roi -> listWidget.add(roi.getName()) }
        }
    }
    def refreshReviewState = {
        highlightedOriginalIds.clear()
        recomputeLabelAnchors(roiSets["soma"])
        renumberRoiSets()
        refreshPersistentViews()
        refreshManagerLists()
        revertButton.setEnabled(!undoStack.isEmpty())
    }
    def sha256File = { File source ->
        MessageDigest digest = MessageDigest.getInstance("SHA-256")
        source.withInputStream { stream ->
            byte[] buffer = new byte[1024 * 1024]
            int count
            while ((count = stream.read(buffer)) > 0) {
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().collect {
            String.format("%02x", ((it as int) & 0xff))
        }.join()
    }
    def identityRecordsFor = { sourceSets ->
        return sourceSets["whole"].collect { Roi roi -> [
            label_id: roiId(roi),
            original_id: originalRoiId(roi),
            cell_uid: cellUid(roi),
            parent_uid: parentCellUid(roi),
            lineage: roiLineage(roi),
            owner_nucleus_id: ownerNucleusId(roi)
        ] }
    }
    def saveLabelMask = { List<Roi> rois, File output ->
        ShortProcessor processor = new ShortProcessor(expected[2], expected[1])
        processor.setValue(0.0d)
        processor.fill()
        rois.each { Roi roi ->
            processor.setValue(roiId(roi) as double)
            processor.fill(roi)
        }
        ImagePlus labelImage = new ImagePlus(output.getName(), processor)
        boolean saved = new FileSaver(labelImage).saveAsTiff(output.getAbsolutePath())
        labelImage.changes = false
        labelImage.close()
        if (!saved || !output.isFile()) {
            throw new IllegalStateException("Could not serialize Cell Edit ROI state")
        }
    }
    def saveCurrentLabelState = { String requestId ->
        File stateDirectory = new File(cellEditConfig.state_dir as String)
        def paths = [:]
        def hashes = [:]
        ["whole", "soma", "processes"].each { key ->
            File output = new File(stateDirectory, requestId + "_" + key + ".tif")
            saveLabelMask(roiSets[key] as List<Roi>, output)
            paths[key] = output.getAbsolutePath()
            hashes[key] = sha256File(output)
        }
        return [paths: paths, hashes: hashes]
    }
    def loadEditedLabelState = { response ->
        int responseCount = (response.roi_count as Number).intValue()
        def records = response.identity_records
        if (!(records instanceof List) || records.size() != responseCount) {
            throw new IllegalStateException("Cell Edit returned invalid identity records")
        }
        def byLabel = records.collectEntries { record ->
            [((record.label_id as Number).intValue()): record]
        }
        if (byLabel.keySet().toList().sort() != (1..responseCount).toList()) {
            throw new IllegalStateException("Cell Edit returned non-contiguous label IDs")
        }
        def suffixes = [whole: "Whole", soma: "Soma", processes: "Processes"]
        def loadedSets = [whole: [], soma: [], processes: []]
        ["whole", "soma", "processes"].each { key ->
            File maskFile = new File(response.label_mask_paths[key] as String)
            String expectedHash = response.label_mask_sha256[key] as String
            if (!maskFile.isFile() || sha256File(maskFile) != expectedHash) {
                throw new IllegalStateException("Cell Edit " + key + " response hash mismatch")
            }
            ImagePlus labelImage = IJ.openImage(maskFile.getAbsolutePath())
            if (labelImage == null ||
                    labelImage.getWidth() != expected[2] ||
                    labelImage.getHeight() != expected[1]) {
                if (labelImage != null) labelImage.close()
                throw new IllegalStateException("Cell Edit " + key + " mask dimensions changed")
            }
            def processor = labelImage.getProcessor()
            for (int label = 1; label <= responseCount; label++) {
                processor.setThreshold(label, label, ImageProcessor.NO_LUT_UPDATE)
                Roi roi = new ThresholdToSelection().convert(processor)
                if (roi == null) {
                    labelImage.close()
                    throw new IllegalStateException(
                        "Cell Edit returned an empty " + key + " label " + label
                    )
                }
                def record = byLabel[label]
                int originalId = (record.original_id as Number).intValue()
                def lineageValues = record.lineage instanceof List ?
                    record.lineage.collect { (it as Number).intValue() } :
                    record.lineage.toString().split(",").collect {
                        Integer.parseInt(it.trim())
                    }
                roi.setName(String.format(
                    "Astrocyte_%03d_%s", label, suffixes[key]
                ))
                roi.setProperty("IHC_ORIGINAL_ID", Integer.toString(originalId))
                roi.setProperty("IHC_LINEAGE", lineageValues.join(","))
                roi.setProperty("IHC_CELL_UID", record.cell_uid as String)
                roi.setProperty("IHC_PARENT_UID", (record.parent_uid ?: "") as String)
                roi.setProperty(
                    "IHC_OWNER_NUCLEUS_ID",
                    (record.owner_nucleus_id ?: "") as String
                )
                roi.setStrokeColor(roiColors[key] as Color)
                roi.setStrokeWidth(2.0d)
                roi.setFillColor(null)
                validateAreaRoi(roi, roi.getName())
                loadedSets[key].add(roi)
            }
            processor.resetThreshold()
            labelImage.changes = false
            labelImage.close()
        }
        validateTripletRoiSets(loadedSets)
        for (int index = 0; index < responseCount; index++) {
            ShapeRoi wholeShape = new ShapeRoi(loadedSets["whole"][index] as Roi)
            ShapeRoi somaShape = new ShapeRoi(loadedSets["soma"][index] as Roi)
            ShapeRoi processShape = new ShapeRoi(loadedSets["processes"][index] as Roi)
            def hasArea = { ShapeRoi shape ->
                Rectangle bounds = shape.getBounds()
                bounds.width > 0 && bounds.height > 0
            }
            if (hasArea(new ShapeRoi(somaShape).not(new ShapeRoi(wholeShape))) ||
                    hasArea(new ShapeRoi(processShape).not(new ShapeRoi(wholeShape))) ||
                    hasArea(new ShapeRoi(somaShape).and(new ShapeRoi(processShape)))) {
                throw new IllegalStateException(
                    "Cell Edit returned overlapping or out-of-Whole compartments"
                )
            }
            ShapeRoi partition = new ShapeRoi(somaShape).or(new ShapeRoi(processShape))
            if (hasArea(new ShapeRoi(wholeShape).not(new ShapeRoi(partition))) ||
                    hasArea(new ShapeRoi(partition).not(new ShapeRoi(wholeShape)))) {
                throw new IllegalStateException(
                    "Cell Edit Soma and Processes do not exactly partition Whole"
                )
            }
        }
        return loadedSets
    }
    def setCellEditBusy = { boolean busy, String statusText ->
        editBusy.set(busy)
        cellEditStatus.setText(statusText)
        editActionButtons.each { Button button -> button.setEnabled(!busy) }
        revertButton.setEnabled(!busy && !undoStack.isEmpty())
        cancelEditButton.setEnabled(busy)
    }
    def requestPythonCellEdit = { String requestedAction, Roi selectedRoi ->
        if (!reviewActive.get()) return
        if (!editBusy.compareAndSet(false, true)) {
            IJ.showMessage("IHC 2D Analysis", "Another Cell Edit is still running.")
            return
        }
        String requestId = UUID.randomUUID().toString().replace("-", "")
        int requestRevision = cellEditStateRevision
        String requestToken = cellEditStateToken
        activeEditRequestId.set(requestId)
        activeEditCancelRequested.set(false)
        setCellEditBusy(true, "Cell Edit: " + requestedAction + " is running...")
        Thread editThread = new Thread({
            try {
                def stateFiles = saveCurrentLabelState(requestId)
                def request = [
                    schema_version: 1,
                    request_id: requestId,
                    action: requestedAction,
                    state_revision: requestRevision,
                    state_token: requestToken,
                    selected_original_id: originalRoiId(selectedRoi),
                    selected_cell_uid: cellUid(selectedRoi),
                    label_mask_paths: stateFiles.paths,
                    label_mask_sha256: stateFiles.hashes,
                    identity_records: identityRecordsFor(roiSets),
                    requested_epoch_ms: System.currentTimeMillis()
                ]
                File requestFile = new File(
                    cellEditConfig.request_dir as String,
                    requestId + ".json"
                )
                atomicWriteText(
                    requestFile,
                    JsonOutput.prettyPrint(JsonOutput.toJson(request))
                )
                File responseFile = new File(
                    cellEditConfig.response_dir as String,
                    requestId + ".json"
                )
                long deadlineMs = System.currentTimeMillis() +
                    Math.round(
                        ((cellEditConfig.timeout_seconds ?: 45.0d) as Number).doubleValue() *
                        1000.0d
                    )
                while (!responseFile.isFile() && System.currentTimeMillis() < deadlineMs) {
                    Thread.sleep(100L)
                }
                if (!responseFile.isFile()) {
                    File cancelFile = new File(
                        cellEditConfig.cancel_dir as String,
                        requestId + ".cancel"
                    )
                    cancelFile.createNewFile()
                    throw new IllegalStateException(
                        "Cell Edit timed out; ROI state was not changed."
                    )
                }
                def response = new JsonSlurper().parse(responseFile)
                if (response.request_id != requestId ||
                        response.action != requestedAction ||
                        (response.state_revision as Number).intValue() != requestRevision ||
                        response.state_token != requestToken) {
                    throw new IllegalStateException(
                        "Stale Cell Edit response was rejected; ROI state was not changed."
                    )
                }
                String responseStatus = response.status as String
                if (responseStatus != "success") {
                    throw new IllegalStateException(
                        (response.reason ?: "Cell Edit was rejected; ROI state was not changed.") as String
                    )
                }
                def replacementSets = loadEditedLabelState(response)
                EventQueue.invokeAndWait {
                    if (!reviewActive.get() ||
                            cellEditStateRevision != requestRevision ||
                            cellEditStateToken != requestToken ||
                            activeEditCancelRequested.get()) {
                        throw new IllegalStateException(
                            "Stale or cancelled Cell Edit result was rejected."
                        )
                    }
                    def snapshot = [
                        roiSets: cloneRoiSets(roiSets),
                        operation: requestedAction,
                        source_ids: roiLineage(selectedRoi)
                    ]
                    undoStack.add(snapshot)
                    try {
                        ["whole", "soma", "processes"].each { key ->
                            roiSets[key] = replacementSets[key].collect {
                                deepCloneRoi(it as Roi)
                            }
                        }
                        validateTripletRoiSets(roiSets)
                        cellEditStateRevision++
                        cellEditStateToken = UUID.randomUUID().toString().replace("-", "")
                        refreshReviewState()
                        def editedResultRois = roiSets["whole"].findAll { Roi roi ->
                            cellUid(roi) == cellUid(selectedRoi) ||
                                parentCellUid(roi) == cellUid(selectedRoi)
                        }
                        reviewAudit.add([
                            sequence: reviewAudit.size() + 1,
                            action: requestedAction,
                            source_ids: snapshot.source_ids,
                            result_lineage: editedResultRois.collectMany {
                                roiLineage(it as Roi)
                            }.toSet().toList().sort(),
                            source_cell_uids: [cellUid(selectedRoi)],
                            result_cell_uids: editedResultRois.collect {
                                cellUid(it as Roi)
                            },
                            reverted: false
                        ])
                    } catch (Throwable commitError) {
                        undoStack.remove(undoStack.size() - 1)
                        ["whole", "soma", "processes"].each { key ->
                            roiSets[key] = snapshot.roiSets[key].collect {
                                deepCloneRoi(it as Roi)
                            }
                        }
                        refreshReviewState()
                        throw commitError
                    }
                    IJ.showStatus(
                        "IHC 2D: " + requestedAction + " committed across Whole, Soma, and Processes"
                    )
                }
            } catch (Throwable editError) {
                EventQueue.invokeLater {
                    IJ.showMessage(
                        "IHC 2D Analysis",
                        (editError.getMessage() ?: editError.toString()) as String
                    )
                }
            } finally {
                activeEditRequestId.compareAndSet(requestId, null)
                EventQueue.invokeLater {
                    setCellEditBusy(false, "Cell Edit: ready")
                }
            }
        } as Runnable, "project-leap-fiji-cell-edit-" + requestedAction)
        editThread.setDaemon(true)
        editThread.start()
    }
    cancelEditButton.addActionListener({ event ->
        String requestId = activeEditRequestId.get()
        if (requestId != null && activeEditCancelRequested.compareAndSet(false, true)) {
            try {
                File cancelFile = new File(
                    cellEditConfig.cancel_dir as String,
                    requestId + ".cancel"
                )
                cancelFile.createNewFile()
                cellEditStatus.setText("Cell Edit: cancellation requested...")
            } catch (Throwable cancelError) {
                IJ.handleException(cancelError)
            }
        }
    } as ActionListener)
    def deleteOriginalAstrocyteIds = { Set<Integer> originalIds ->
        if (!reviewActive.get() || originalIds.isEmpty()) return
        if (editBusy.get()) {
            IJ.showMessage("IHC 2D Analysis", "Wait for the active Cell Edit to finish.")
            return
        }
        if (originalIds.isEmpty()) {
            throw new IllegalStateException("The selected Astrocyte IDs are no longer available")
        }
        if (roiSets["whole"].size() - originalIds.size() < 1) {
            IJ.showMessage("IHC 2D Analysis", "At least one Whole Astrocyte ROI must remain.")
            return
        }
        def deletedCellUids = roiSets["whole"].findAll {
            originalIds.contains(originalRoiId(it))
        }.collect { cellUid(it as Roi) }
        def snapshot = [
            roiSets: cloneRoiSets(roiSets),
            operation: "delete",
            source_ids: originalIds.toList().sort(),
            source_cell_uids: deletedCellUids
        ]
        undoStack.add(snapshot)
        try {
            ["whole", "processes", "soma"].each { key ->
                roiSets[key] = roiSets[key].findAll {
                    !originalIds.contains(originalRoiId(it))
                }
            }
            refreshReviewState()
            cellEditStateRevision++
            cellEditStateToken = UUID.randomUUID().toString().replace("-", "")
        } catch (Throwable deletionError) {
            undoStack.remove(undoStack.size() - 1)
            ["whole", "processes", "soma"].each { key ->
                roiSets[key] = snapshot.roiSets[key].collect { deepCloneRoi(it as Roi) }
            }
            refreshReviewState()
            throw deletionError
        }
        IJ.showStatus(
            "IHC 2D: deleted linked Original Astrocyte IDs " +
            originalIds.toList().sort()
        )
        reviewAudit.add([
            sequence: reviewAudit.size() + 1,
            action: "delete",
            source_ids: originalIds.toList().sort(),
            result_lineage: [],
            source_cell_uids: deletedCellUids,
            result_cell_uids: [],
            reverted: false
        ])
    }
    def revertLastDeletion = {
        if (!reviewActive.get() || undoStack.isEmpty()) return
        if (editBusy.get()) {
            IJ.showMessage("IHC 2D Analysis", "Wait for the active Cell Edit to finish.")
            return
        }
        def restored = undoStack.remove(undoStack.size() - 1)
        ["whole", "processes", "soma"].each { key ->
            roiSets[key] = restored.roiSets[key].collect { deepCloneRoi(it as Roi) }
        }
        for (int index = reviewAudit.size() - 1; index >= 0; index--) {
            if (!(reviewAudit[index].reverted as boolean)) {
                reviewAudit[index].reverted = true
                break
            }
        }
        revertedActions++
        cellEditStateRevision++
        cellEditStateToken = UUID.randomUUID().toString().replace("-", "")
        refreshReviewState()
        IJ.showStatus("IHC 2D: reverted the last Cell Edit operation")
    }
    def selectedOriginalIds = { String key, java.awt.List listWidget ->
        Set<Integer> selected = new LinkedHashSet<Integer>()
        listWidget.getSelectedIndexes().each { index ->
            int selectedIndex = (index as Number).intValue()
            if (selectedIndex >= 0 && selectedIndex < roiSets[key].size()) {
                selected.add(originalRoiId(roiSets[key][selectedIndex]) as Integer)
            }
        }
        return selected
    }
    def syncSelectionAndHighlight = { String sourceKey, java.awt.List sourceList ->
        if (!reviewActive.get()) return
        Set<Integer> selected = selectedOriginalIds(sourceKey, sourceList)
        highlightedOriginalIds.clear()
        highlightedOriginalIds.addAll(selected)
        if (!highlightRefreshPending.compareAndSet(false, true)) return
        EventQueue.invokeLater {
            try {
                if (reviewActive.get()) refreshSelectionHighlights()
            } catch (Throwable highlightError) {
                IJ.handleException(highlightError)
            } finally {
                highlightRefreshPending.set(false)
            }
        }
    }
    def deleteSelection = { String key, java.awt.List listWidget ->
        Set<Integer> selected = selectedOriginalIds(key, listWidget)
        if (selected.isEmpty()) {
            IJ.showMessage("IHC 2D Analysis", "Select at least one Astrocyte ID first.")
        } else {
            deleteOriginalAstrocyteIds(selected)
        }
    }
    double reviewMergeMaxSomaGapUm =
        (cfg.review_merge_max_soma_gap_um as Number).doubleValue()
    double reviewPixelWidthUm = (cfg.pixel_width_um as Number).doubleValue()
    double reviewPixelHeightUm = (cfg.pixel_height_um as Number).doubleValue()
    if (reviewMergeMaxSomaGapUm <= 0.0d ||
            reviewPixelWidthUm <= 0.0d || reviewPixelHeightUm <= 0.0d) {
        throw new IllegalStateException("Review Merge requires positive physical calibration")
    }
    AffineTransform reviewPhysicalTransform = AffineTransform.getScaleInstance(
        reviewPixelWidthUm,
        reviewPixelHeightUm
    )
    def physicalAreaFor = { Roi roi ->
        def bounds = roi.getBounds()
        AffineTransform imagePositionTransform = AffineTransform.getTranslateInstance(
            bounds.x as double,
            bounds.y as double
        )
        def imagePositionShape = imagePositionTransform.createTransformedShape(
            new ShapeRoi(roi).getShape()
        )
        return new Area(
            reviewPhysicalTransform.createTransformedShape(imagePositionShape)
        )
    }
    def somaRoisAreNear = { Roi left, Roi right ->
        Area leftArea = physicalAreaFor(left) as Area
        Area rightArea = physicalAreaFor(right) as Area
        Area directIntersection = new Area(leftArea)
        directIntersection.intersect(rightArea)
        if (!directIntersection.isEmpty()) return true
        BasicStroke proximityStroke = new BasicStroke(
            (float) (2.0d * reviewMergeMaxSomaGapUm),
            BasicStroke.CAP_ROUND,
            BasicStroke.JOIN_ROUND
        )
        Area expandedLeft = new Area(leftArea)
        expandedLeft.add(new Area(proximityStroke.createStrokedShape(leftArea)))
        expandedLeft.intersect(rightArea)
        return !expandedLeft.isEmpty()
    }
    def selectedSomaGraphIsConnected = { List<Roi> somaRois ->
        int count = somaRois.size()
        if (count < 2) return false
        def adjacency = (0..<count).collect { new LinkedHashSet<Integer>() }
        for (int left = 0; left < count; left++) {
            for (int right = left + 1; right < count; right++) {
                if (somaRoisAreNear(somaRois[left], somaRois[right])) {
                    adjacency[left].add(right)
                    adjacency[right].add(left)
                }
            }
        }
        def visited = new LinkedHashSet<Integer>()
        def pending = new ArrayList<Integer>()
        pending.add(0)
        while (!pending.isEmpty()) {
            int current = pending.remove(pending.size() - 1)
            if (!visited.add(current)) continue
            adjacency[current].each { neighbor ->
                if (!visited.contains(neighbor as Integer)) {
                    pending.add(neighbor as Integer)
                }
            }
        }
        return visited.size() == count
    }
    def mergedRoi = { List<Roi> sources, String compartment, Color color, List<Integer> lineage ->
        if (sources.isEmpty()) {
            throw new IllegalStateException("Cannot merge an empty " + compartment + " ROI set")
        }
        ShapeRoi mergedShape = new ShapeRoi(sources[0])
        sources.drop(1).each { Roi source ->
            mergedShape = mergedShape.or(new ShapeRoi(source))
        }
        validateAreaRoi(mergedShape, "Merged " + compartment)
        int canonicalId = lineage.min()
        Roi canonicalSource = sources.min {
            left, right -> originalRoiId(left) <=> originalRoiId(right)
        } as Roi
        mergedShape.setProperty("IHC_ORIGINAL_ID", Integer.toString(canonicalId))
        mergedShape.setProperty("IHC_LINEAGE", lineage.join(","))
        mergedShape.setProperty("IHC_CELL_UID", cellUid(canonicalSource))
        mergedShape.setProperty("IHC_PARENT_UID", parentCellUid(canonicalSource))
        mergedShape.setProperty("IHC_OWNER_NUCLEUS_ID", ownerNucleusId(canonicalSource))
        mergedShape.setName(String.format("Astrocyte_%03d_%s", canonicalId, compartment))
        mergedShape.setStrokeColor(color)
        mergedShape.setStrokeWidth(2.0d)
        mergedShape.setFillColor(null)
        return mergedShape
    }
    def mergeOriginalAstrocyteIds = { Set<Integer> selectedIds ->
        if (!reviewActive.get()) return
        if (editBusy.get()) {
            IJ.showMessage("IHC 2D Analysis", "Wait for the active Cell Edit to finish.")
            return
        }
        if (selectedIds.size() < 2) {
            IJ.showMessage(
                "IHC 2D Analysis",
                "Select at least two nearby Soma IDs to merge."
            )
            return
        }
        def ids = selectedIds.toList().sort()
        def wholeSources = ids.collect { selectedId ->
            roiSets["whole"].find { originalRoiId(it) == selectedId }
        }
        def somaSources = ids.collect { selectedId ->
            roiSets["soma"].find { originalRoiId(it) == selectedId }
        }
        if (wholeSources.any { it == null } || somaSources.any { it == null }) {
            throw new IllegalStateException("The selected Astrocyte IDs are no longer available")
        }
        if (!selectedSomaGraphIsConnected(somaSources as List<Roi>)) {
            IJ.showMessage(
                "IHC 2D Analysis",
                "Merge rejected: the selected Soma ROIs do not form one proximity chain " +
                "within " + String.format("%.2f", reviewMergeMaxSomaGapUm) + " um."
            )
            return
        }
        def lineage = wholeSources.collectMany {
            roiLineage(it as Roi)
        }.toSet().toList().sort()
        def mergeSourceUids = wholeSources.collect { cellUid(it as Roi) }
        def snapshot = [
            roiSets: cloneRoiSets(roiSets),
            operation: "merge",
            source_ids: lineage,
            source_cell_uids: mergeSourceUids
        ]
        undoStack.add(snapshot)
        try {
            Roi wholeMerged = mergedRoi(
                wholeSources as List<Roi>,
                "Whole",
                wholeColor,
                lineage
            )
            ShapeRoi somaUnion = mergedRoi(
                somaSources as List<Roi>,
                "Soma",
                somaColor,
                lineage
            ) as ShapeRoi
            somaUnion = somaUnion.and(new ShapeRoi(wholeMerged))
            validateAreaRoi(somaUnion, "Merged Soma")
            int canonicalId = lineage.min()
            somaUnion.setProperty("IHC_ORIGINAL_ID", Integer.toString(canonicalId))
            somaUnion.setProperty("IHC_LINEAGE", lineage.join(","))
            somaUnion.setProperty("IHC_CELL_UID", cellUid(wholeMerged))
            somaUnion.setProperty("IHC_PARENT_UID", parentCellUid(wholeMerged))
            somaUnion.setProperty("IHC_OWNER_NUCLEUS_ID", ownerNucleusId(wholeMerged))
            somaUnion.setName(String.format("Astrocyte_%03d_Soma", canonicalId))
            somaUnion.setStrokeColor(somaColor)
            somaUnion.setStrokeWidth(2.0d)
            somaUnion.setFillColor(null)
            ShapeRoi processMerged = new ShapeRoi(wholeMerged)
            processMerged = processMerged.not(new ShapeRoi(somaUnion))
            validateAreaRoi(processMerged, "Merged Processes")
            processMerged.setProperty("IHC_ORIGINAL_ID", Integer.toString(canonicalId))
            processMerged.setProperty("IHC_LINEAGE", lineage.join(","))
            processMerged.setProperty("IHC_CELL_UID", cellUid(wholeMerged))
            processMerged.setProperty("IHC_PARENT_UID", parentCellUid(wholeMerged))
            processMerged.setProperty("IHC_OWNER_NUCLEUS_ID", ownerNucleusId(wholeMerged))
            processMerged.setName(String.format("Astrocyte_%03d_Processes", canonicalId))
            processMerged.setStrokeColor(processColor)
            processMerged.setStrokeWidth(2.0d)
            processMerged.setFillColor(null)
            def replacements = [whole: wholeMerged, soma: somaUnion, processes: processMerged]
            ["whole", "soma", "processes"].each { compartment ->
                roiSets[compartment] = roiSets[compartment].findAll {
                    !ids.contains(originalRoiId(it))
                }
                roiSets[compartment].add(replacements[compartment])
            }
            refreshReviewState()
            cellEditStateRevision++
            cellEditStateToken = UUID.randomUUID().toString().replace("-", "")
            reviewAudit.add([
                sequence: reviewAudit.size() + 1,
                action: "merge",
                source_ids: lineage,
                result_lineage: lineage,
                source_cell_uids: mergeSourceUids,
                result_cell_uids: [cellUid(wholeMerged)],
                reverted: false
            ])
            IJ.showStatus("IHC 2D: merged linked Soma proximity chain " + ids)
        } catch (Throwable mergeError) {
            undoStack.remove(undoStack.size() - 1)
            ["whole", "processes", "soma"].each { key ->
                roiSets[key] = snapshot.roiSets[key].collect { deepCloneRoi(it as Roi) }
            }
            refreshReviewState()
            throw mergeError
        }
    }
    def mergeSelection = { java.awt.List listWidget ->
        mergeOriginalAstrocyteIds(selectedOriginalIds("soma", listWidget))
    }
    def buildManagerFrame = { String key, int position ->
        Frame frame = new Frame(displayNames[key] + " ROI Manager")
        frame.setLayout(new BorderLayout(4, 4))
        boolean isSomaManager = key == "soma"
        boolean isWholeManager = key == "whole"
        frame.add(
            new Label(
                isSomaManager ?
                "Delete or Merge Soma IDs; Enlarge recalculates one selected Soma." :
                (isWholeManager ?
                    "Delete cascades across all compartments; Split recalculates one selected Whole Cell." :
                    "Read-only geometry. Delete cascades across all compartments.")
            ),
            BorderLayout.NORTH
        )
        java.awt.List listWidget = new java.awt.List(12, true)
        managerLists[key] = listWidget
        frame.add(listWidget, BorderLayout.CENTER)
        int controlCount = 2 +
            (isSomaManager ? 1 : 0) +
            (isSomaManager && enlargeEnabled ? 1 : 0) +
            (isWholeManager && splitEnabled ? 1 : 0)
        Panel controls = new Panel(new GridLayout(1, controlCount, 4, 0))
        Button showButton = new Button("Show ROI Images")
        Button deleteButton = new Button("Delete Selected ID(s)")
        deleteButtons.add(deleteButton)
        editActionButtons.add(deleteButton)
        showButton.addActionListener({ event ->
            ImagePlus compositeView = compositeViews[key] as ImagePlus
            ImagePlus rawView = rawViews[key] as ImagePlus
            if (compositeView.getWindow() == null) compositeView.show()
            if (rawView.getWindow() == null) rawView.show()
            compositeView.getWindow().toFront()
            rawView.getWindow().toFront()
        } as ActionListener)
        deleteButton.addActionListener({ event ->
            try {
                deleteSelection(key, listWidget)
            } catch (Throwable deletionError) {
                IJ.handleException(deletionError)
            }
        } as ActionListener)
        listWidget.addItemListener({ ItemEvent event ->
            try {
                syncSelectionAndHighlight(key, listWidget)
            } catch (Throwable selectionError) {
                IJ.handleException(selectionError)
            }
        } as ItemListener)
        listWidget.addKeyListener(new KeyAdapter() {
            void keyPressed(KeyEvent event) {
                if (event.getKeyCode() == KeyEvent.VK_DELETE ||
                        event.getKeyCode() == KeyEvent.VK_BACK_SPACE) {
                    event.consume()
                    try {
                        deleteSelection(key, listWidget)
                    } catch (Throwable deletionError) {
                        IJ.handleException(deletionError)
                    }
                }
            }
        })
        controls.add(showButton)
        controls.add(deleteButton)
        if (isSomaManager) {
            Button mergeButton = new Button("Merge Selected Soma IDs")
            mergeButtons.add(mergeButton)
            editActionButtons.add(mergeButton)
            mergeButton.addActionListener({ event ->
                try {
                    mergeSelection(listWidget)
                } catch (Throwable mergeError) {
                    IJ.handleException(mergeError)
                }
            } as ActionListener)
            controls.add(mergeButton)
            if (enlargeEnabled) {
                Button enlargeButton = new Button("Enlarge Selected Soma")
                enlargeButtons.add(enlargeButton)
                editActionButtons.add(enlargeButton)
                enlargeButton.addActionListener({ event ->
                    try {
                        Set<Integer> selected = selectedOriginalIds("soma", listWidget)
                        if (selected.size() != 1) {
                            IJ.showMessage(
                                "IHC 2D Analysis",
                                "Select exactly one Soma ID to enlarge."
                            )
                        } else {
                            int selectedId = selected.iterator().next()
                            Roi selectedRoi = roiSets["soma"].find {
                                originalRoiId(it) == selectedId
                            } as Roi
                            requestPythonCellEdit("enlarge", selectedRoi)
                        }
                    } catch (Throwable enlargeError) {
                        IJ.handleException(enlargeError)
                    }
                } as ActionListener)
                controls.add(enlargeButton)
            }
        }
        if (isWholeManager && splitEnabled) {
            Button splitButton = new Button("Split Selected Whole Cell")
            splitButtons.add(splitButton)
            editActionButtons.add(splitButton)
            splitButton.addActionListener({ event ->
                try {
                    Set<Integer> selected = selectedOriginalIds("whole", listWidget)
                    if (selected.size() != 1) {
                        IJ.showMessage(
                            "IHC 2D Analysis",
                            "Select exactly one Whole Cell ID to split."
                        )
                    } else {
                        int selectedId = selected.iterator().next()
                        Roi selectedRoi = roiSets["whole"].find {
                            originalRoiId(it) == selectedId
                        } as Roi
                        requestPythonCellEdit("split", selectedRoi)
                    }
                } catch (Throwable splitError) {
                    IJ.handleException(splitError)
                }
            } as ActionListener)
            controls.add(splitButton)
        }
        frame.add(controls, BorderLayout.SOUTH)
        frame.addWindowListener(new WindowAdapter() {
            void windowClosing(WindowEvent event) {
                event.getWindow().setVisible(false)
            }
        })
        frame.setSize(620, 310)
        frame.setLocation(30 + position * 445, 60)
        frame.setVisible(true)
        managerFrames.add(frame)
    }
    def chooseAction = {
        CountDownLatch choiceLatch = new CountDownLatch(1)
        AtomicReference<String> choice = new AtomicReference<String>("cancel")
        AtomicBoolean choiceClosed = new AtomicBoolean(false)
        EventQueue.invokeLater {
            Dialog dialog = new Dialog((Frame) null, "Astrocyte ROI Reviewer", true)
            dialog.setLayout(new BorderLayout(8, 8))
            TextArea message = new TextArea(
                "Six Astrocyte ROI Views Are Ready\n" +
                "3 Overlay Views + 3 Raw Grayscale Views (Whole Cell | Soma | Processes)\n" +
                "Review supports linked Delete/Merge/Revert and, when available, local Split/Enlarge.\n" +
                "Choose whether to review ROIs before raw grayscale fluorescence measurement.\n" +
                "Dingcheng Wang | Dr. Min Zhou Lab, The Ohio State University",
                7,
                108,
                TextArea.SCROLLBARS_VERTICAL_ONLY
            )
            message.setEditable(false)
            dialog.add(message, BorderLayout.CENTER)
            Panel buttons = new Panel(new GridLayout(1, 3, 6, 0))
            Button measureNow = new Button("Measure Now")
            Button review = new Button("Review ROIs")
            Button cancel = new Button("Cancel Analysis")
            def finishChoice = { String value ->
                if (choiceClosed.compareAndSet(false, true)) {
                    choice.set(value)
                    dialog.setVisible(false)
                    dialog.dispose()
                    choiceLatch.countDown()
                }
            }
            measureNow.addActionListener({ finishChoice("measure") } as ActionListener)
            review.addActionListener({ finishChoice("review") } as ActionListener)
            cancel.addActionListener({ finishChoice("cancel") } as ActionListener)
            buttons.add(measureNow)
            buttons.add(review)
            buttons.add(cancel)
            dialog.add(buttons, BorderLayout.SOUTH)
            dialog.addWindowListener(new WindowAdapter() {
                void windowClosing(WindowEvent event) { finishChoice("cancel") }
            })
            dialog.pack()
            dialog.setLocationRelativeTo(null)
            dialog.setVisible(true)
        }
        choiceLatch.await()
        return choice.get()
    }

    roiManager.reset()
    roiManager.setVisible(false)
    refreshPersistentViews()
    atomicWriteText(readyFile, JsonOutput.toJson([
        stage: "review_decision",
        roi_count: roiSets["whole"].size(),
        measurement_title: (rawViews["whole"] as ImagePlus).getTitle(),
        image_windows: 6,
        ready_epoch_ms: System.currentTimeMillis()
    ]))
    long reviewStartedNs = System.nanoTime()
    String action = Boolean.TRUE.equals(cfg.auto_continue) ? "measure" : chooseAction()
    boolean manualReviewUsed = action == "review"
    if (action == "review") {
        CountDownLatch reviewLatch = new CountDownLatch(1)
        AtomicReference<String> reviewOutcome = new AtomicReference<String>("cancel")
        AtomicBoolean reviewClosed = new AtomicBoolean(false)
        reviewActive.set(true)
        EventQueue.invokeAndWait {
            buildManagerFrame("whole", 0)
            buildManagerFrame("soma", 1)
            buildManagerFrame("processes", 2)
            refreshManagerLists()
        }
        Frame controlFrame = new Frame("Astrocyte ROI Reviewer-Waiting")
        EventQueue.invokeAndWait {
            controlFrame.setLayout(new BorderLayout(6, 6))
            TextArea reviewMessage = new TextArea(
                "You can select to delete specific Whole Cell, Soma or Processes in three different ROI Managers. Deleting any one also deletes the corresponding ROIs in the other two compartments. \n" +
                "Merge and Enlarge are available in the Soma Manager; Split is available in the Whole Cell Manager. Delete, Merge, Split and Enlarge always update Whole, Soma and Processes together. Revert steps backward through committed edits.",
                6,
                112,
                TextArea.SCROLLBARS_VERTICAL_ONLY
            )
            reviewMessage.setEditable(false)
            controlFrame.add(reviewMessage, BorderLayout.NORTH)
            Panel centerPanel = new Panel(new BorderLayout(4, 4))
            centerPanel.add(cellEditStatus, BorderLayout.NORTH)
            int reviewControlCount = (splitEnabled || enlargeEnabled) ? 4 : 3
            Panel controls = new Panel(new GridLayout(1, reviewControlCount, 6, 0))
            Button finishButton = new Button("Finish Review and Measure")
            Button cancelButton = new Button("Cancel Analysis")
            def completeReview = { String value ->
                if (editBusy.get()) {
                    IJ.showMessage(
                        "IHC 2D Analysis",
                        "Cancel or wait for the active Cell Edit before leaving review."
                    )
                    return
                }
                if (reviewClosed.compareAndSet(false, true)) {
                    reviewActive.set(false)
                    highlightedOriginalIds.clear()
                    refreshPersistentViews()
                    finishButton.setEnabled(false)
                    cancelButton.setEnabled(false)
                    revertButton.setEnabled(false)
                    deleteButtons.each { it.setEnabled(false) }
                    mergeButtons.each { it.setEnabled(false) }
                    splitButtons.each { it.setEnabled(false) }
                    enlargeButtons.each { it.setEnabled(false) }
                    cancelEditButton.setEnabled(false)
                    reviewOutcome.set(value)
                    reviewLatch.countDown()
                }
            }
            revertButton.addActionListener({ event ->
                try {
                    revertLastDeletion()
                } catch (Throwable revertError) {
                    IJ.handleException(revertError)
                }
            } as ActionListener)
            finishButton.addActionListener({ completeReview("finish") } as ActionListener)
            cancelButton.addActionListener({ completeReview("cancel") } as ActionListener)
            controls.add(finishButton)
            controls.add(cancelButton)
            controls.add(revertButton)
            if (splitEnabled || enlargeEnabled) controls.add(cancelEditButton)
            centerPanel.add(controls, BorderLayout.CENTER)
            controlFrame.add(centerPanel, BorderLayout.CENTER)
            controlFrame.addWindowListener(new WindowAdapter() {
                void windowClosing(WindowEvent event) { completeReview("cancel") }
            })
            controlFrame.setSize(1180, 235)
            controlFrame.setLocation(30, 390)
            controlFrame.setVisible(true)
        }
        reviewLatch.await()
        action = reviewOutcome.get() == "finish" ? "measure" : "cancel"
        EventQueue.invokeAndWait {
            controlFrame.dispose()
            managerFrames.each { it.dispose() }
        }
    }
    double reviewWaitSeconds = (System.nanoTime() - reviewStartedNs) / 1.0e9d
    if (action == "cancel") {
        atomicWriteText(doneFile, JsonOutput.prettyPrint(JsonOutput.toJson([
            cancelled: true,
            stage: "cancelled_before_measurement",
            manual_review_used: manualReviewUsed,
            review_wait_seconds: reviewWaitSeconds
        ])))
        return
    }

    validateTripletRoiSets(roiSets)
    def originalToFinal = renumberRoiSets()
    def finalWholeIds = roiSets["whole"].collect { roiId(it) }
    def finalProcessIds = roiSets["processes"].collect { roiId(it) }
    def finalSomaIds = roiSets["soma"].collect { roiId(it) }
    if (finalProcessIds != finalWholeIds) {
        throw new IllegalStateException("Final Processes IDs do not match Whole IDs")
    }
    if (finalSomaIds != finalWholeIds) {
        throw new IllegalStateException("Final Soma IDs do not match Whole IDs")
    }
    refreshPersistentViews()
    saveOverlay(
        compositeViews["whole"] as ImagePlus,
        roiSets["whole"],
        wholeColor,
        cfg.overlay_output_paths["whole"] as String,
        "IHC_2D_Whole_Astrocyte_Overlay"
    )
    saveOverlay(
        compositeViews["processes"] as ImagePlus,
        roiSets["processes"],
        processColor,
        cfg.overlay_output_paths["processes"] as String,
        "IHC_2D_Astrocyte_Processes_Overlay"
    )
    saveOverlay(
        compositeViews["soma"] as ImagePlus,
        roiSets["soma"],
        somaColor,
        cfg.overlay_output_paths["soma"] as String,
        "IHC_2D_Astrocyte_Soma_Overlay"
    )

    Analyzer.setMeasurements(
        Measurements.AREA | Measurements.MEAN | Measurements.MEDIAN |
        Measurements.MIN_MAX | Measurements.INTEGRATED_DENSITY | Measurements.LABELS
    )
    Analyzer.setPrecision(3)
    def measureRoiSet = { rois, String compartment, String resultTitle, ImagePlus rawView ->
        if (rois.isEmpty()) {
            throw new IllegalStateException("No " + compartment + " ROI remains for measurement")
        }
        roiManager.reset()
        roiManager.setVisible(false)
        rois.each { roiManager.addRoi((Roi) it.clone()) }
        if (rawView.getWindow() == null) rawView.show()
        WindowManager.setCurrentWindow(rawView.getWindow())
        ResultsTable work = new ResultsTable()
        Analyzer.setResultsTable(work)
        if (!rois.isEmpty()) {
            roiManager.runCommand("Select All")
            boolean measured = roiManager.runCommand(rawView, "Measure")
            if (!measured || work.size() != rois.size()) {
                throw new IllegalStateException(
                    compartment + " Fiji Measure returned " + work.size() +
                    " rows for " + rois.size() + " ROIs"
                )
            }
        }
        double areaSum = 0.0d
        double intDenSum = 0.0d
        double rawIntDenSum = 0.0d
        def rowData = []
        rois.eachWithIndex { Roi roi, int index ->
            int id = roiId(roi)
            int originalId = originalRoiId(roi)
            work.setValue("ROI_Index", index, index + 1)
            work.setValue("Astrocyte_ID", index, id)
            work.setValue("Original_Astrocyte_ID", index, originalId)
            String lineageText = roiLineage(roi).join(",")
            work.setValue("Source_Original_Astrocyte_IDs", index, lineageText)
            work.setValue("Cell_UID", index, cellUid(roi))
            work.setValue("Parent_Cell_UID", index, parentCellUid(roi))
            work.setValue("Owner_Nucleus_ID", index, ownerNucleusId(roi))
            work.setValue("Compartment", index, compartment)
            work.setValue("ROI_Name", index, roi.getName())
            areaSum += work.getValue("Area", index)
            intDenSum += work.getValue("IntDen", index)
            if (work.columnExists("RawIntDen")) rawIntDenSum += work.getValue("RawIntDen", index)
            rowData.add([
                Label: work.getLabel(index) ?: "",
                Area: work.getValue("Area", index),
                Mean: work.getValue("Mean", index),
                Median: work.getValue("Median", index),
                Min: work.getValue("Min", index),
                Max: work.getValue("Max", index),
                IntDen: work.getValue("IntDen", index),
                RawIntDen: work.columnExists("RawIntDen") ? work.getValue("RawIntDen", index) : null,
                ROI_Index: index + 1,
                Astrocyte_ID: id,
                Original_Astrocyte_ID: originalId,
                Source_Original_Astrocyte_IDs: lineageText,
                Cell_UID: cellUid(roi),
                Parent_Cell_UID: parentCellUid(roi),
                Owner_Nucleus_ID: ownerNucleusId(roi),
                Compartment: compartment,
                ROI_Name: roi.getName()
            ])
        }
        ResultsTable independent = (ResultsTable) work.clone()
        independent.setIsResultsTable(false)
        def defaultWindow = ResultsTable.getResultsWindow()
        if (defaultWindow != null) defaultWindow.close(false)
        def previousWindow = WindowManager.getWindow(resultTitle)
        if (previousWindow != null) previousWindow.dispose()
        independent.show(resultTitle)
        return [
            title: resultTitle,
            rows: independent.size(),
            astrocyte_ids: rois.collect { roiId(it) },
            original_astrocyte_ids: rois.collect { originalRoiId(it) },
            source_original_astrocyte_ids: rois.collect { roiLineage(it) },
            cell_uids: rois.collect { cellUid(it) },
            parent_cell_uids: rois.collect { parentCellUid(it) },
            owner_nucleus_ids: rois.collect { ownerNucleusId(it) },
            headings: independent.getHeadings().toList(),
            area_sum: areaSum,
            integrated_density_sum: intDenSum,
            raw_integrated_density_sum: rawIntDenSum,
            row_data: rowData
        ]
    }

    long measurementStartedNs = System.nanoTime()
    def wholeResults = measureRoiSet(
        roiSets["whole"], "Whole", "01 Whole Astrocyte Cell Results", rawViews["whole"] as ImagePlus
    )
    def processResults = measureRoiSet(
        roiSets["processes"], "Processes", "02 Astrocyte Processes Results", rawViews["processes"] as ImagePlus
    )
    def somaResults = measureRoiSet(
        roiSets["soma"], "Soma", "03 Astrocyte Soma Results", rawViews["soma"] as ImagePlus
    )
    double measurementSeconds = (System.nanoTime() - measurementStartedNs) / 1.0e9d
    roiManager.reset()
    roiManager.setVisible(false)
    def retainedOriginalIds = new LinkedHashSet<Integer>(
        roiSets["whole"].collectMany { roiLineage(it) }
    )
    def deletedOriginalIds = originalWholeIds.findAll { !retainedOriginalIds.contains(it) }.sort()
    def sourceToFinalIds = new LinkedHashMap<Integer, List<Integer>>()
    roiSets["whole"].each { Roi roi ->
        int finalId = roiId(roi)
        roiLineage(roi).each { sourceId ->
            int source = sourceId as Integer
            if (!sourceToFinalIds.containsKey(source)) {
                sourceToFinalIds[source] = new ArrayList<Integer>()
            }
            sourceToFinalIds[source].add(finalId)
        }
    }
    atomicWriteText(doneFile, JsonOutput.prettyPrint(JsonOutput.toJson([
        cancelled: false,
        roi_count: roiSets["whole"].size(),
        final_roi_ids: finalWholeIds,
        final_process_ids: finalProcessIds,
        final_soma_ids: finalSomaIds,
        final_cell_uids: roiSets["whole"].collect { cellUid(it) },
        original_to_final_id: originalToFinal.collectEntries { key, value ->
            [(Integer.toString(key as Integer)): value]
        },
        source_to_final_ids: sourceToFinalIds.collectEntries { key, value ->
            [(Integer.toString(key as Integer)): value.toSet().toList().sort()]
        },
        deleted_original_ids: deletedOriginalIds,
        manual_review_used: manualReviewUsed,
        reverted_actions: revertedActions,
        review_audit: reviewAudit,
        review_wait_seconds: reviewWaitSeconds,
        measurement_seconds: measurementSeconds,
        result_sets: [whole: wholeResults, processes: processResults, soma: somaResults],
        measurement_title: (rawViews["whole"] as ImagePlus).getTitle(),
        image_window_titles: ["whole", "soma", "processes"].collectMany { key -> [
            (compositeViews[key] as ImagePlus).getTitle(),
            (rawViews[key] as ImagePlus).getTitle()
        ] },
        overlay_paths: cfg.overlay_output_paths,
        partition_qc: [
            processes_defined_as_whole_minus_soma: true,
            soma_complementary_to_processes: true
        ]
    ])))
} catch (Throwable error) {
    def writer = new StringWriter()
    error.printStackTrace(new PrintWriter(writer))
    errorFile.text = writer.toString()
    IJ.handleException(error)
    throw error
}
return null
