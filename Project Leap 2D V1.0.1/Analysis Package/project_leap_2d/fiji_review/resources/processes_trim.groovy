/**
 * In-memory, pixel-exact Processes trimming for one linked Cell Edit session.
 * These are helper classes only: loading this resource starts no work or UI.
 */
class ProcessesTrimSession implements java.io.Closeable {
    private final int width
    private final int height
    private final String sessionKey = java.util.UUID.randomUUID().toString()
    private final LinkedHashMap<Integer, ProcessesTrimCell> sourceCells = new LinkedHashMap<>()
    private LinkedHashMap<Integer, java.util.BitSet> drafts = new LinkedHashMap<>()
    private final List<Map> history = new ArrayList<>()
    private List<Map> strokes = new ArrayList<>()
    private List<ij.gui.Roi> displayCache = null
    private final Map<Integer, Map> previewCache = new LinkedHashMap<>()
    private final Object previewLock = new Object()
    private final java.util.concurrent.ExecutorService workers
    private long revision = 0L
    private volatile boolean closed = false

    ProcessesTrimSession(int width, int height, List<Map> cells) {
        if (width <= 0 || height <= 0 || ((long) width) * height > Integer.MAX_VALUE) {
            throw new IllegalArgumentException('Trim image dimensions are invalid')
        }
        this.width = width
        this.height = height
        if (cells == null || cells.isEmpty()) {
            throw new IllegalArgumentException('Trim requires linked cell ROI records')
        }
        cells.each { Map record ->
            if (!(record.id instanceof Number)) {
                throw new IllegalArgumentException('Trim requires a numeric cell ID')
            }
            int id = ((Number) record.id).intValue()
            if (sourceCells.containsKey(id)) {
                throw new IllegalArgumentException('Trim cell IDs must be unique')
            }
            sourceCells.put(id, new ProcessesTrimCell(id, width, height,
                (ij.gui.Roi) record.whole, (ij.gui.Roi) record.soma,
                (ij.gui.Roi) record.processes))
        }
        int count = Math.max(1, Math.min(sourceCells.size(), Runtime.runtime.availableProcessors()))
        def serial = new java.util.concurrent.atomic.AtomicInteger(0)
        java.util.concurrent.ThreadFactory factory = { Runnable work ->
            Thread thread = new Thread(work, 'Processes-Trim-' + serial.incrementAndGet())
            thread.daemon = true
            return thread
        } as java.util.concurrent.ThreadFactory
        workers = java.util.concurrent.Executors.newFixedThreadPool(count, factory)
    }

    /** Attribute a freehand area only; never compute residuals or connectivity here. */
    synchronized Set<Integer> addStroke(ij.gui.Roi stroke) {
        requireOpen()
        if (stroke == null || !stroke.isArea()) {
            throw new IllegalArgumentException('Trim requires a closed freehand area selection')
        }
        ij.gui.Roi captured = (ij.gui.Roi) stroke.clone()
        LinkedHashMap<Integer, java.util.BitSet> updated = new LinkedHashMap<>(drafts)
        LinkedHashSet<Integer> changed = new LinkedHashSet<>()
        sourceCells.each { Integer id, ProcessesTrimCell cell ->
            if (!captured.bounds.intersects(cell.bounds)) return
            java.util.BitSet contribution = ProcessesTrimKernel.rasterize(captured, cell.bounds)
            contribution = contribution.and(cell.processesMask)
            if (contribution.isEmpty()) return
            java.util.BitSet merged = drafts.containsKey(id) ?
                (java.util.BitSet) drafts.get(id).clone() : new java.util.BitSet()
            merged = merged.or(contribution)
            updated.put(id, merged)
            changed.add(id)
        }
        if (!changed.isEmpty()) {
            List<Map> updatedStrokes = new ArrayList<>(strokes)
            updatedStrokes.add([roi: captured, initialIds: new LinkedHashSet<>(changed), activeIds: new LinkedHashSet<>(changed)])
            installDraftChange(updated, updatedStrokes, changed)
        }
        return immutableIds(changed)
    }

    /** Clear this cell's contribution, including its part of a multi-cell stroke. */
    synchronized Set<Integer> clearCellDraft(int id) {
        requireOpen()
        if (!sourceCells.containsKey(id)) throw new IllegalArgumentException('Unknown Trim cell ID: ' + id)
        if (!drafts.containsKey(id)) return immutableIds([])
        LinkedHashMap<Integer, java.util.BitSet> updated = new LinkedHashMap<>(drafts)
        updated.remove(id)
        Set<Integer> changed = new LinkedHashSet<>([id])
        List<Map> updatedStrokes = []
        strokes.each { Map stroke ->
            Set<Integer> active = new LinkedHashSet<>((Set<Integer>) stroke.activeIds)
            active.remove(id)
            if (!active.isEmpty()) updatedStrokes.add([roi: stroke.roi, initialIds: stroke.initialIds, activeIds: active])
        }
        installDraftChange(updated, updatedStrokes, changed)
        return immutableIds(changed)
    }

    /** Undo the last effective stroke or Clear Cell Draft operation. */
    synchronized Set<Integer> undo() {
        requireOpen()
        if (history.isEmpty()) return immutableIds([])
        Map previous = history.remove(history.size() - 1)
        Set<Integer> changed = (Set<Integer>) previous.changedIds
        drafts = new LinkedHashMap<>((Map<Integer, java.util.BitSet>) previous.drafts)
        strokes = new ArrayList<>((List<Map>) previous.strokes)
        revision++
        displayCache = null
        return immutableIds(changed)
    }

    synchronized List<Integer> affectedIds() {
        requireOpen()
        return Collections.unmodifiableList(orderedDraftIds(drafts))
    }

    /** Original handdrawn contours; no residual or actual-deletion contour is computed. */
    synchronized List<ij.gui.Roi> draftRois() {
        requireOpen()
        if (displayCache == null) {
            displayCache = []
            strokes.each { Map stroke ->
                ij.gui.Roi contour = (ij.gui.Roi) ((ij.gui.Roi) stroke.roi).clone()
                Set<Integer> cleared = new LinkedHashSet<>((Set<Integer>) stroke.initialIds)
                cleared.removeAll((Set<Integer>) stroke.activeIds)
                if (!cleared.isEmpty()) {
                    ij.gui.ShapeRoi remaining = new ij.gui.ShapeRoi(contour)
                    cleared.each { Integer id ->
                        remaining = remaining.not(new ij.gui.ShapeRoi(sourceCells.get(id).whole))
                    }
                    contour = remaining
                }
                List<Integer> active = new ArrayList<>((Set<Integer>) stroke.activeIds)
                contour.setName('Trim draft cells ' + active.join(','))
                contour.setProperty('IHC_TRIM_CELL_IDS', active.join(','))
                if (active.size() == 1) contour.setProperty('IHC_TRIM_CELL_ID', active[0].toString())
                displayCache.add(contour)
            }
        }
        return Collections.unmodifiableList(displayCache.collect { ij.gui.Roi roi -> (ij.gui.Roi) roi.clone() })
    }

    /** Immutable public map; its private per-cell masks are copied into the snapshot. */
    synchronized Map snapshot() {
        requireOpen()
        return new ProcessesTrimSnapshot(sessionKey, revision, orderedDraftIds(drafts), drafts)
    }

    /**
     * Call off the AWT event thread. Cells execute on a bounded worker pool.
     * Every response includes all currently affected cells; computedIds identifies
     * only the cells actually recomputed for this immutable snapshot.
     */
    Map preview(Map snapshot) {
        if (java.awt.EventQueue.isDispatchThread()) {
            throw new IllegalStateException('Trim preview must run outside the UI event thread')
        }
        if (!(snapshot instanceof ProcessesTrimSnapshot) || !snapshot.belongsTo(sessionKey)) {
            throw new IllegalArgumentException('Trim preview snapshot belongs to another session')
        }
        synchronized (previewLock) {
            requireOpen()
            Map<Integer, java.util.BitSet> captured = snapshot.copyDrafts()
            List<Integer> ids = (List<Integer>) snapshot.get('affectedIds')
            Map<Integer, java.util.concurrent.Future<Map>> futures = new LinkedHashMap<>()
            List<Integer> computed = []
            ids.each { Integer id ->
                Map cached = previewCache.get(id)
                if (cached == null || !captured.get(id).equals(cached.draft)) {
                    ProcessesTrimCell cell = sourceCells.get(id)
                    java.util.BitSet deletion = captured.get(id)
                    futures.put(id, workers.submit({ -> computeCell(cell, deletion) } as java.util.concurrent.Callable<Map>))
                    computed.add(id)
                }
            }
            Map<Integer, Map> replacements = new LinkedHashMap<>()
            try {
                futures.each { Integer id, java.util.concurrent.Future<Map> future ->
                    replacements.put(id, [draft: captured.get(id).clone(), result: future.get()])
                }
                requireOpen()
            } catch (Throwable error) {
                futures.values().each { it.cancel(true) }
                if (error instanceof InterruptedException) Thread.currentThread().interrupt()
                throw error
            }
            previewCache.putAll(replacements)
            previewCache.keySet().retainAll(ids)
            Map<Integer, Map> cells = new LinkedHashMap<>()
            Map<Integer, String> errors = new LinkedHashMap<>()
            ids.each { Integer id ->
                Map result = cloneResult((Map) previewCache.get(id).result)
                cells.put(id, result)
                if (result.error != null) errors.put(id, (String) result.error)
            }
            return Collections.unmodifiableMap([
                revision: snapshot.get('revision'), valid: errors.isEmpty(),
                cells: Collections.unmodifiableMap(cells), errors: Collections.unmodifiableMap(errors),
                computedIds: Collections.unmodifiableList(computed)
            ])
        }
    }

    /** Cheap commit check: exact current subtraction, unchanged Soma, and partition. */
    synchronized void validateReplacement(int id, ij.gui.Roi whole, ij.gui.Roi soma, ij.gui.Roi processes) {
        requireOpen()
        ProcessesTrimCell cell = sourceCells.get(id)
        if (cell == null) throw new IllegalArgumentException('Unknown Trim cell ID: ' + id)
        [whole, soma, processes].each { ij.gui.Roi roi ->
            if (roi == null || !roi.isArea() || !cell.bounds.contains(roi.bounds)) {
                throw new IllegalArgumentException('Trim replacement exceeds its source cell bounds')
            }
        }
        java.util.BitSet w = ProcessesTrimKernel.rasterize(whole, cell.bounds)
        java.util.BitSet s = ProcessesTrimKernel.rasterize(soma, cell.bounds)
        java.util.BitSet p = ProcessesTrimKernel.rasterize(processes, cell.bounds)
        java.util.BitSet expectedW = (java.util.BitSet) cell.wholeMask.clone()
        java.util.BitSet expectedP = (java.util.BitSet) cell.processesMask.clone()
        java.util.BitSet deletion = drafts.get(id)
        if (deletion != null) {
            expectedW.andNot(deletion)
            expectedP.andNot(deletion)
        }
        if (p.isEmpty() || !w.equals(expectedW) || !p.equals(expectedP) || !s.equals(cell.somaMask) ||
                !ProcessesTrimKernel.validPartition(w, s, p)) {
            throw new IllegalArgumentException('Trim replacement does not match the current draft and unchanged Soma')
        }
    }

    @Override
    synchronized void close() {
        if (closed) return
        closed = true
        workers.shutdownNow().each { Runnable queued ->
            if (queued instanceof java.util.concurrent.Future) ((java.util.concurrent.Future) queued).cancel(true)
        }
        drafts.clear()
        history.clear()
        displayCache = null
        strokes.clear()
    }

    private void requireOpen() {
        if (closed) throw new IllegalStateException('Trim session is closed')
    }

    private void installDraftChange(LinkedHashMap<Integer, java.util.BitSet> updated, List<Map> updatedStrokes, Set<Integer> changed) {
        if (changed.isEmpty()) return
        // Draft masks and stroke records are copy-on-write, preserving each Undo step.
        history.add([drafts: new LinkedHashMap<>(drafts), strokes: new ArrayList<>(strokes), changedIds: new LinkedHashSet<>(changed)])
        drafts = updated
        strokes = updatedStrokes
        revision++
        displayCache = null
    }

    private List<Integer> orderedDraftIds(Map<Integer, java.util.BitSet> state) {
        return sourceCells.keySet().findAll { state.containsKey(it) } as List<Integer>
    }

    private static Set<Integer> immutableIds(Collection<Integer> ids) {
        return Collections.unmodifiableSet(new LinkedHashSet<Integer>(ids))
    }

    private static Map cloneResult(Map result) {
        return Collections.unmodifiableMap([
            whole: result.whole == null ? null : ((ij.gui.Roi) result.whole).clone(),
            processes: result.processes == null ? null : ((ij.gui.Roi) result.processes).clone(),
            deleted: result.deleted == null ? null : ((ij.gui.Roi) result.deleted).clone(),
            error: result.error
        ])
    }

    private static Map computeCell(ProcessesTrimCell cell, java.util.BitSet draft) {
        java.util.BitSet deletion = (java.util.BitSet) draft.clone()
        deletion = deletion.and(cell.processesMask)
        java.util.BitSet p = (java.util.BitSet) cell.processesMask.clone()
        java.util.BitSet w = (java.util.BitSet) cell.wholeMask.clone()
        p.andNot(deletion)
        w.andNot(deletion)
        String error = null
        if (p.isEmpty()) {
            error = 'empty_processes'
        } else {
            if (cell.baseReachable == null) {
                cell.baseReachable = ProcessesTrimKernel.reachable(cell.wholeMask, cell.somaMask, cell.bounds.@width, cell.bounds.@height)
            }
            java.util.BitSet required = (java.util.BitSet) cell.baseReachable.clone()
            required = required.and(p)
            required.andNot(ProcessesTrimKernel.reachable(w, cell.somaMask, cell.bounds.@width, cell.bounds.@height))
            if (!required.isEmpty()) error = 'new_disconnection'
        }
        if (!ProcessesTrimKernel.validPartition(w, cell.somaMask, p)) {
            throw new IllegalStateException('Trim produced an invalid compartment partition')
        }
        return [
            whole: ProcessesTrimKernel.toRoi(w, cell.bounds, cell.whole),
            processes: ProcessesTrimKernel.toRoi(p, cell.bounds, cell.processes),
            deleted: ProcessesTrimKernel.toRoi(deletion, cell.bounds, null),
            error: error
        ]
    }
}

/** A snapshot never exposes its mutable BitSets or its owning session. */
@groovy.transform.CompileStatic
class ProcessesTrimSnapshot extends java.util.AbstractMap<String, Object> {
    private final String owner
    private final Map<Integer, java.util.BitSet> masks = new LinkedHashMap<Integer, java.util.BitSet>()
    private final Map<String, Object> publicValues

    ProcessesTrimSnapshot(String owner, long revision, List<Integer> ids, Map<Integer, java.util.BitSet> drafts) {
        this.owner = owner
        for (Integer id : ids) masks.put(id, (java.util.BitSet) drafts.get(id).clone())
        Map<String, Object> values = new LinkedHashMap<String, Object>()
        values.put('revision', Long.valueOf(revision))
        values.put('affectedIds', Collections.unmodifiableList(new ArrayList<Integer>(ids)))
        publicValues = Collections.unmodifiableMap(values)
    }

    boolean belongsTo(String key) { return owner.equals(key) }

    Map<Integer, java.util.BitSet> copyDrafts() {
        Map<Integer, java.util.BitSet> copy = new LinkedHashMap<Integer, java.util.BitSet>()
        for (Map.Entry<Integer, java.util.BitSet> entry : masks.entrySet()) {
            copy.put(entry.key, (java.util.BitSet) entry.value.clone())
        }
        return copy
    }

    @Override
    Set<Map.Entry<String, Object>> entrySet() { return publicValues.entrySet() }
}

class ProcessesTrimCell {
    final int id
    final java.awt.Rectangle bounds
    final ij.gui.Roi whole
    final ij.gui.Roi soma
    final ij.gui.Roi processes
    final java.util.BitSet wholeMask
    final java.util.BitSet somaMask
    final java.util.BitSet processesMask
    java.util.BitSet baseReachable = null

    ProcessesTrimCell(int id, int width, int height, ij.gui.Roi whole, ij.gui.Roi soma, ij.gui.Roi processes) {
        this.id = id
        java.awt.Rectangle imageBounds = new java.awt.Rectangle(0, 0, width, height)
        [whole, soma, processes].each { ij.gui.Roi roi ->
            if (roi == null || !roi.isArea() || roi.bounds.isEmpty() || !imageBounds.contains(roi.bounds)) {
                throw new IllegalArgumentException('Trim source ROIs must be non-empty areas inside the image')
            }
        }
        this.whole = (ij.gui.Roi) whole.clone()
        this.soma = (ij.gui.Roi) soma.clone()
        this.processes = (ij.gui.Roi) processes.clone()
        bounds = whole.bounds.union(soma.bounds).union(processes.bounds)
        wholeMask = ProcessesTrimKernel.rasterize(this.whole, bounds)
        somaMask = ProcessesTrimKernel.rasterize(this.soma, bounds)
        processesMask = ProcessesTrimKernel.rasterize(this.processes, bounds)
        if (wholeMask.isEmpty() || somaMask.isEmpty() || processesMask.isEmpty() ||
                !ProcessesTrimKernel.validPartition(wholeMask, somaMask, processesMask)) {
            throw new IllegalArgumentException('Trim requires Whole = Soma union Processes without overlap')
        }
    }
}

/** Compiled pixel kernels; all connectivity and residual conversion run in workers. */
@groovy.transform.CompileStatic
class ProcessesTrimKernel {
    static java.util.BitSet rasterize(ij.gui.Roi roi, java.awt.Rectangle bounds) {
        java.util.BitSet mask = new java.util.BitSet(bounds.@width * bounds.@height)
        java.awt.Rectangle roiBounds = roi.getBounds()
        java.awt.Rectangle clip = bounds.intersection(roiBounds)
        if (clip.isEmpty()) return mask
        // ImageJ supplies the same pixel mask used for area measurement. Rasterize
        // once instead of testing a complex polygon separately for every pixel.
        ij.process.ImageProcessor roiMask = roi.getMask()
        for (int y = clip.@y; y < clip.@y + clip.@height; y++) {
            int targetRow = (y - bounds.@y) * bounds.@width
            if (roiMask == null) {
                // ImageJ represents an ordinary rectangular ROI by a null mask.
                mask.set(targetRow + clip.@x - bounds.@x,
                    targetRow + clip.@x + clip.@width - bounds.@x)
            } else {
                int maskRow = y - roiBounds.@y
                for (int x = clip.@x; x < clip.@x + clip.@width; x++) {
                    if (roiMask.get(x - roiBounds.@x, maskRow) != 0) {
                        mask.set(targetRow + x - bounds.@x)
                    }
                }
            }
        }
        return mask
    }

    static boolean validPartition(java.util.BitSet whole, java.util.BitSet soma, java.util.BitSet processes) {
        // Groovy's BitSet and/or operators return new masks; retain the return values.
        java.util.BitSet overlap = soma.and(processes)
        if (!overlap.isEmpty()) return false
        java.util.BitSet union = soma.or(processes)
        return union.equals(whole)
    }

    /** Same-cell Soma reachability with the analysis route's 8-neighbor convention. */
    static java.util.BitSet reachable(java.util.BitSet whole, java.util.BitSet soma, int width, int height) {
        java.util.BitSet seen = (java.util.BitSet) soma.clone()
        seen = seen.and(whole)
        int[] queue = new int[whole.cardinality()]
        int start = 0
        int end = 0
        for (int pixel = seen.nextSetBit(0); pixel >= 0; pixel = seen.nextSetBit(pixel + 1)) {
            queue[end++] = pixel
        }
        while (start < end) {
            if ((start & 4095) == 0 && Thread.currentThread().isInterrupted()) {
                throw new java.util.concurrent.CancellationException('Trim preview cancelled')
            }
            int pixel = queue[start++]
            int x = pixel % width
            int y = Math.floorDiv(pixel, width)
            for (int dy = -1; dy <= 1; dy++) {
                int ny = y + dy
                if (ny < 0 || ny >= height) continue
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx
                    if (nx < 0 || nx >= width || (dx == 0 && dy == 0)) continue
                    int next = ny * width + nx
                    if (whole.get(next) && !seen.get(next)) {
                        seen.set(next)
                        queue[end++] = next
                    }
                }
            }
        }
        return seen
    }

    static ij.gui.Roi toRoi(java.util.BitSet mask, java.awt.Rectangle bounds, ij.gui.Roi metadata) {
        if (mask.isEmpty()) return null
        byte[] pixels = new byte[bounds.@width * bounds.@height]
        for (int pixel = mask.nextSetBit(0); pixel >= 0; pixel = mask.nextSetBit(pixel + 1)) {
            pixels[pixel] = (byte) 255
        }
        ij.process.ByteProcessor processor = new ij.process.ByteProcessor(bounds.@width, bounds.@height, pixels, null)
        processor.setThreshold(255, 255, ij.process.ImageProcessor.NO_LUT_UPDATE)
        ij.gui.Roi roi = new ij.plugin.filter.ThresholdToSelection().convert(processor)
        if (roi == null) throw new IllegalStateException('Trim could not represent a non-empty pixel mask')
        java.awt.Rectangle local = roi.getBounds()
        roi.setLocation(local.@x + bounds.@x, local.@y + bounds.@y)
        if (metadata != null) {
            roi.setName(metadata.getName())
            if (metadata.getProperties() != null) roi.setProperties(metadata.getProperties())
            roi.setStrokeColor(metadata.getStrokeColor())
            roi.setFillColor(metadata.getFillColor())
            roi.setStrokeWidth(metadata.getStrokeWidth())
            if (metadata.hasHyperStackPosition()) {
                roi.setPosition(metadata.getCPosition(), metadata.getZPosition(), metadata.getTPosition())
            } else {
                roi.setPosition(metadata.getPosition())
            }
        }
        return roi
    }
}
