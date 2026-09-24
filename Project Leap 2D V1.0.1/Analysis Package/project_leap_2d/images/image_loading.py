# This functional source module is assembled into one shared runtime.
from __future__ import annotations

def resolution_um_per_pixel(resolution: object, unit: object) -> float | None:
    if resolution is None or unit is None:
        return None
    try:
        if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            pixels_per_unit = float(resolution[0]) / float(resolution[1])
        else:
            pixels_per_unit = float(resolution)
        unit_code = int(unit)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if pixels_per_unit <= 0:
        return None
    if unit_code == 2:  # inch
        return 25400.0 / pixels_per_unit
    if unit_code == 3:  # centimeter
        return 10000.0 / pixels_per_unit
    return None

def imagej_axis_scale_um(imagej_metadata: dict, axis: str) -> tuple[float | None, str | None]:
    """Read a calibrated SCIFIO/ImageJ axis scale without assuming isotropic voxels."""

    def convert_to_um(value: object, unit: object) -> float | None:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(scale) or scale <= 0:
            return None
        normalized_unit = (
            str(unit)
            .replace("\\u00B5", "u")
            .replace("µ", "u")
            .replace("μ", "u")
            .strip()
            .lower()
        )
        factors = {
            "um": 1.0,
            "micron": 1.0,
            "microns": 1.0,
            "micrometer": 1.0,
            "micrometers": 1.0,
            "micrometre": 1.0,
            "micrometres": 1.0,
            "nm": 0.001,
            "mm": 1000.0,
        }
        factor = factors.get(normalized_unit)
        return None if factor is None else scale * factor

    axis = axis.upper()
    if axis == "Z":
        spacing_um = convert_to_um(
            imagej_metadata.get("spacing"),
            imagej_metadata.get("unit"),
        )
        if spacing_um is not None:
            return spacing_um, "ImageJ spacing/unit"

    axes = [value.strip().upper() for value in str(imagej_metadata.get("axes", "")).split(",")]
    scales = [value.strip() for value in str(imagej_metadata.get("scales", "")).split(",")]
    units = [value.strip() for value in str(imagej_metadata.get("units", "")).split(",")]
    if axis not in axes or len(scales) != len(axes):
        return None, None
    index = axes.index(axis)
    unit = units[index] if len(units) == len(axes) else str(imagej_metadata.get("unit", ""))
    scale_um = convert_to_um(scales[index], unit)
    if scale_um is None:
        return None, None
    return scale_um, "SCIFIO axes/scales/units"

def read_meta(path: Path) -> dict:
    with tf.TiffFile(str(path)) as tif:
        series = tif.series[0]
        page0 = tif.pages[0]
        tags = page0.tags
        imagej_metadata = tif.imagej_metadata or {}
        x_resolution = tags["XResolution"].value if "XResolution" in tags else None
        y_resolution = tags["YResolution"].value if "YResolution" in tags else None
        resolution_unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
        pixel_depth_um, pixel_depth_source = imagej_axis_scale_um(imagej_metadata, "Z")
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
            "shape": tuple(int(x) for x in series.shape),
            "axes": series.axes,
            "dtype": str(series.dtype),
            "pages": len(tif.pages),
            "image_width": int(tags["ImageWidth"].value) if "ImageWidth" in tags else None,
            "image_length": int(tags["ImageLength"].value) if "ImageLength" in tags else None,
            "bits_per_sample": tags["BitsPerSample"].value if "BitsPerSample" in tags else None,
            "x_resolution": str(x_resolution) if x_resolution is not None else None,
            "y_resolution": str(y_resolution) if y_resolution is not None else None,
            "resolution_unit": str(resolution_unit) if resolution_unit is not None else None,
            "pixel_width_um": resolution_um_per_pixel(x_resolution, resolution_unit),
            "pixel_height_um": resolution_um_per_pixel(y_resolution, resolution_unit),
            "pixel_depth_um": pixel_depth_um,
            "pixel_depth_source": pixel_depth_source,
            "imagej_metadata": {
                k: v
                for k, v in imagej_metadata.items()
                if k
                in {
                    "channels",
                    "slices",
                    "frames",
                    "unit",
                    "spacing",
                    "axes",
                    "scales",
                    "units",
                }
            },
        }

def load_stack(path: Path) -> np.ndarray:
    arr = tf.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(f"Expected ZYX stack for {path}, got {arr.shape}")
    return arr
