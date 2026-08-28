"""
raster_align.radiometric_correction

Radiometric calibration for three sensor families:

  Resourcesat-2A (LISS-3 / LISS-4)
  ──────────────────────────────────
  - Locate BAND_META.txt next to a raster.
  - Read the ENTIRE BAND_META.txt (every key/value pair).
  - Compute MaxDN = (2^BitsPerPixel) - 1.
  - Identify band from filename (single-band) or BandNumbers (multi-band).
  - Apply linear scaling: L = Lmin + (Lmax - Lmin) * (DN / MaxDN).
  - Compute Day-of-Year, look up Earth-Sun distance, read SunElevation.
  - Apply: rho = (pi * L * d^2) / (E0 * cos(theta_z)).

  Sentinel-2 (L1C JP2 only -- TOA reflectance)
  ───────────────────────────
  - Detect sensor by .jp2 extension or SAFE directory structure.
  - Navigate three directories above the selected JP2 to the .SAFE folder.
  - Locate MTD_MSIL1C.xml. L2A products (MTD_MSIL2A.xml, BOA/atmospherically
    corrected) are explicitly rejected with an error -- this pipeline
    computes TOA reflectance only, never BOA.
  - Extract Quantification Value and per-band Additive Offset (L1C TOA
    field names).
  - Apply: Reflectance = (DN + Additive_Offset) / Quantification_Value.

  Landsat (Collection 1 / Collection 2)
  ───────────────────────────────────────
  - Detect by locating a *_MTL.txt file in the same folder as the image.
  - Read REFLECTANCE_MULT_BAND_x, REFLECTANCE_ADD_BAND_x, SUN_ELEVATION.
  - Apply: rho_prime = Mult * DN + Add.
  - Apply sun-angle correction: rho = rho_prime / sin(radians(SUN_ELEVATION)).

Architecture
────────────
Every sensor exposes the same interface through these top-level functions:

  detect_sensor(info)              -> sensor string
  locate_metadata(sensor, info)    -> path to metadata file
  load_metadata(sensor, path)      -> calibration dict
  compute_common_reflectance()     -> dispatcher (sensor-agnostic); the
                                       sole entry point called by pipeline.py
  compute_reflectance_for_array_auto() -> pixel-level computation

Adding a new satellite means updating detect_sensor / locate_metadata /
load_metadata and adding one calculate_<sensor>() function.  Everything
else -- pipeline, GUI, plots -- stays unchanged.

EarthSunDistance.csv must live in the project root directory (the same
folder as main.py) and contain two columns: DOY and EarthSunDistance,
with a header row.
"""

import csv
import glob
import math
import os
import re
import xml.etree.ElementTree as ET
from os.path import (join, isfile, splitext, basename, dirname,
                     abspath, isdir)

import numpy as np

from .resources import resource_path

BAND_META_FILENAME = "BAND_META.txt"
EARTH_SUN_CSV_FILENAME = "EarthSunDistance.csv"
_PROJECT_ROOT = dirname(dirname(abspath(__file__)))
REQUIRED_FIELDS = ("bitsperpixel",)

_S2_META_NAME_L1C = "MTD_MSIL1C.xml"
_S2_META_NAME_L2A = "MTD_MSIL2A.xml"  # recognised only to give a clear error

_ESUN = {
    "L3": {
        "B2": 181.4309,
        "B3": 156.1578,
        "B4": 108.1718,
        "B5":  24.3011,
    },
    "L4FX": {
        "B2": 181.9947,
        "B3": 157.3955,
        "B4": 109.6400,
    },
}

_MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}



def detect_sensor(info):
    """
    Determine which radiometric correction workflow applies to the raster
    described by *info* (as returned by io_utils.read_and_prepare).

    Detection rules (evaluated in order):
      1. If the file extension is .jp2 -> "SENTINEL2".
      2. If a .SAFE directory is found above the image -> "SENTINEL2".
      3. If BAND_META.txt exists next to the image -> "RESOURCESAT2A".
      4. If a *_MTL.txt file exists next to the image -> "LANDSAT".
      5. Raises ValueError (no sensor detected).

    Returns
    -------
    str
        "SENTINEL2", "RESOURCESAT2A", or "LANDSAT".
    """
    filepath = info.get("filepath", "")
    ext = splitext(filepath)[1].lower()

    if ext == ".jp2":
        return "SENTINEL2"

    if _find_safe_dir(filepath) is not None:
        return "SENTINEL2"

    band_meta_path = join(info["image_dir"], BAND_META_FILENAME)
    if isfile(band_meta_path):
        return "RESOURCESAT2A"

    if _find_landsat_mtl(info["image_dir"]) is not None:
        return "LANDSAT"

    raise ValueError(
        f"[Radiometric] Cannot detect sensor for '{info['filename']}'. "
        f"No .jp2 extension, no .SAFE parent, no BAND_META.txt, and no *_MTL.txt "
        f"found in '{info['image_dir']}'. "
        f"Supported sensors: Resourcesat-2A (BAND_META.txt), "
        f"Sentinel-2 (.jp2 / .SAFE), Landsat (*_MTL.txt)."
    )



def locate_band_meta(info, filename=BAND_META_FILENAME):
    """
    Locate band_meta.txt in the same directory as the raster described by
    `info` (as returned by io_utils.read_and_prepare(), which now carries
    "image_dir").
    """
    meta_path = join(info["image_dir"], filename)
    if not isfile(meta_path):
        raise FileNotFoundError(
            f"[Radiometric] {filename} not found next to {info['filename']} "
            f"(expected at {meta_path})."
        )
    return meta_path


def _parse_band_meta_raw(meta_path):
    """
    Parse EVERY key/value pair out of band_meta.txt into a raw dict,
    keyed by lowercased field name, values kept as raw strings.

    Tolerant of "KEY = value", "KEY: value", and "KEY value" formats, one
    entry per line, case-insensitive keys, with blank lines and '#'
    comment lines ignored.
    """
    raw = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            key, val = None, None
            for sep in ("=", ":"):
                if sep in line:
                    key, _, val = line.partition(sep)
                    break
            if key is None:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, val = parts

            if key is None:
                continue

            raw[key.strip().lower()] = val.strip()

    return raw


def validate_required_fields(meta_raw, meta_path, required=REQUIRED_FIELDS):
    """Raise if any required field is missing from a parsed band_meta dict."""
    missing = [k for k in required if k not in meta_raw]
    if missing:
        raise ValueError(
            f"[Radiometric] {meta_path} is missing required key(s): "
            f"{', '.join(missing)}"
        )


def read_band_metadata(meta_path):
    """
    Open band_meta.txt and read it in FULL: every key/value pair it
    contains is kept in the returned metadata dictionary (lowercased
    keys), not just LMAX/LMIN/BPP.

    On top of the raw key/value pairs, this also adds a couple of
    derived/typed entries so downstream code doesn't have to re-parse
    strings:

        metadata["bitsperpixel"]  -> float(BitsPerPixel)
        metadata["max_dn"]        -> (2 ** BitsPerPixel) - 1
        metadata["bandnumbers"]   -> BandNumbers as a string, e.g. "2345"

    Everything else from the file (B2_Lmin, B2_Lmax, B3_Lmin, B3_Lmax,
    sensor name, acquisition date, ...) stays available under its own
    lowercased key, e.g. metadata["b2_lmin"], metadata["b2_lmax"].
    """
    raw = _parse_band_meta_raw(meta_path)
    validate_required_fields(raw, meta_path)

    metadata = dict(raw)

    try:
        metadata["bitsperpixel"] = float(raw["bitsperpixel"])
    except ValueError as e:
        raise ValueError(
            f"[Radiometric] Could not parse numeric field in {meta_path}: {e}"
        )

    metadata["max_dn"] = calculate_max_dn(metadata["bitsperpixel"])

    if "bandnumbers" in raw:
        metadata["bandnumbers"] = str(raw["bandnumbers"]).strip()

    return metadata


def calculate_max_dn(bpp):
    """MaxDN = (2^BPP) - 1"""
    return (2 ** bpp) - 1


def _normalize_band_name(band_name):
    """
    Normalize a band identifier to the "B<N>" form used as a metadata
    key prefix, e.g. "4" -> "B4", "b4" -> "B4", "B4" -> "B4".
    """
    band_name = str(band_name).strip().upper()
    if not band_name.startswith("B"):
        band_name = "B" + band_name
    return band_name


def get_band_lmin_lmax(metadata, band_name, meta_path):
    """
    Dynamically look up (Lmin, Lmax) for `band_name` out of the metadata
    dictionary.

    RS2A uses keys like  b2_lmin / b2_lmax  (band_name e.g. "B2").
    Cartosat uses keys like  b1lmin / b1lmax  (band_name e.g. "B1", no
    underscore).  Detection: try the RS2A key first; if absent, fall back
    to the Cartosat key format.  Returns (lmin, lmax, sensor_family) where
    sensor_family is "RESOURCESAT2A" or "CARTOSAT".
    """
    band_key = _normalize_band_name(band_name)

    # --- RS2A style: b<N>_lmin / b<N>_lmax ---
    rs2a_lmin_key = f"{band_key.lower()}_lmin"
    rs2a_lmax_key = f"{band_key.lower()}_lmax"
    if rs2a_lmin_key in metadata:
        try:
            return float(metadata[rs2a_lmin_key]), float(metadata[rs2a_lmax_key]), "RESOURCESAT2A"
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"[Radiometric] Could not parse RS2A lmin/lmax from {meta_path}: {e}"
            )

    # --- Cartosat style: b<N>lmin / b<N>lmax (no underscore) ---
    carto_lmin_key = f"{band_key.lower()}lmin"
    carto_lmax_key = f"{band_key.lower()}lmax"
    if carto_lmin_key in metadata:
        try:
            return float(metadata[carto_lmin_key]), float(metadata[carto_lmax_key]), "CARTOSAT"
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"[Radiometric] Could not parse Cartosat lmin/lmax from {meta_path}: {e}"
            )

    raise ValueError(
        f"[Radiometric] {meta_path} has neither RS2A keys ({rs2a_lmin_key}/{rs2a_lmax_key}) "
        f"nor Cartosat keys ({carto_lmin_key}/{carto_lmax_key}) for band {band_key}. "
        f"Cannot determine sensor family or read calibration data."
    )


def get_band_calibration_full(info):
    """
    Locate + read a raster's band_meta.txt, dynamically resolve which
    band it corresponds to, and return the full metadata dictionary and
    resolved sensor identifier.  Used by pipeline.py so it has everything
    it needs for reflectance without a second trip to band_meta.txt.

      * single-band raster (info["band_count"] == 1): band name is
        parsed from the filename, e.g. "BAND4.tif" -> "B4".
      * multi-band / FCC raster (info["band_count"] > 1): band name is
        taken from BandNumbers' first entry, e.g. "2345" -> "B2"
        (corresponding to the raster's first band, index 0).

    No band name is ever hardcoded -- both the metadata keys and the
    filename pattern are derived dynamically.

    Returns
    -------
    (lmin, lmax, max_dn, band_name, meta_path, metadata, sensor)
    """
    meta_path = locate_band_meta(info)
    metadata = read_band_metadata(meta_path)

    # Detect sensor family by probing lmin keys before resolving band name.
    # RS2A BAND_META uses "b2_lmin"; Cartosat uses "b1lmin" (no underscore).
    # Pick a candidate band name to probe with, then re-resolve properly.
    is_cartosat = (
        not any(k for k in metadata if k.endswith("_lmin"))
        and any(k for k in metadata if k.endswith("lmin") and not k.endswith("_lmin"))
    )

    band_count = info.get("band_count", 1)
    if band_count > 1:
        band_numbers = metadata.get("bandnumbers")
        if not band_numbers:
            raise ValueError(
                f"[Radiometric] {meta_path} has no 'BandNumbers' entry, "
                f"required for multi-band/FCC images (band_count={band_count})."
            )
        first_band_digit = str(band_numbers).strip()[0]
        band_name = _normalize_band_name(first_band_digit)
    else:
        filename = basename(info.get("filepath", info.get("filename", "")))
        stem = splitext(filename)[0]
        if is_cartosat:
            # Cartosat files are named e.g. "BAND1.tif", "BAND2.tif", etc.
            # Band number MUST be parsed from the filename -- silently
            # defaulting to B1 on a mismatch would misattribute another
            # band's DN to B1's Lmin/Lmax/SolarRadiance without any error.
            match = re.match(r"BAND(\d+)$", stem, re.IGNORECASE)
            if not match:
                raise ValueError(
                    f"[Radiometric] Cannot determine Cartosat band number from "
                    f"filename '{filename}'. Expected 'BAND<number>.tif' "
                    f"(e.g. 'BAND2.tif'). Refusing to silently assume B1, since "
                    f"that would apply the wrong band's calibration coefficients."
                )
            band_name = f"B{match.group(1)}"
        else:
            match = re.match(r"BAND(\d+)$", stem, re.IGNORECASE)
            if not match:
                raise ValueError(
                    f"[Radiometric] Filename '{filename}' does not match the "
                    f"expected 'BAND<number>.tif' format."
                )
            band_name = f"B{match.group(1)}"

    lmin, lmax, sensor_family = get_band_lmin_lmax(metadata, band_name, meta_path)
    sensor = get_sensor(metadata, meta_path, sensor_family)

    return lmin, lmax, metadata["max_dn"], band_name, meta_path, metadata, sensor


def compute_toa_radiance(arr, lmin, lmax, max_dn):
    """
    Standard linear DN -> TOA radiance scaling:

        radiance = LMIN + (LMAX - LMIN) * (DN / MaxDN)

    Returns a float32 array the same shape as `arr`.
    """
    dn = arr.astype(np.float32)
    return (lmin + (lmax - lmin) * (dn / max_dn)).astype(np.float32)


def compute_reflectance_for_array(dn_arr, lmin, lmax, max_dn,
                                   band_name, sensor,
                                   metadata, meta_path):
    """
    Full DN -> Radiance -> Reflectance pipeline for a single Resourcesat-2A
    array.

    Returns the TOA Reflectance array (float32). Never prints -- see
    compute_common_reflectance() for the self-contained, logging-free
    entry point that wraps this.
    """
    radiance = compute_toa_radiance(dn_arr, lmin, lmax, max_dn)

    if "dateofpass" not in metadata:
        raise ValueError(
            f"[Radiometric] 'DateOfPass' not found in {meta_path}. "
            f"This field is required for TOA reflectance calculation."
        )
    earth_sun_dist, *_ = get_earth_sun_distance(metadata["dateofpass"])

    solar_zenith_rad = get_solar_zenith_radians(metadata, meta_path)

    esun = get_esun(sensor, band_name, metadata=metadata, meta_path=meta_path)

    reflectance = compute_toa_reflectance(radiance, earth_sun_dist, esun, solar_zenith_rad)

    return reflectance



def _is_leap_year(year):
    """Return True if *year* is a leap year (proleptic Gregorian calendar)."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _parse_date_of_pass(date_str):
    """
    Parse DateOfPass in "DD-MON-YYYY" format (e.g. "29-MAR-2025").

    Returns (day: int, month: int, year: int).
    Raises ValueError with a descriptive message on any parse failure.
    """
    date_str = date_str.strip().upper()
    parts = date_str.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"[Radiometric] DateOfPass '{date_str}' is not in expected "
            f"DD-MON-YYYY format (e.g. '29-MAR-2025')."
        )
    day_str, mon_str, year_str = parts
    if mon_str not in _MONTH_ABBR:
        raise ValueError(
            f"[Radiometric] DateOfPass month '{mon_str}' is not a recognised "
            f"3-letter month abbreviation."
        )
    try:
        day = int(day_str)
        year = int(year_str)
    except ValueError:
        raise ValueError(
            f"[Radiometric] DateOfPass '{date_str}' contains non-integer "
            f"day or year fields."
        )
    return day, _MONTH_ABBR[mon_str], year


def _day_of_year(day, month, year):
    """
    Calculate the Day of Year (DOY, 1-based) for the given date.
    Accounts for leap years.
    """
    days_per_month = [31, 29 if _is_leap_year(year) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return sum(days_per_month[:month - 1]) + day


def _read_earth_sun_distance(csv_path, doy):
    """
    Read the Earth-Sun distance for *doy* (Day of Year) from a CSV file.

    The CSV must have a header row.  The first column is the DOY (integer
    or float) and the second column is the Earth-Sun distance in AU.
    Any leading/trailing whitespace in cell values is stripped.
    """
    if not isfile(csv_path):
        raise FileNotFoundError(
            f"[Radiometric] Earth-Sun distance CSV not found: {csv_path}. "
            f"'{EARTH_SUN_CSV_FILENAME}' must live in the project root directory "
            f"(the same folder as main.py) -- NOT next to the raster or its "
            f"BAND_META.txt. Place it at: {csv_path}"
        )
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                row_doy = int(float(row[0].strip()))
            except ValueError:
                continue
            if row_doy == doy:
                try:
                    return float(row[1].strip())
                except ValueError:
                    raise ValueError(
                        f"[Radiometric] Earth-Sun distance value in "
                        f"'{csv_path}' for DOY {doy} is not a valid number: "
                        f"'{row[1].strip()}'."
                    )
    raise ValueError(
        f"[Radiometric] DOY {doy} not found in Earth-Sun distance CSV: "
        f"'{csv_path}'."
    )


def get_earth_sun_distance(date_of_pass_str):
    """
    Full automatic Earth-Sun distance lookup from a DateOfPass string.

    Steps:
      1. Parse DateOfPass -> (day, month, year).
      2. Determine leap year.
      3. Calculate Day of Year.
      4. Read EarthSunDistance.csv from the project root directory.
      5. Return (distance_au, doy, is_leap, day, month, year).
    """
    day, month, year = _parse_date_of_pass(date_of_pass_str)
    is_leap = _is_leap_year(year)
    doy = _day_of_year(day, month, year)
    earth_sun_path = resource_path(EARTH_SUN_CSV_FILENAME)
    csv_path = earth_sun_path
    distance = _read_earth_sun_distance(csv_path, doy)
    return distance, doy, is_leap, day, month, year



def get_solar_zenith_radians(metadata, meta_path):
    """
    Read SunElevationAtCenter from *metadata* and return the Solar Zenith
    Angle in radians.

    theta_z (degrees) = 90 - SunElevationAtCenter
    theta_z (radians) = theta_z * pi / 180
    """
    key = "sunelevationatcenter"
    if key not in metadata:
        raise ValueError(
            f"[Radiometric] 'SunElevationAtCenter' not found in {meta_path}. "
            f"This field is required for TOA reflectance calculation."
        )
    try:
        sun_elev = float(metadata[key])
    except ValueError:
        raise ValueError(
            f"[Radiometric] Could not parse 'SunElevationAtCenter' as a "
            f"number in {meta_path}: '{metadata[key]}'."
        )
    zenith_deg = 90.0 - sun_elev
    zenith_rad = math.radians(zenith_deg)
    return zenith_rad



def get_sensor(metadata, meta_path, sensor_family=None):
    """
    Return the normalised sensor identifier.

    For RS2A (sensor_family == "RESOURCESAT2A") the Sensor field is read
    and must be "L3" or "L4FX" (validated against _ESUN).
    For Cartosat (sensor_family == "CARTOSAT") the Sensor field is read
    for informational purposes only (e.g. "P6-PAN-MX") and "CARTOSAT"
    is returned directly -- ESUN is not table-looked-up for Cartosat.
    If sensor_family is not supplied it is inferred the old way.
    """
    if sensor_family == "CARTOSAT":
        return "CARTOSAT"

    key = "sensor"
    if key not in metadata:
        raise ValueError(
            f"[Radiometric] 'Sensor' field not found in {meta_path}. "
            f"This field is required for TOA reflectance calculation."
        )
    sensor = metadata[key].strip().upper()
    if sensor not in _ESUN:
        supported = ", ".join(sorted(_ESUN))
        raise ValueError(
            f"[Radiometric] Sensor '{sensor}' in {meta_path} is not "
            f"supported for reflectance conversion. Supported sensors: "
            f"{supported}."
        )
    return sensor


def get_esun(sensor, band_name, metadata=None, meta_path=None):
    """
    Return the ESUN (E0) value for *sensor* and *band_name*.

    RS2A (sensor in _ESUN): looked up from the hardcoded _ESUN table.
    Cartosat (sensor == "CARTOSAT"): read from the B<N>SolarRadiance
    key in *metadata*, e.g. "B1SolarRadiance" for band B1.
    """
    band_key = _normalize_band_name(band_name)

    if sensor == "CARTOSAT":
        sat_rad_key = f"{band_key}solarradiance".lower()
        # metadata keys are already lowercased by _parse_band_meta_raw
        if metadata is None or sat_rad_key not in metadata:
            raise ValueError(
                f"[Radiometric] Cartosat ESUN: key '{band_key}SolarRadiance' "
                f"not found in {meta_path or 'metadata'}. "
                f"Ensure BAND_META.txt contains this field."
            )
        try:
            return float(metadata[sat_rad_key])
        except ValueError:
            raise ValueError(
                f"[Radiometric] Could not parse '{band_key}SolarRadiance' "
                f"as a number in {meta_path or 'metadata'}: "
                f"'{metadata[sat_rad_key]}'."
            )

    sensor_table = _ESUN.get(sensor, {})
    if band_key not in sensor_table:
        defined = ", ".join(sorted(sensor_table))
        raise ValueError(
            f"[Radiometric] No ESUN value defined for sensor '{sensor}' "
            f"band '{band_key}'. Defined bands for this sensor: {defined}."
        )
    return sensor_table[band_key]



def compute_toa_reflectance(radiance, earth_sun_distance, esun, solar_zenith_rad):
    """
    Compute TOA Reflectance from TOA Radiance:

        rho = (pi * L * d^2) / (E0 * cos(theta_z))
    """
    cos_theta = math.cos(solar_zenith_rad)
    denominator = esun * cos_theta
    if denominator == 0.0:
        raise ValueError(
            "[Radiometric] Cannot compute TOA Reflectance: denominator "
            "(ESUN x cos(theta_z)) is zero."
        )
    return ((math.pi * radiance.astype(np.float32) * (earth_sun_distance ** 2))
            / denominator).astype(np.float32)



def _find_safe_dir(filepath):
    """
    Walk up the directory tree from *filepath* and return the path of the
    first ancestor directory whose name ends with '.SAFE', or None.
    """
    d = dirname(abspath(filepath))
    for _ in range(6):
        if isdir(d) and d.upper().endswith(".SAFE"):
            return d
        parent = dirname(d)
        if parent == d:   # filesystem root
            break
        d = parent
    return None


def _locate_sentinel_metadata(filepath):
    """
    Locate the Sentinel-2 L1C XML metadata file (MTD_MSIL1C.xml) by
    navigating directory levels above *filepath* to reach the .SAFE folder.

    L1C ONLY. If the .SAFE folder instead contains MTD_MSIL2A.xml (an L2A/
    BOA product), this raises a clear error rather than silently falling
    back to it -- L2A's BOA_QUANTIFICATION_VALUE/BOA_ADD_OFFSET fields
    produce Bottom-of-Atmosphere reflectance, not TOA, and this pipeline
    computes TOA reflectance only.

    Returns
    -------
    str
        Absolute path to the found MTD_MSIL1C.xml file.
    """
    safe_dir = _find_safe_dir(filepath)
    if safe_dir is None:
        raise FileNotFoundError(
            f"[Radiometric] Could not locate a .SAFE directory by walking up "
            f"from '{filepath}'. Ensure the file is inside a standard "
            f"Sentinel-2 .SAFE product folder structure."
        )

    l1c_candidate = join(safe_dir, _S2_META_NAME_L1C)
    if isfile(l1c_candidate):
        return l1c_candidate

    l2a_candidate = join(safe_dir, _S2_META_NAME_L2A)
    if isfile(l2a_candidate):
        raise ValueError(
            f"[Radiometric] Found '{_S2_META_NAME_L2A}' at '{safe_dir}' -- this "
            f"is an L2A (Bottom-of-Atmosphere) product. This pipeline computes "
            f"TOA (Top-of-Atmosphere) reflectance only and does not support "
            f"L2A/BOA inputs. Supply the corresponding L1C product instead."
        )

    raise FileNotFoundError(
        f"[Radiometric] Found .SAFE directory at '{safe_dir}' but "
        f"'{_S2_META_NAME_L1C}' does not exist there."
    )


def read_sentinel_metadata(filepath):
    """
    Parse a Sentinel-2 L1C XML metadata file (MTD_MSIL1C.xml) and extract
    the TOA Quantification Value and per-band TOA Additive Offset for all
    bands.

    L1C ONLY -- see _locate_sentinel_metadata(). Only the L1C field names
    (QUANTIFICATION_VALUE, RADIO_ADD_OFFSET) are read; the L2A/BOA field
    names (BOA_QUANTIFICATION_VALUE, BOA_ADD_OFFSET) are intentionally not
    looked up here, since this pipeline computes TOA reflectance only.

    Returns
    -------
    tuple of (meta_path: str, quant_value: float, band_offsets: dict)
    """
    meta_path = _locate_sentinel_metadata(filepath)
    tree = ET.parse(meta_path)
    root = tree.getroot()

    def _find_text(tag):
        for elem in root.iter():
            if elem.tag.split("}")[-1] == tag and elem.text:
                return elem.text.strip()
        return None

    quant_text = _find_text("QUANTIFICATION_VALUE")
    if quant_text is None:
        raise ValueError(
            f"[Radiometric] Could not find QUANTIFICATION_VALUE in '{meta_path}'. "
            f"Ensure this is a valid Sentinel-2 Level-1C product."
        )
    try:
        quant_value = float(quant_text)
    except ValueError:
        raise ValueError(
            f"[Radiometric] QUANTIFICATION_VALUE '{quant_text}' in '{meta_path}' "
            f"is not a valid number."
        )

    band_offsets = {}
    _band_id_to_name = {
        0: "B01", 1: "B02", 2: "B03", 3: "B04", 4: "B05", 5: "B06",
        6: "B07", 7: "B08", 8: "B8A", 9: "B09", 10: "B10", 11: "B11",
        12: "B12",
    }

    for elem in root.iter():
        if elem.tag.split("}")[-1] == "RADIO_ADD_OFFSET":
            band_id_str = elem.get("band_id")
            if band_id_str is None or elem.text is None:
                continue
            try:
                band_id = int(band_id_str)
                offset_val = float(elem.text.strip())
            except ValueError:
                continue
            band_name = _band_id_to_name.get(band_id)
            if band_name:
                band_offsets[band_name] = offset_val

    if not band_offsets:
        print(
            f"[Radiometric] No per-band additive offsets found in '{meta_path}'. "
            f"Proceeding with offset=0.0 for all bands (valid for older L1C products)."
        )

    return meta_path, quant_value, band_offsets


def _sentinel_band_name_from_filename(filename):
    """
    Extract the Sentinel-2 band name from a JP2 filename.

    Examples:
        T44QKF_20240315T051641_B02.jp2  -> "B02"
        T44QKF_20240315T051641_B8A.jp2  -> "B8A"
    """
    stem = splitext(filename)[0]
    match = re.search(r"_(B\d{2}|B8A)$", stem, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^(B\d{2}|B8A)$", stem, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def calculate_sentinel_reflectance(dn_arr, additive_offset, quant_value):
    """
    Apply the Sentinel-2 radiometric calibration formula:

        Reflectance = (DN + Additive_Offset) / Quantification_Value

    Returns
    -------
    np.ndarray (float32)
        Calibrated reflectance values (dimensionless, typically 0-1 range).
    """
    if quant_value == 0.0:
        raise ValueError(
            "[Radiometric] Quantification Value is zero -- cannot divide. "
            "Check the Sentinel-2 metadata XML for a valid QUANTIFICATION_VALUE."
        )
    dn = dn_arr.astype(np.float32)
    return ((dn + additive_offset) / quant_value).astype(np.float32)


def apply_sentinel_correction(dn_arr, info):
    """
    Sentinel-2 metadata resolution -- NO LONGER converts DN -> reflectance
    here. This is called from io_utils.read_and_prepare() at load time, and
    reflectance for every sensor (Sentinel-2 included) is computed in
    exactly one place: compute_common_reflectance(), called from
    pipeline.py step "1" immediately after read_and_prepare() returns.
    Keeping DN untouched at load time (this function only resolves
    metadata) means every sensor arrives at that single conversion call
    uniformly as raw DN, instead of Sentinel-2/Landsat silently already
    being in reflectance units while Resourcesat-2A was still DN. From
    that conversion call onward -- i.e. for alignment, homogeneity,
    composite, and every later pipeline stage -- all images are
    reflectance, not DN.

    Still resolves and returns the calibration metadata (meta_path,
    quant_value, band_offsets, band_name, additive_offset) since that's
    cheap, useful for the caller's info dict, and lets load-time failures
    (e.g. missing MTD_MSIL*.xml) still surface early as before.

    Returns
    -------
    np.ndarray
        dn_arr, unchanged.
    tuple
        (meta_path, quant_value, band_offsets, band_name, additive_offset)
    """
    filepath = info["filepath"]
    filename = info["filename"]

    meta_path, quant_value, band_offsets = read_sentinel_metadata(filepath)
    band_name = _sentinel_band_name_from_filename(filename)

    if band_name and band_name in band_offsets:
        additive_offset = band_offsets[band_name]
    elif band_name:
        print(
            f"[Radiometric] No additive offset entry for band '{band_name}' in "
            f"'{meta_path}'. Using offset=0.0."
        )
        additive_offset = 0.0
    else:
        print(
            f"[Radiometric] Could not determine band name from filename '{filename}'. "
            f"Using additive offset=0.0."
        )
        band_name = "UNKNOWN"
        additive_offset = 0.0

    return dn_arr, (meta_path, quant_value, band_offsets, band_name, additive_offset)





def _find_landsat_mtl(image_dir):
    """
    Search *image_dir* for a Landsat MTL parameter file matching *_MTL.txt.

    Returns the path to the first match, or None if not found.
    """
    pattern = join(image_dir, "*_MTL.txt")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _locate_landsat_metadata(image_dir):
    """
    Locate the Landsat MTL file in *image_dir*.

    Returns
    -------
    str
        Absolute path to the *_MTL.txt file.

    Raises
    ------
    FileNotFoundError
        If no *_MTL.txt file exists in *image_dir*.
    """
    path = _find_landsat_mtl(image_dir)
    if path is None:
        raise FileNotFoundError(
            f"[Radiometric] No *_MTL.txt file found in '{image_dir}'. "
            f"Place the Landsat MTL parameter file alongside the image band files."
        )
    return path


def read_landsat_metadata(image_dir):
    """
    Parse a Landsat MTL parameter file and return a flat dict of all
    key/value pairs (lowercased keys, string values).

    MTL files use the format:
        KEY = VALUE
    with GROUP / END_GROUP block markers that are ignored here -- we read
    every KEY = VALUE line regardless of which group it belongs to.

    Returns
    -------
    tuple of (meta_path: str, mtl_dict: dict)
        meta_path -- absolute path to the MTL file used.
        mtl_dict  -- flat dict, keys lowercased, e.g.:
                     {"reflectance_mult_band_4": "2.0000E-05",
                      "reflectance_add_band_4":  "-0.100000",
                      "sun_elevation":           "54.75000", ...}
    """
    meta_path = _locate_landsat_metadata(image_dir)
    mtl_dict = {}

    with open(meta_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("GROUP") or line.startswith("END"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip().lower()
                val = val.strip().strip('"')
                mtl_dict[key] = val

    return meta_path, mtl_dict


def _landsat_band_number_from_filename(filename):
    """
    Extract the Landsat band number from an image filename.

    Handles both Collection-1 and Collection-2 naming conventions:
        LC08_L1TP_144048_20241001_B4.TIF   -> "4"
        LC09_L2SP_144048_20241001_B5.TIF   -> "5"
        LT05_L1TP_144048_19900901_B3.TIF   -> "3"
        B4.TIF                             -> "4"  (bare band file)

    Returns
    -------
    str or None
        Band number as a string (e.g. "4"), or None if not parseable.
    """
    stem = splitext(filename)[0]
    match = re.search(r"_B(\d+)$", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.match(r"^B(\d+)$", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def calculate_landsat_reflectance(dn_arr, mult, add, sun_elevation_deg):
    """
    Apply the Landsat TOA reflectance formula with sun-angle correction.

    Step 1 -- scaled reflectance (without sun-angle correction):
        rho_prime = REFLECTANCE_MULT_BAND_x * DN + REFLECTANCE_ADD_BAND_x

    Step 2 -- convert sun elevation to radians:
        sun_rad = radians(SUN_ELEVATION)

    Step 3 -- final TOA reflectance:
        rho = rho_prime / sin(sun_rad)

    Parameters
    ----------
    dn_arr : np.ndarray
        Raw DN pixel values.
    mult : float
        REFLECTANCE_MULT_BAND_x from the MTL file.
    add : float
        REFLECTANCE_ADD_BAND_x from the MTL file.
    sun_elevation_deg : float
        SUN_ELEVATION from the MTL file, in decimal degrees.

    Returns
    -------
    np.ndarray (float32)
        TOA Reflectance array (dimensionless, typically 0-1 range).
    """
    sun_rad = math.radians(sun_elevation_deg)
    sin_sun = math.sin(sun_rad)
    if sin_sun == 0.0:
        raise ValueError(
            "[Radiometric] Cannot compute Landsat reflectance: "
            "sin(SUN_ELEVATION) is zero. Check the SUN_ELEVATION field in the MTL file."
        )
    dn = dn_arr.astype(np.float32)
    rho_prime = mult * dn + add
    return (rho_prime / sin_sun).astype(np.float32)


def apply_landsat_correction(dn_arr, info):
    """
    Landsat MTL metadata resolution -- NO LONGER converts DN -> reflectance
    here. This is called from io_utils.read_and_prepare() at load time, and
    reflectance for every sensor (Landsat included) is computed in
    exactly one place: compute_common_reflectance(), called from
    pipeline.py step "1" immediately after read_and_prepare() returns.
    Keeping DN untouched at load time (this function only resolves
    metadata) means every sensor arrives at that single conversion call
    uniformly as raw DN, instead of Sentinel-2/Landsat silently already
    being in reflectance units while Resourcesat-2A was still DN. From
    that conversion call onward -- i.e. for alignment, homogeneity,
    composite, and every later pipeline stage -- all images are
    reflectance, not DN.

    Still locates the *_MTL.txt file and reads REFLECTANCE_MULT_BAND_x,
    REFLECTANCE_ADD_BAND_x, and SUN_ELEVATION, since that's cheap, useful
    for the caller's info dict, and lets load-time failures (e.g. missing
    MTL file) still surface early as before.

    Parameters
    ----------
    dn_arr : np.ndarray
        Raw DN pixel values.
    info : dict
        Image info dict as returned by io_utils.read_and_prepare.

    Returns
    -------
    np.ndarray
        dn_arr, unchanged.
    tuple
        (meta_path, mult, add, sun_elevation_deg, band_number, mtl_dict)
        -- all resolved values for the caller's summary log.
    """
    image_dir = info["image_dir"]
    filename = info["filename"]

    meta_path, mtl_dict = read_landsat_metadata(image_dir)
    band_number = _landsat_band_number_from_filename(filename)

    if band_number is None:
        raise ValueError(
            f"[Radiometric] Could not determine band number from Landsat "
            f"filename '{filename}'. Expected a name ending in '_B<N>.TIF' "
            f"(e.g. LC08_..._B4.TIF)."
        )

    mult_key = f"reflectance_mult_band_{band_number}"
    add_key = f"reflectance_add_band_{band_number}"
    sun_key = "sun_elevation"

    missing = [k for k in (mult_key, add_key, sun_key) if k not in mtl_dict]
    if missing:
        raise ValueError(
            f"[Radiometric] MTL file '{meta_path}' is missing required key(s): "
            f"{', '.join(missing)}. Ensure this is a valid Landsat MTL file "
            f"with reflectance calibration coefficients."
        )

    try:
        mult = float(mtl_dict[mult_key])
        add = float(mtl_dict[add_key])
        sun_elevation_deg = float(mtl_dict[sun_key])
    except ValueError as e:
        raise ValueError(
            f"[Radiometric] Could not parse numeric calibration field from "
            f"'{meta_path}': {e}"
        )


    return dn_arr, (meta_path, mult, add, sun_elevation_deg, band_number, mtl_dict)

def get_band_calibration_full_auto(info):
    """
    Extended calibration lookup that works for Resourcesat-2A, Sentinel-2,
    and Landsat.

    For Resourcesat-2A this is identical to get_band_calibration_full().
    For Sentinel-2 and Landsat it returns a compatible tuple with
    sensor-specific fields packed into the metadata dict so that
    pipeline.py can treat all sensor types uniformly.

    Returns
    -------
    For RESOURCESAT2A:
        (lmin, lmax, max_dn, band_name, meta_path, metadata, sensor)
        -- identical to get_band_calibration_full().

    For SENTINEL2:
        (None, None, None, band_name, meta_path, s2_meta_dict, "SENTINEL2")
        where s2_meta_dict = {
            "quant_value": float,
            "band_offsets": dict,
            "additive_offset": float,
            "_sensor_family": "SENTINEL2",
        }

    For LANDSAT:
        (None, None, None, band_name, meta_path, landsat_meta_dict, "LANDSAT")
        where landsat_meta_dict = {
            "mult": float,
            "add": float,
            "sun_elevation": float,
            "band_number": str,
            "_sensor_family": "LANDSAT",
        }

    Callers must check sensor == "SENTINEL2" / "LANDSAT" and use
    compute_reflectance_for_array_auto() rather than
    compute_reflectance_for_array() for non-RS2A images.
    """
    sensor = detect_sensor(info)

    if sensor == "RESOURCESAT2A":
        return get_band_calibration_full(info)

    if sensor == "SENTINEL2":
        filepath = info["filepath"]
        filename = info["filename"]
        meta_path, quant_value, band_offsets = read_sentinel_metadata(filepath)
        band_name = _sentinel_band_name_from_filename(filename) or "UNKNOWN"
        additive_offset = band_offsets.get(band_name, 0.0)
        # Read SPACECRAFT_NAME so _instrument_name() can tell 2A from 2B.
        # MTD_MSIL1C.xml contains e.g. <SPACECRAFT_NAME>Sentinel-2A</SPACECRAFT_NAME>.
        _s2_tree = ET.parse(meta_path)
        _s2_root = _s2_tree.getroot()
        _spacecraft_name = None
        for _elem in _s2_root.iter():
            if _elem.tag.split("}")[-1] == "SPACECRAFT_NAME" and _elem.text:
                _spacecraft_name = _elem.text.strip()
                break
        s2_meta = {
            "quant_value": quant_value,
            "band_offsets": band_offsets,
            "additive_offset": additive_offset,
            "_sensor_family": "SENTINEL2",
            "spacecraft_name": _spacecraft_name,   # e.g. "Sentinel-2A" / "Sentinel-2B"
        }
        return None, None, None, band_name, meta_path, s2_meta, "SENTINEL2"

    image_dir = info["image_dir"]
    filename = info["filename"]
    meta_path, mtl_dict = read_landsat_metadata(image_dir)
    band_number = _landsat_band_number_from_filename(filename)

    if band_number is None:
        raise ValueError(
            f"[Radiometric] Could not determine band number from Landsat "
            f"filename '{filename}'."
        )

    mult_key = f"reflectance_mult_band_{band_number}"
    add_key = f"reflectance_add_band_{band_number}"
    sun_key = "sun_elevation"

    missing = [k for k in (mult_key, add_key, sun_key) if k not in mtl_dict]
    if missing:
        raise ValueError(
            f"[Radiometric] MTL file '{meta_path}' is missing required key(s): "
            f"{', '.join(missing)}."
        )

    mult = float(mtl_dict[mult_key])
    add = float(mtl_dict[add_key])
    sun_elevation_deg = float(mtl_dict[sun_key])
    band_name = f"B{band_number}"

    landsat_meta = {
        "mult": mult,
        "add": add,
        "sun_elevation": sun_elevation_deg,
        "band_number": band_number,
        "_sensor_family": "LANDSAT",
        "sensor_id": mtl_dict.get("sensor_id"),
        "spacecraft_id": mtl_dict.get("spacecraft_id"),
    }
    return None, None, None, band_name, meta_path, landsat_meta, "LANDSAT"



_SATELLITE_DISPLAY_NAMES = {
    "RESOURCESAT2A": "Resourcesat-2A",
    "SENTINEL2": "Sentinel-2",
    "LANDSAT": "Landsat",
    "CARTOSAT": "Cartosat",
}

_LISS_PATTERN = re.compile(r"LISS[-_]?([34])", re.IGNORECASE)


def _instrument_name(sensor, metadata, info):
    """
    Resolve the instrument/sensor name (as opposed to the satellite/
    platform name already in `sensor`) for the plot-label helper below.

    SENTINEL2 only ever carries the MSI instrument, so that's fixed.

    LANDSAT: reads SENSOR_ID out of the MTL dict captured above (e.g.
    "OLI_TIRS", "ETM+", "TM") -- falls back to "OLI" only if that key is
    genuinely absent from the MTL file.

    RESOURCESAT2A: the pipeline doesn't otherwise distinguish LISS-3 vs
    LISS-4 (both use BAND_META.txt), so this looks for a "LISS3"/"LISS4"
    /"LISS-3"/"LISS-4" token in the filename or its directory path --
    a convention already used in this project's own example paths
    (see cli.py's __main__ block). Falls back to the generic "LISS" if
    no such token is found.
    """
    if sensor == "SENTINEL2":
        # spacecraft_name is e.g. "Sentinel-2A" or "Sentinel-2B".
        # Return just the instrument part "MSI" -- the satellite suffix (2A/2B)
        # is handled in build_plot_label() via the satellite display name.
        return "MSI"

    if sensor == "LANDSAT":
        sensor_id = metadata.get("sensor_id") if isinstance(metadata, dict) else None
        return sensor_id.upper() if sensor_id else "OLI"

    if sensor == "RESOURCESAT2A":
        haystack = f"{info.get('filepath', '')} {info.get('image_dir', '')}"
        m = _LISS_PATTERN.search(haystack)
        return f"LISS{m.group(1)}" if m else "LISS"

    if sensor == "CARTOSAT":
        # Sensor field in BAND_META typically holds e.g. "P6-PAN-MX" or "P6-PAN-HX".
        sensor_field = metadata.get("sensor", "") if isinstance(metadata, dict) else ""
        return sensor_field.upper() if sensor_field else "PAN"

    return sensor


def build_plot_label(sensor, band_name, metadata, info):
    """
    Build the "{satellite}_{sensor}_{band}" label used for plot_common's
    axis labels (e.g. "Resourcesat-2A_LISS4_B3", "Sentinel-2A_MSI_B04",
    "Landsat-8_OLI_TIRS_B4", "Landsat-9_OLI-2_TIRS-2_B4").

    For Landsat the MTL's SPACECRAFT_ID field (e.g. "LANDSAT_8",
    "LANDSAT_9") is used to produce a specific name ("Landsat-8",
    "Landsat-9") rather than the generic "Landsat" fallback.

    For Sentinel-2 the XML's SPACECRAFT_NAME field (e.g. "Sentinel-2A",
    "Sentinel-2B") is used directly, falling back to the generic
    "Sentinel-2" only when that field was not found in the metadata.
    """
    if sensor == "LANDSAT" and isinstance(metadata, dict):
        spacecraft_id = metadata.get("spacecraft_id")  # e.g. "LANDSAT_8", "LANDSAT_9"
        if spacecraft_id:
            # "LANDSAT_8" -> "Landsat-8",  "LANDSAT_9" -> "Landsat-9"
            satellite = spacecraft_id.replace("_", "-").title()  # "Landsat-8"
        else:
            satellite = _SATELLITE_DISPLAY_NAMES.get(sensor, sensor)
    elif sensor == "SENTINEL2" and isinstance(metadata, dict):
        # spacecraft_name was injected into s2_meta above, e.g. "Sentinel-2A"
        satellite = metadata.get("spacecraft_name") or _SATELLITE_DISPLAY_NAMES.get(sensor, sensor)
    else:
        satellite = _SATELLITE_DISPLAY_NAMES.get(sensor, sensor)

    instrument = _instrument_name(sensor, metadata, info)
    return f"{satellite}_{instrument}_{band_name}"


def compute_reflectance_for_array_auto(dn_arr, lmin, lmax, max_dn,
                                        band_name, sensor,
                                        metadata, meta_path):
    """
    Sensor-agnostic reflectance computation, compatible with the signature
    of compute_reflectance_for_array().

    For RESOURCESAT2A, delegates to compute_reflectance_for_array().
    For SENTINEL2, reads additive_offset and quant_value from *metadata*.
    For LANDSAT, reads mult, add, and sun_elevation from *metadata*.

    All other parameters follow the same meaning as compute_reflectance_for_array.
    """
    if isinstance(metadata, dict):
        family = metadata.get("_sensor_family")

        if family == "SENTINEL2":
            additive_offset = metadata.get("additive_offset", 0.0)
            quant_value = metadata.get("quant_value", 10000.0)
            return calculate_sentinel_reflectance(dn_arr, additive_offset, quant_value)

        if family == "LANDSAT":
            mult = metadata["mult"]
            add = metadata["add"]
            sun_elevation_deg = metadata["sun_elevation"]
            return calculate_landsat_reflectance(dn_arr, mult, add, sun_elevation_deg)

    return compute_reflectance_for_array(
        dn_arr, lmin, lmax, max_dn, band_name, sensor, metadata, meta_path,
    )



def _print_radiometric_stats(label, arr):
    """Small helper for building check-lines describing a reflectance array's
    valid-pixel range/mean -- returns a string, never prints (that's the
    caller's job, via logging_utils.print_module_summary)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"{label} reflectance: no valid pixels."
    return (f"{label} reflectance: min={finite.min():.6f}, max={finite.max():.6f}, "
            f"mean={finite.mean():.6f}")


def compute_saturation_mask(dn_arr, max_dn, sensor):
    """
    Identify saturated DN pixels using the sensor's valid DN range.

    Saturated pixels (DN at or above the sensor's maximum recordable value)
    carry no reliable radiometric information -- the true reflectance could
    be anything at or above what the sensor could represent. Left in, they
    show up as artificial clustering/clipping at the high-reflectance end
    of any downstream scatter plot or regression.

    Parameters
    ----------
    dn_arr : ndarray
        Raw DN array (pre radiometric conversion).
    max_dn : float or None
        Sensor's maximum valid DN. If None, a sensor-appropriate default is
        used: RESOURCESAT2A/LANDSAT are conventionally delivered as 12-bit
        data stored in a 16-bit container (max_dn=4095) unless the caller's
        own calibration lookup already supplied a different value; Sentinel-2
        L1C/L2A DN is a 16-bit unsigned reflectance-scaled product (nominal
        ceiling 65535, though in practice values rarely approach it since
        quant_value=10000 maps reflectance=1.0 to DN=10000).
    sensor : str
        One of "RESOURCESAT2A", "SENTINEL2", "LANDSAT" (used only to pick
        the default max_dn when max_dn is None).

    Returns
    -------
    ndarray (bool)
        True where the pixel is saturated (should be excluded downstream).
    """
    if max_dn is None:
        max_dn = {
            "SENTINEL2": 65535.0,
            "LANDSAT": 65535.0,
        }.get(sensor, 4095.0)
    return dn_arr >= max_dn


def compute_common_reflectance(info_src, info_tgt, dn_src, dn_tgt):
    """
    Fully self-contained radiometric conversion: locates metadata
    (band_meta.txt / Landsat MTL / Sentinel-2 metadata -- whichever
    applies), resolves calibration parameters, and converts
    DN -> TOA Radiance -> TOA Reflectance for both images, including
    saturation masking (DN >= max_dn -> NaN, applied before conversion).

    Despite the "_common" in the name (kept for backward compatibility --
    several other modules reference this function by name in comments),
    this does NOT require its inputs to be pre-restricted to common-area
    pixels. It works on an array of any shape via
    compute_reflectance_for_array_auto()'s plain elementwise numpy ops.

    Called exactly ONCE per pipeline run, immediately after
    read_and_prepare() and BEFORE common-grid construction/any resampling
    (pipeline.py step "1", the radiometric-conversion sub-block) -- on the
    full src/target arrays, not a common-pixel slice. This is intentional:
    every downstream stage (ECC alignment, resampling,
    homogeneity, composite, output rasters) should see physical TOA
    reflectance, not sensor-native DN, so unit-mixing bugs (e.g. verifying
    flux preservation in DN space instead of reflectance space) can't creep
    back in. The plot/cross-sensor section near the end of the pipeline
    reuses this call's `details` dict and slices the already-reflectance
    common-grid arrays directly -- it does not call this function again.

    DN -> Reflectance conversion happens ONLY here now, for every sensor
    (Resourcesat-2A, Sentinel-2, Landsat, or any future sensor added to
    detect_sensor()). read_and_prepare() no longer pre-converts Sentinel-2
    or Landsat at load time (apply_sentinel_correction /
    apply_landsat_correction now only resolve calibration metadata and
    return dn_arr unchanged) -- so every image arrives here still in raw
    DN, regardless of sensor.

    This function never prints -- it returns everything the caller needs
    (including a `details` dict) so all console/log output can happen in
    one place, via logging_utils.print_module_summary().

    Returns (reflectance_src, reflectance_tgt, details).
    """
    (lmin_src, lmax_src, max_dn_src, band_name_src,
     meta_path_src, metadata_src, sensor_src) = get_band_calibration_full_auto(info_src)
    (lmin_tgt, lmax_tgt, max_dn_tgt, band_name_tgt,
     meta_path_tgt, metadata_tgt, sensor_tgt) = get_band_calibration_full_auto(info_tgt)

    def _resolve_family(s):
        if s in ("L3", "L4FX"):
            return "RESOURCESAT2A"
        return s  # SENTINEL2, LANDSAT, CARTOSAT pass through unchanged

    sensor_src_family = _resolve_family(sensor_src)
    sensor_tgt_family = _resolve_family(sensor_tgt)

    sat_mask_src = compute_saturation_mask(dn_src, max_dn_src, sensor_src_family)
    sat_mask_tgt = compute_saturation_mask(dn_tgt, max_dn_tgt, sensor_tgt_family)
    n_sat_src = int(np.count_nonzero(sat_mask_src))
    n_sat_tgt = int(np.count_nonzero(sat_mask_tgt))
    dn_src_masked = dn_src.astype(np.float32)
    dn_tgt_masked = dn_tgt.astype(np.float32)
    if n_sat_src:
        dn_src_masked = np.where(sat_mask_src, np.nan, dn_src_masked)
    if n_sat_tgt:
        dn_tgt_masked = np.where(sat_mask_tgt, np.nan, dn_tgt_masked)

    reflectance_src = compute_reflectance_for_array_auto(
        dn_src_masked, lmin_src, lmax_src, max_dn_src,
        band_name_src, sensor_src, metadata_src, meta_path_src,
    )
    reflectance_tgt = compute_reflectance_for_array_auto(
        dn_tgt_masked, lmin_tgt, lmax_tgt, max_dn_tgt,
        band_name_tgt, sensor_tgt, metadata_tgt, meta_path_tgt,
    )

    details = {
        "sensor_src": sensor_src_family, "sensor_tgt": sensor_tgt_family,
        "band_name_src": band_name_src, "band_name_tgt": band_name_tgt,
        "meta_path_src": meta_path_src, "meta_path_tgt": meta_path_tgt,
        "lmin_src": lmin_src, "lmax_src": lmax_src, "max_dn_src": max_dn_src,
        "lmin_tgt": lmin_tgt, "lmax_tgt": lmax_tgt, "max_dn_tgt": max_dn_tgt,
        "n_saturated_src": n_sat_src, "n_saturated_tgt": n_sat_tgt,
        "reflectance_src_stats": _print_radiometric_stats("source", reflectance_src),
        "reflectance_tgt_stats": _print_radiometric_stats("target", reflectance_tgt),
        "plot_label_src": build_plot_label(sensor_src_family, band_name_src, metadata_src, info_src),
        "plot_label_tgt": build_plot_label(sensor_tgt_family, band_name_tgt, metadata_tgt, info_tgt),
        "metadata_src": metadata_src, "metadata_tgt": metadata_tgt,
        "info_src": info_src, "info_tgt": info_tgt,
    }
    return reflectance_src, reflectance_tgt, details
