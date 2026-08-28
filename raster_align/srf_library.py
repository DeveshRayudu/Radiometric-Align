"""
raster_align.srf_library
========================

Spectral Response Function (SRF) library for compute_sbaf().

Two SRF sources are used, chosen automatically per sensor/band:

1. MEASURED --
   a) Sentinel-2 -- the exact per-nm Relative Spectral Response curve
      digitised by ESA and shipped inside every Sentinel-2 product's own
      MTD_MSIL1C.xml (Product_Image_Characteristics /
      Spectral_Information_List / Spectral_Information / Spectral_Response
      / VALUES). This is parsed straight from the scene's own metadata file
      at runtime by parse_sentinel_rsr_xml() -- the same meta_path already
      resolved by radiometric_correction.read_sentinel_metadata() for the
      TOA reflectance conversion -- so it's exact for whichever Sentinel-2A
      or Sentinel-2B granule is actually being processed, not a generic
      "S2A" placeholder. get_srf() and compute_sbaf_for_bands() use this
      automatically whenever an xml_path is supplied for a Sentinel-2 band;
      compute_auto_sbaf() supplies it automatically from rad_dict.
   b) Landsat-8 OLI -- the official per-band SRF extracted from Ball
      Aerospace's BA_RSR.v1.1.xlsx, bundled as static RTTOV-format
      coefficient files under resources/landsat8_oli_srf/ (one file per
      band, ch01=B1 .. ch09=B9). Unlike Sentinel-2, this is fixed per-
      instrument rather than per-scene (OLI carries no per-scene SRF in
      its own metadata), so it's shipped as a package resource and loaded
      once via _load_l8oli_measured_srf(), not parsed from the scene being
      processed. get_srf() uses this automatically for sensor="L8OLI" (or
      the "Landsat8"/"LANDSAT8OLI" aliases) whenever the bundled file for
      that band parses successfully; falls back to Gaussian otherwise.
      NOT applied to L9OLI2 -- Landsat-9's OLI-2 is spectrally very close
      to OLI but not identical, and no measured OLI-2 curve has been
      bundled, so L9OLI2 still uses the Gaussian approximation rather than
      silently reusing OLI-8's measured data under a different sensor.

2. GAUSSIAN APPROXIMATION (LISS-3, LISS-4, Landsat-9 OLI-2, and any
   Sentinel-2/Landsat-8 band/scene where the measured curve can't be
   parsed) -- centred on the band midpoint with Ïƒ chosen so that the
   Gaussian's FWHM equals the official band width from the USGS system-
   characterisation reports. This is Option 2 from the project
   methodology document -- a well-motivated analytic form, used here
   because no digitised per-nm RSR is available for that sensor/band.

Sources
-------
LISS-3  : USGS OFR 2021-1030-H (Table 2 / Figure 1)               [Gaussian]
LISS-4  : USGS OFR 2021-1030-U (Table 2 / Figure 1)               [Gaussian]
Landsat-8 OLI  : rtcoef_landsat_8_oli_srf_ch01..09.txt (RTTOV coefficient
                 format, SRF extracted from Ball_BA_RSR.v1.1.xlsx),
                 bundled under resources/landsat8_oli_srf/           [Measured]
                 Falls back to band centres/FWHM from
                 https://landsat.usgs.gov/sites/default/files/spectral_response/L8_OLI_SRF_20150706.xlsx
                 (Gaussian) if a file is missing or fails to parse.
Landsat-9 OLI-2: same source family as L8 OLI's Gaussian fallback;
                 OLI-2 is spectrally near-identical to OLI, but no
                 measured OLI-2 curve is bundled                      [Gaussian]
Sentinel-2 MSI : the scene's own MTD_MSIL1C.xml Spectral_Response VALUES  [Measured]
                 Falls back to ESA S2-SRF document COPE-GSEG-EOPG-TN-15-0007
                 v3.0 band centres/FWHM (Gaussian) if XML parsing fails
                 or no xml_path is supplied.

Public API
----------
get_srf(sensor, band, wavelengths_nm, xml_path=None)
    -> np.ndarray of normalised RSR values on the supplied wavelength grid.
    Uses the measured curve from xml_path when sensor is Sentinel-2 and
    the band is found there; Gaussian otherwise.

parse_sentinel_rsr_xml(xml_path)
    -> dict[str, tuple[np.ndarray, np.ndarray]] mapping normalised band
    names (e.g. "B02", "B8A") to (wavelengths_nm, rsr) digitised curves,
    read straight from that scene's MTD_MSIL1C.xml. Cached per path.

compute_sbaf_for_bands(sensor_ref, band_ref, sensor_tgt, band_tgt,
                        spectrum="dry_grass", wl_min=400, wl_max=2500, step=1,
                        srf_ref_xml=None, srf_tgt_xml=None)
    -> float  (SBAF factor, ready to pass to pipeline's sbaf= argument)

list_sensors()   -> list[str]
list_bands(sensor) -> list[str]

Internal helpers
----------------
_gaussian_rsr(wavelengths_nm, center_nm, fwhm_nm) -> np.ndarray
_make_reference_spectrum(name, wavelengths_nm)     -> np.ndarray
"""

from __future__ import annotations

import functools
import re as _re
import xml.etree.ElementTree as _ET

import numpy as np


_BAND_DEFS: dict[str, dict[str, tuple[float, float]]] = {

    "LISS3": {
        "B2":   (555.0, 70.0),   # Green  0.52â€“0.59 Âµm
        "B3":   (650.0, 60.0),   # Red    0.62â€“0.68 Âµm
        "B4":   (815.0, 90.0),   # NIR    0.77â€“0.86 Âµm
        "B5":  (1625.0, 150.0),  # SWIR   1.55â€“1.70 Âµm
        "green": (555.0, 70.0),
        "red":   (650.0, 60.0),
        "nir":   (815.0, 90.0),
        "swir":  (1625.0, 150.0),
    },

    "LISS4": {
        "B2":   (555.0, 70.0),   # Green  0.52â€“0.59 Âµm
        "B3":   (650.0, 60.0),   # Red    0.62â€“0.68 Âµm
        "B4":   (815.0, 90.0),   # NIR    0.77â€“0.86 Âµm
        "green": (555.0, 70.0),
        "red":   (650.0, 60.0),
        "nir":   (815.0, 90.0),
    },

    "L8OLI": {
        "B1":   (443.0,  20.0),  # Coastal/Aerosol
        "B2":   (482.0,  60.0),  # Blue
        "B3":   (561.5,  57.0),  # Green
        "B4":   (654.5,  37.5),  # Red
        "B5":   (865.0,  28.0),  # NIR
        "B6":  (1608.5,  84.5),  # SWIR-1
        "B7":  (2200.5, 187.0),  # SWIR-2
        "B8":   (590.0, 180.0),  # Panchromatic
        "B9":  (1373.5,  20.0),  # Cirrus
        "coastal": (443.0,  20.0),
        "blue":  (482.0,  60.0),
        "green": (561.5,  57.0),
        "red":   (654.5,  37.5),
        "nir":   (865.0,  28.0),
        "swir1": (1608.5,  84.5),
        "swir2": (2200.5, 187.0),
        "pan":   (590.0, 180.0),
        "cirrus": (1373.5,  20.0),
    },

    "L9OLI2": {
        "B2":   (482.0,  60.0),
        "B3":   (561.5,  57.0),
        "B4":   (654.5,  37.5),
        "B5":   (865.0,  28.0),
        "B6":  (1608.5,  84.5),
        "B7":  (2200.5, 187.0),
        "blue":  (482.0,  60.0),
        "green": (561.5,  57.0),
        "red":   (654.5,  37.5),
        "nir":   (865.0,  28.0),
        "swir1": (1608.5,  84.5),
        "swir2": (2200.5, 187.0),
    },

    "S2A": {
        "B02":  (492.4,  66.0),  # Blue
        "B03":  (559.8,  36.0),  # Green
        "B04":  (664.6,  31.0),  # Red
        "B08":  (832.8, 106.0),  # NIR broad
        "B8A":  (864.7,  21.0),  # NIR narrow
        "B11": (1613.7,  91.0),  # SWIR-1
        "B12": (2202.4, 175.0),  # SWIR-2
        "blue":  (492.4,  66.0),
        "green": (559.8,  36.0),
        "red":   (664.6,  31.0),
        "nir":   (832.8, 106.0),
        "swir1": (1613.7,  91.0),
        "swir2": (2202.4, 175.0),
    },

    "S2B": {
        "B02":  (492.1,  66.0),
        "B03":  (558.9,  36.0),
        "B04":  (664.9,  31.0),
        "B08":  (832.9, 106.0),
        "B8A":  (864.8,  21.0),
        "B11": (1610.4,  94.0),
        "B12": (2185.7, 185.0),
        "blue":  (492.1,  66.0),
        "green": (558.9,  36.0),
        "red":   (664.9,  31.0),
        "nir":   (832.9, 106.0),
        "swir1": (1610.4,  94.0),
        "swir2": (2185.7, 185.0),
    },
}

_SENSORS = list(_BAND_DEFS.keys())


def _spec_flat(wl: np.ndarray) -> np.ndarray:
    """Perfectly flat (Lambertian grey) spectrum -- useful baseline."""
    return np.ones_like(wl, dtype=np.float64)


def _spec_dry_grass(wl: np.ndarray) -> np.ndarray:
    """
    Analytic approximation to dry / senescent grass reflectance.

    Matches the broad shape of USGS splib07 'Dry_grass' spectrum:
      - Low in blue (~0.10), rising through visible
      - Slight red-edge bump around 720 nm, then high NIR plateau (~0.40)
      - Cellulose/lignin dip near 2100 nm in SWIR
    """
    rfl = np.empty_like(wl, dtype=np.float64)
    for i, w in enumerate(wl):
        if w < 400:
            rfl[i] = 0.05
        elif w < 700:
            rfl[i] = 0.10 + 0.20 * ((w - 400) / 300)
        elif w < 750:
            rfl[i] = 0.30 + 0.10 * ((w - 700) / 50)
        elif w < 1300:
            rfl[i] = 0.40 - 0.02 * ((w - 750) / 550)
        elif w < 1400:
            rfl[i] = 0.38 - 0.15 * ((w - 1300) / 100)
        elif w < 1800:
            rfl[i] = 0.30 - 0.05 * ((w - 1400) / 400)
        elif w < 1950:
            rfl[i] = 0.25 - 0.15 * ((w - 1800) / 150)
        elif w < 2500:
            rfl[i] = 0.20 - 0.10 * ((w - 1950) / 550)
        else:
            rfl[i] = 0.10
    return rfl


def _spec_bare_soil(wl: np.ndarray) -> np.ndarray:
    """
    Analytic approximation to dry bare soil (loamy sand type).

    Matches USGS splib07 'Dry soil' broadly:
      - Smoothly rising from ~0.10 (blue) to ~0.35 (SWIR-1)
      - Small dip at water vapour bands
    """
    rfl = np.empty_like(wl, dtype=np.float64)
    for i, w in enumerate(wl):
        if w < 400:
            rfl[i] = 0.08
        elif w < 1350:
            rfl[i] = 0.08 + 0.27 * ((w - 400) / 950)
        elif w < 1450:
            rfl[i] = 0.35 - 0.10 * ((w - 1350) / 100)  # water band dip
        elif w < 1800:
            rfl[i] = 0.30 + 0.05 * ((w - 1450) / 350)
        elif w < 2000:
            rfl[i] = 0.35 - 0.12 * ((w - 1800) / 200)  # water band dip
        elif w < 2500:
            rfl[i] = 0.28 - 0.05 * ((w - 2000) / 500)
        else:
            rfl[i] = 0.23
    return rfl


def _spec_green_veg(wl: np.ndarray) -> np.ndarray:
    """
    Analytic approximation to green vegetation (full canopy).

    Matches the chlorophyll absorption / red-edge / NIR plateau shape:
      - Very low in red (~0.05), strong red-edge, high NIR (~0.50)
    """
    rfl = np.empty_like(wl, dtype=np.float64)
    for i, w in enumerate(wl):
        if w < 400:
            rfl[i] = 0.03
        elif w < 500:
            rfl[i] = 0.05
        elif w < 560:
            rfl[i] = 0.05 + 0.10 * ((w - 500) / 60)
        elif w < 670:
            rfl[i] = 0.15 - 0.10 * ((w - 560) / 110)
        elif w < 730:
            rfl[i] = 0.05 + 0.45 * ((w - 670) / 60)
        elif w < 1300:
            rfl[i] = 0.50 - 0.05 * ((w - 730) / 570)
        elif w < 1450:
            rfl[i] = 0.45 - 0.20 * ((w - 1300) / 150)
        elif w < 1800:
            rfl[i] = 0.30 - 0.05 * ((w - 1450) / 350)
        elif w < 2000:
            rfl[i] = 0.25 - 0.15 * ((w - 1800) / 200)
        elif w < 2500:
            rfl[i] = 0.15 - 0.08 * ((w - 2000) / 500)
        else:
            rfl[i] = 0.07
    return rfl


_REFERENCE_SPECTRA: dict[str, callable] = {
    "flat":       _spec_flat,
    "dry_grass":  _spec_dry_grass,
    "bare_soil":  _spec_bare_soil,
    "green_veg":  _spec_green_veg,
}



def _gaussian_rsr(
    wavelengths_nm: np.ndarray,
    center_nm: float,
    fwhm_nm: float,
) -> np.ndarray:
    """
    Gaussian RSR curve normalised to peak = 1.

    sigma is set so that the Gaussian's FWHM equals ``fwhm_nm``, which in
    turn equals the official band width (upper_limit - lower_limit).

    Parameters
    ----------
    wavelengths_nm : np.ndarray
        Wavelength grid (nm), monotonically increasing.
    center_nm : float
        Band centre wavelength (nm).
    fwhm_nm : float
        Full-width at half-maximum (nm).

    Returns
    -------
    np.ndarray
        RSR values in [0, 1], same shape as ``wavelengths_nm``.
    """
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    rsr = np.exp(-0.5 * ((wavelengths_nm - center_nm) / sigma) ** 2)
    return rsr  # already normalised (peak = 1 at center_nm)



def _normalize_s2_band(physical_band: str) -> str:
    """
    Normalise a Sentinel-2 XML ``physicalBand`` attribute (e.g. ``"B1"``,
    ``"B8A"``, ``"B12"``) to this library's band-key convention
    (``"B01"``, ``"B8A"``, ``"B12"``), matching the zero-padded keys
    already used in ``_BAND_DEFS["S2A"/"S2B"]`` (``"B02"``, ``"B03"``,
    ``"B04"``, ``"B08"``, ``"B8A"``, ``"B11"``, ``"B12"``).
    """
    physical_band = physical_band.upper()
    if physical_band.endswith("A"):
        return physical_band
    m = _re.match(r"B(\d+)$", physical_band)
    if not m:
        return physical_band
    return f"B{int(m.group(1)):02d}"


@functools.lru_cache(maxsize=16)
def parse_sentinel_rsr_xml(xml_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Parse the measured per-nm Relative Spectral Response curve for every
    band out of a Sentinel-2 ``MTD_MSIL1C.xml`` metadata file.

    Reads ``Product_Image_Characteristics / Spectral_Information_List /
    Spectral_Information`` (one per band), each of which has a
    ``Wavelength/MIN`` (nm), a ``Spectral_Response/STEP`` (nm), and a
    ``Spectral_Response/VALUES`` whitespace-separated list of RSR samples
    starting at MIN and stepping by STEP. The wavelength grid for each
    band is therefore reconstructed as ``MIN + STEP * arange(len(VALUES))``.

    Cached per ``xml_path`` (an XML file's RSR data is fixed for a given
    satellite/instrument, so repeated calls for the same scene -- once
    for the source image, once as a cross-check -- don't re-parse).

    Parameters
    ----------
    xml_path : str
        Path to an MTD_MSIL1C.xml file (as returned by
        ``radiometric_correction.read_sentinel_metadata()``'s
        ``meta_path``, or ``rad_details["meta_path_src"/"meta_path_tgt"]``
        in the pipeline).

    Returns
    -------
    dict[str, (np.ndarray, np.ndarray)]
        Maps normalised band name (e.g. ``"B02"``, ``"B8A"``) to
        ``(wavelengths_nm, rsr)`` arrays, both float64, RSR peak == 1.0.
        Empty dict if the file can't be parsed or has no spectral data
        (caller should fall back to the Gaussian approximation).
    """
    try:
        tree = _ET.parse(xml_path)
    except (OSError, _ET.ParseError):
        return {}

    root = tree.getroot()
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for spec_info in root.iter():
        if spec_info.tag.split("}")[-1] != "Spectral_Information":
            continue
        physical_band = spec_info.attrib.get("physicalBand")
        if not physical_band:
            continue

        wl_min = step = None
        values = None
        for child in spec_info.iter():
            tag = child.tag.split("}")[-1]
            if tag == "MIN" and child.text:
                wl_min = float(child.text)
            elif tag == "STEP" and child.text:
                step = float(child.text)
            elif tag == "VALUES" and child.text:
                values = np.array([float(v) for v in child.text.split()],
                                   dtype=np.float64)

        if wl_min is None or step is None or values is None or len(values) == 0:
            continue

        wl = wl_min + step * np.arange(len(values), dtype=np.float64)
        peak = values.max()
        if peak > 0:
            values = values / peak
        curves[_normalize_s2_band(physical_band)] = (wl, values)

    return curves


def _measured_s2_rsr(
    xml_path: str,
    band: str,
    wavelengths_nm: np.ndarray,
) -> np.ndarray | None:
    """
    Return the measured RSR for a Sentinel-2 band interpolated onto
    ``wavelengths_nm``, or None if unavailable (caller falls back to
    Gaussian). Zero outside the digitised curve's own wavelength range
    since these are narrow-band filters with negligible out-of-band
    response beyond what ESA digitised.
    """
    curves = parse_sentinel_rsr_xml(xml_path)
    if not curves:
        return None
    band_key = _normalize_s2_band(band)
    if band_key not in curves:
        return None
    wl_curve, rsr_curve = curves[band_key]
    return np.interp(wavelengths_nm, wl_curve, rsr_curve, left=0.0, right=0.0)



_L8OLI_SRF_FILENAMES = {
    "B1": "rtcoef_landsat_8_oli_srf_ch01.txt",
    "B2": "rtcoef_landsat_8_oli_srf_ch02.txt",
    "B3": "rtcoef_landsat_8_oli_srf_ch03.txt",
    "B4": "rtcoef_landsat_8_oli_srf_ch04.txt",
    "B5": "rtcoef_landsat_8_oli_srf_ch05.txt",
    "B6": "rtcoef_landsat_8_oli_srf_ch06.txt",
    "B7": "rtcoef_landsat_8_oli_srf_ch07.txt",
    "B8": "rtcoef_landsat_8_oli_srf_ch08.txt",
    "B9": "rtcoef_landsat_8_oli_srf_ch09.txt",
}


def parse_rttov_srf_txt(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Parse one RTTOV-format SRF coefficient file (wavenumber/response pairs)
    into ``(wavelengths_nm, rsr)``, sorted ascending by wavelength with
    peak RSR normalised to 1.0.

    File format (see resources/landsat8_oli_srf/*.txt):
        <channel>, <name>  <lo> - <hi> mu   ;<provenance comment>
        Number of data points:
        <N>
        Wavenumber (cm-1)   Filter response
            <wavenumber_1>       <response_1>
            ...

    Wavenumber -> wavelength conversion: wavelength_nm = 1e7 / wavenumber_cm-1
    (standard cm^-1 <-> nm relation). The file's own rows are typically in
    ascending-wavenumber (= descending-wavelength) order; this function
    always re-sorts by ascending wavelength regardless of input order, so
    ``np.interp`` (which requires ascending x) works correctly downstream.

    Returns
    -------
    (np.ndarray, np.ndarray) or None
        ``(wavelengths_nm, rsr)``, both float64, or None if the file is
        missing/malformed (caller falls back to the Gaussian
        approximation).
    """
    try:
        with open(path, "r") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    wavenumbers = []
    responses = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            wn = float(parts[0])
            resp = float(parts[1])
        except ValueError:
            continue
        wavenumbers.append(wn)
        responses.append(resp)

    if len(wavenumbers) < 2:
        return None

    wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)

    with np.errstate(divide="ignore"):
        wavelengths_nm = np.where(wavenumbers > 0, 1.0e7 / wavenumbers, np.nan)
    valid = np.isfinite(wavelengths_nm)
    wavelengths_nm, responses = wavelengths_nm[valid], responses[valid]
    if wavelengths_nm.size < 2:
        return None

    order = np.argsort(wavelengths_nm)
    wavelengths_nm, responses = wavelengths_nm[order], responses[order]

    peak = responses.max()
    if peak > 0:
        responses = responses / peak

    return wavelengths_nm, responses


@functools.lru_cache(maxsize=1)
def _load_l8oli_measured_srf() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load and cache all bundled Landsat-8 OLI measured SRF curves.

    Returns
    -------
    dict[str, (np.ndarray, np.ndarray)]
        Maps band name (``"B1"``..``"B9"``) to ``(wavelengths_nm, rsr)``.
        Bands whose file is missing/unparseable are simply absent from the
        dict (caller falls back to Gaussian for that band only).
    """
    from .resources import resource_path
    import os as _os

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for band, filename in _L8OLI_SRF_FILENAMES.items():
        path = resource_path(_os.path.join("resources", "landsat8_oli_srf", filename))
        parsed = parse_rttov_srf_txt(path)
        if parsed is not None:
            curves[band] = parsed
    return curves


def _measured_l8oli_rsr(
    band: str,
    wavelengths_nm: np.ndarray,
) -> np.ndarray | None:
    """
    Return the measured Landsat-8 OLI RSR for ``band``, interpolated onto
    ``wavelengths_nm``, or None if unavailable (caller falls back to
    Gaussian). Zero outside the digitised curve's own wavelength range.
    """
    curves = _load_l8oli_measured_srf()
    band_key = band.upper() if band.upper().startswith("B") and band[1:].isdigit() else None
    if band_key is None:
        alias_to_band = {
            "coastal": "B1", "blue": "B2", "green": "B3", "red": "B4",
            "nir": "B5", "swir1": "B6", "swir2": "B7", "pan": "B8",
            "cirrus": "B9",
        }
        band_key = alias_to_band.get(band.lower())
    if band_key is None or band_key not in curves:
        return None
    wl_curve, rsr_curve = curves[band_key]
    return np.interp(wavelengths_nm, wl_curve, rsr_curve, left=0.0, right=0.0)


def _resolve_band(sensor: str, band: str) -> tuple[float, float]:
    """Return (center_nm, fwhm_nm) for a sensor/band pair."""
    sensor_key = sensor.upper().replace("-", "").replace("_", "")
    _sensor_map = {
        "LISS3": "LISS3",
        "LISS4": "LISS4",
        "L8OLI": "L8OLI",
        "LANDSAT8": "L8OLI",
        "LANDSAT8OLI": "L8OLI",
        "L9OLI2": "L9OLI2",
        "LANDSAT9": "L9OLI2",
        "LANDSAT9OLI2": "L9OLI2",
        "S2A": "S2A",
        "SENTINEL2A": "S2A",
        "SENTINEL2AMSI": "S2A",
        "S2B": "S2B",
        "SENTINEL2B": "S2B",
        "SENTINEL2BMSI": "S2B",
    }
    resolved_sensor = _sensor_map.get(sensor_key)
    if resolved_sensor is None:
        raise KeyError(
            f"Unknown sensor '{sensor}'. "
            f"Available: {list_sensors()}"
        )
    band_defs = _BAND_DEFS[resolved_sensor]
    band_key = band.upper() if band.upper() in band_defs else band.lower()
    if band_key not in band_defs:
        raise KeyError(
            f"Band '{band}' not found for sensor '{sensor}'. "
            f"Available bands: {list_bands(sensor)}"
        )
    return band_defs[band_key]



def list_sensors() -> list[str]:
    """Return canonical sensor names supported by this library."""
    return list(_SENSORS)


def list_bands(sensor: str) -> list[str]:
    """Return canonical band names (excluding lowercase aliases) for a sensor."""
    sensor_key = sensor.upper().replace("-", "").replace("_", "")
    _sensor_map = {
        "LISS3": "LISS3", "LISS4": "LISS4",
        "L8OLI": "L8OLI", "LANDSAT8": "L8OLI",
        "L9OLI2": "L9OLI2", "LANDSAT9": "L9OLI2",
        "S2A": "S2A", "SENTINEL2A": "S2A",
        "S2B": "S2B", "SENTINEL2B": "S2B",
    }
    resolved = _sensor_map.get(sensor_key, sensor_key)
    if resolved not in _BAND_DEFS:
        raise KeyError(f"Unknown sensor '{sensor}'.")
    return [k for k in _BAND_DEFS[resolved] if not k[0].islower()]


def get_srf(
    sensor: str,
    band: str,
    wavelengths_nm: np.ndarray,
    xml_path: str | None = None,
) -> np.ndarray:
    """
    Return the SRF for a sensor/band on a given wavelength grid.

    If ``xml_path`` is supplied and ``sensor`` is a Sentinel-2 variant
    (``"S2A"``/``"S2B"``), this first tries to use the **measured**
    per-nm RSR curve parsed from that scene's own MTD_MSIL1C.xml (see
    :func:`parse_sentinel_rsr_xml`). If that curve isn't available for
    this band/file (parse failure, missing spectral block, or a non-
    Sentinel-2 sensor), it falls back to the Gaussian approximation.

    Parameters
    ----------
    sensor : str
        One of the sensor names returned by ``list_sensors()``, or a common
        alias (e.g. ``"Landsat8"``, ``"Sentinel2A"``, ``"LISS-3"``).
    band : str
        Band name (e.g. ``"B3"``, ``"red"``, ``"B04"``).
    wavelengths_nm : array_like
        Wavelength grid in nm.  The returned SRF is evaluated at these points.
    xml_path : str, optional
        Path to the Sentinel-2 scene's MTD_MSIL1C.xml. Ignored for non-
        Sentinel-2 sensors. When omitted, Sentinel-2 also falls back to
        the Gaussian approximation (previous behaviour).

    Returns
    -------
    np.ndarray
        Normalised RSR values (peak = 1) at each wavelength. For the
        Gaussian fallback, values outside Â±3Ïƒ of the band centre are
        effectively zero; for the measured curve, values outside the
        digitised range are exactly zero.
    """
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    center_nm, fwhm_nm = _resolve_band(sensor, band)

    sensor_key = sensor.upper().replace("-", "").replace("_", "")

    if xml_path is not None:
        _s2_sensor_map = {
            "S2A": "S2A", "SENTINEL2A": "S2A", "SENTINEL2AMSI": "S2A",
            "S2B": "S2B", "SENTINEL2B": "S2B", "SENTINEL2BMSI": "S2B",
        }
        if sensor_key in _s2_sensor_map:
            measured = _measured_s2_rsr(xml_path, band, wl)
            if measured is not None:
                return measured

    if sensor_key in ("L8OLI", "LANDSAT8", "LANDSAT8OLI"):
        measured = _measured_l8oli_rsr(band, wl)
        if measured is not None:
            return measured

    return _gaussian_rsr(wl, center_nm, fwhm_nm)


def get_reference_spectrum(
    name: str,
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """
    Return a built-in reference spectrum evaluated on a wavelength grid.

    Parameters
    ----------
    name : str
        One of: ``"flat"``, ``"dry_grass"``, ``"bare_soil"``, ``"green_veg"``.
    wavelengths_nm : array_like
        Wavelength grid in nm.

    Returns
    -------
    np.ndarray
        Reflectance values (0â€“1) at each wavelength.
    """
    if name not in _REFERENCE_SPECTRA:
        raise KeyError(
            f"Unknown spectrum '{name}'. "
            f"Available: {list(_REFERENCE_SPECTRA.keys())}"
        )
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    return _REFERENCE_SPECTRA[name](wl)


def compute_sbaf_for_bands(
    sensor_ref: str,
    band_ref: str,
    sensor_tgt: str,
    band_tgt: str,
    spectrum: str = "dry_grass",
    wl_min: float = 400.0,
    wl_max: float = 2500.0,
    step: float = 1.0,
    custom_spectrum: np.ndarray | None = None,
    custom_wavelengths: np.ndarray | None = None,
    srf_ref_xml: str | None = None,
    srf_tgt_xml: str | None = None,
) -> float:
    """
    Compute SBAF between two sensor bands.

    Uses each side's measured Sentinel-2 RSR curve (parsed from
    ``srf_ref_xml``/``srf_tgt_xml``, that scene's own MTD_MSIL1C.xml) when
    supplied and the corresponding sensor is Sentinel-2; the Gaussian
    approximation otherwise (LISS-3/4, Landsat-8/9, or Sentinel-2 without
    an xml_path). This is the single function to call from the pipeline
    or CLI. It:
      1. Builds a uniform wavelength grid from ``wl_min`` to ``wl_max``.
      2. Evaluates the SRF for both bands on that grid (measured or Gaussian,
         see above).
      3. Evaluates the chosen reference spectrum on the same grid.
      4. Calls ``compute_sbaf()`` from cross_sensor_metrics.

    The returned float is the SBAF multiplicative factor.  Pass it directly
    to ``run_pipeline(..., sbaf=<value>)`` or ``compute_cross_sensor_report()``.

    Parameters
    ----------
    sensor_ref, band_ref : str
        Reference sensor and band (the one being adjusted *to*).
    sensor_tgt, band_tgt : str
        Target sensor and band (the one whose data is being corrected).
    spectrum : str
        Built-in reference spectrum: ``"flat"``, ``"dry_grass"`` (default),
        ``"bare_soil"``, or ``"green_veg"``.
        Ignored if ``custom_spectrum`` and ``custom_wavelengths`` are given.
    wl_min, wl_max, step : float
        Wavelength grid bounds and step size in nm.  1 nm step is fine for
        all sensors in this library.
    custom_spectrum : np.ndarray, optional
        User-supplied reflectance array (0â€“1) matching ``custom_wavelengths``.
    custom_wavelengths : np.ndarray, optional
        Wavelengths (nm) for ``custom_spectrum``.
    srf_ref_xml, srf_tgt_xml : str, optional
        Path to the reference/target scene's MTD_MSIL1C.xml, if that side
        is Sentinel-2 and a measured RSR curve should be used instead of
        the Gaussian approximation. Ignored for non-Sentinel-2 sensors.

    Returns
    -------
    float
        SBAF factor.  Multiply the target reflectance by this value to bring
        it onto the reference sensor's spectral scale.

    Examples
    --------
    >>> from raster_align.srf_library import compute_sbaf_for_bands
    >>> sbaf = compute_sbaf_for_bands("LISS3", "B3", "LISS4", "B3")
    >>> print(f"LISS-3 Red vs LISS-4 Red SBAF = {sbaf:.4f}")

    >>> sbaf = compute_sbaf_for_bands("LISS3", "B3", "S2A", "B04",
    ...                                spectrum="bare_soil")
    >>> print(f"LISS-3 Red vs Sentinel-2A B4 SBAF = {sbaf:.4f}")

    >>> # Using the scene's own measured Sentinel-2 RSR curve:
    >>> sbaf = compute_sbaf_for_bands("LISS3", "B3", "S2B", "B04",
    ...                                srf_tgt_xml="/path/to/MTD_MSIL1C.xml")
    """
    from .cross_sensor_metrics import compute_sbaf

    wl = np.arange(wl_min, wl_max + step, step, dtype=np.float64)

    if custom_spectrum is not None and custom_wavelengths is not None:
        wl_cust = np.asarray(custom_wavelengths, dtype=np.float64)
        spec_cust = np.asarray(custom_spectrum, dtype=np.float64)
        ref_spec = np.interp(wl, wl_cust, spec_cust, left=0.0, right=0.0)
    else:
        ref_spec = get_reference_spectrum(spectrum, wl)

    srf_ref = get_srf(sensor_ref, band_ref, wl, xml_path=srf_ref_xml)
    srf_tgt = get_srf(sensor_tgt, band_tgt, wl, xml_path=srf_tgt_xml)

    return compute_sbaf(ref_spec, wl, srf_ref, srf_tgt)


import re as _re
_LISS_VARIANT_PATTERN = _re.compile(r"LISS[-_]?([34])", _re.IGNORECASE)


def _resolve_pipeline_sensor(sensor_family: str, metadata: dict, info: dict) -> str:
    """
    Convert a pipeline sensor identifier to an SRF library sensor key.

    Two forms arrive here depending on which calibration path ran:

    * High-level family (``detect_sensor()``):
      ``"RESOURCESAT2A"``, ``"SENTINEL2"``, ``"LANDSAT"``.
    * Instrument code (``get_sensor()`` reading BAND_META.txt ``Sensor`` field):
      ``"L3"`` (LISS-3), ``"L4"`` (LISS-4), ``"L4FX"`` (LISS-4 FX).
      These are stored directly in ``rad_details["sensor_src/tgt"]`` because
      ``get_band_calibration_full`` returns the BAND_META value verbatim.
    """
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(info, dict):
        info = {}

    _rs2a_map = {
        "L3":   "LISS3",  # LISS-III
        "L4":   "LISS4",  # LISS-IV (standard mode)
        "L4FX": "LISS4",  # LISS-IV FX -- same spectral bands, higher spatial res
    }
    if sensor_family.upper() in _rs2a_map:
        return _rs2a_map[sensor_family.upper()]


    if sensor_family == "RESOURCESAT2A":
        haystack = f"{info.get('filepath', '')} {info.get('image_dir', '')}"
        m = _LISS_VARIANT_PATTERN.search(haystack)
        return f"LISS{m.group(1)}" if m else "LISS3"

    if sensor_family == "SENTINEL2":
        haystack = (f"{info.get('filepath', '')} "
                    f"{info.get('filename', '')} "
                    f"{info.get('image_dir', '')}")
        if _re.search(r"S2B", haystack, _re.IGNORECASE):
            return "S2B"
        return "S2A"

    if sensor_family == "LANDSAT":
        spacecraft_id = str(metadata.get("spacecraft_id", "")).upper()
        if "LANDSAT_9" in spacecraft_id or "LANDSAT9" in spacecraft_id:
            return "L9OLI2"
        return "L8OLI"

    raise KeyError(
        f"compute_auto_sbaf: unknown sensor identifier '{sensor_family}'. "
        f"Expected a BAND_META instrument code (L3, L4, L4FX) or a sensor "
        f"family (RESOURCESAT2A, SENTINEL2, LANDSAT)."
    )


def compute_auto_sbaf(rad_dict: dict) -> tuple:
    """
    Resolve sensor/band information from a pipeline ``rad_details`` dict and
    compute the SBAF automatically.

    This is the function called by ``pipeline.py`` when the caller has not
    supplied an explicit ``sbaf=`` value.  It extracts the sensor family and
    band names from the dict produced by
    ``radiometric_correction.compute_common_reflectance()``, maps them to
    the SRF library's sensor keys, and delegates to
    :func:`compute_sbaf_for_bands`.

    Parameters
    ----------
    rad_dict : dict
        A copy of the ``rad_details`` dict enriched with the extra keys that
        ``pipeline.py`` adds before calling this function:

        * ``sensor_src``, ``sensor_tgt``   -- pipeline sensor-family strings
          (``"RESOURCESAT2A"``, ``"SENTINEL2"``, ``"LANDSAT"``).
        * ``band_name_src``, ``band_name_tgt`` -- band labels (e.g. ``"B3"``,
          ``"B04"``, ``"B5"``).
        * ``metadata_src``, ``metadata_tgt`` -- metadata dicts (used to
          resolve Landsat 8 vs 9 from ``spacecraft_id``).
        * ``info_src``, ``info_tgt`` -- info dicts (used to detect LISS
          variant and Sentinel-2A/B from file paths).

    Returns
    -------
    (sbaf_factor, description) : (float, str)
        * ``sbaf_factor`` -- the SBAF multiplicative factor; pass directly
          to ``compute_cross_sensor_report(sbaf=...)``.
        * ``description`` -- human-readable string describing what was
          resolved and computed, for log output.

    Raises
    ------
    KeyError
        If a sensor family or band cannot be resolved to an SRF library key.
    ValueError
        Propagated from :func:`compute_sbaf_for_bands` if the SRF integral
        is zero or inputs are invalid.
    """
    sensor_src = rad_dict.get("sensor_src", "")
    sensor_tgt = rad_dict.get("sensor_tgt", "")
    band_src   = rad_dict.get("band_name_src", "")
    band_tgt   = rad_dict.get("band_name_tgt", "")
    meta_src   = rad_dict.get("metadata_src") or {}
    meta_tgt   = rad_dict.get("metadata_tgt") or {}
    info_src   = rad_dict.get("info_src") or {}
    info_tgt   = rad_dict.get("info_tgt") or {}

    srf_key_src = _resolve_pipeline_sensor(sensor_src, meta_src, info_src)
    srf_key_tgt = _resolve_pipeline_sensor(sensor_tgt, meta_tgt, info_tgt)

    srf_ref_xml = rad_dict.get("meta_path_src") if sensor_src == "SENTINEL2" else None
    srf_tgt_xml = rad_dict.get("meta_path_tgt") if sensor_tgt == "SENTINEL2" else None

    sbaf = compute_sbaf_for_bands(
        srf_key_src, band_src, srf_key_tgt, band_tgt,
        srf_ref_xml=srf_ref_xml, srf_tgt_xml=srf_tgt_xml,
    )
    _srf_note = []
    if srf_ref_xml is not None:
        _srf_note.append(f"{srf_key_src} measured RSR")
    if srf_tgt_xml is not None:
        _srf_note.append(f"{srf_key_tgt} measured RSR")
    _srf_desc_suffix = f", {' + '.join(_srf_note)}" if _srf_note else ""
    desc = (
        f"{srf_key_src}/{band_src} â†’ {srf_key_tgt}/{band_tgt} "
        f"(dry_grass spectrum{_srf_desc_suffix}) = {sbaf:.4f}"
    )
    return sbaf, desc


def sbaf_table(
    sensor_ref: str,
    band_ref: str,
    target_sensors: list[str] | None = None,
    spectra: list[str] | None = None,
) -> dict:
    """
    Compute SBAF for one reference band against multiple targets and spectra.

    Useful for a quick sensitivity analysis across land-cover types.

    Parameters
    ----------
    sensor_ref, band_ref : str
        The reference sensor/band (the "truth" side).
    target_sensors : list[str], optional
        List of (sensor, band) tuples to compare against.
        Defaults to all sensors with a matching band alias.
    spectra : list[str], optional
        Reference spectra to loop over.  Defaults to all four built-ins.

    Returns
    -------
    dict
        Nested dict  {(sensor_tgt, band_tgt): {spectrum_name: sbaf_value}}.
    """
    if spectra is None:
        spectra = list(_REFERENCE_SPECTRA.keys())

    if target_sensors is None:
        band_alias = band_ref.lower()
        target_sensors = []
        for s in _SENSORS:
            if s == sensor_ref.upper():
                continue
            try:
                list_bands(s)  # just to check sensor exists
                get_srf(s, band_alias, np.array([600.0]))  # test alias works
                target_sensors.append((s, band_alias))
            except KeyError:
                pass

    results = {}
    for tgt in target_sensors:
        s_tgt, b_tgt = tgt
        results[(s_tgt, b_tgt)] = {}
        for sp in spectra:
            try:
                val = compute_sbaf_for_bands(sensor_ref, band_ref,
                                             s_tgt, b_tgt, spectrum=sp)
                results[(s_tgt, b_tgt)][sp] = val
            except (ValueError, KeyError) as exc:
                results[(s_tgt, b_tgt)][sp] = f"ERROR: {exc}"

    return results


if __name__ == "__main__":
    print("=== SRF Library Self-Test ===\n")

    wl = np.arange(400, 2501, 1, dtype=np.float64)

    print("Sensors available:", list_sensors())
    for s in list_sensors():
        print(f"  {s}: {list_bands(s)}")
    print()

    sbaf_same = compute_sbaf_for_bands("LISS3", "B3", "LISS4", "B3")
    print(f"LISS-3 Red vs LISS-4 Red  (dry_grass):  {sbaf_same:.6f}  "
          f"[expect ~1.000 -- identical band limits]")

    for sp in ("flat", "dry_grass", "bare_soil", "green_veg"):
        sbaf_val = compute_sbaf_for_bands("LISS3", "B3", "S2A", "B04", spectrum=sp)
        print(f"LISS-3 B3 vs S2A B04  ({sp:12s}):  {sbaf_val:.6f}")
    print()

    for sp in ("flat", "dry_grass", "bare_soil", "green_veg"):
        sbaf_val = compute_sbaf_for_bands("LISS3", "B3", "L8OLI", "B4", spectrum=sp)
        print(f"LISS-3 B3 vs L8 B4    ({sp:12s}):  {sbaf_val:.6f}")
    print()

    print("Sensitivity table: LISS-3 Red vs all matching Red bands")
    tbl = sbaf_table("LISS3", "red")
    for key, sp_vals in tbl.items():
        s, b = key
        vals_str = "  ".join(f"{sp}={v:.4f}" for sp, v in sp_vals.items()
                             if not str(v).startswith("ERROR"))
        print(f"  vs {s:10s} {b}: {vals_str}")
