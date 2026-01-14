"""
SpotiPy: Data Reduction & Analysis Module
==================================================

Description:
    This module implements the core scientific algorithms for processing solar imagery 
    from SDO/HMI and SDO/AIA. It serves as the computational backbone of the analysis suite, 
    transforming raw FITS data into spatially-resolved physical observables.

    The pipeline performs the following critical scientific tasks:
    1.  **Data Ingestion:** Automated interfacing with the JSOC to retrieve high-fidelity 
        scientific data products (Intensity, Magnetograms, Dopplergrams).
    2.  **Image Co-registration:** Precise spatial alignment of UV (AIA) imagery to the 
        visible-light (HMI) coordinate system using WCS reprojection.
    3.  **Photometric Correction:** Empirical modeling and removal of Center-to-Limb Variation 
        (Limb Darkening) from UV images to isolate intrinsic brightness variations.
    4.  **Morphological Segmentation:** Application of intensity thresholds and morphological 
        operators to robustly classify solar features (Umbra, Penumbra, Plage, Network, Quiet Sun).
    5.  **Data Extraction:** Sampling of calibrated physical quantities (Intensity, Magnetic Field, 
        Doppler Velocity) within defined solar regions for statistical analysis.

Usage:
    This module is designed to be imported and executed via the `run_analysis.py` interface.
    The primary entry point is the `analyze_sunspot()` function.

Author: Muhammad Saqib Sumra
Date:   2026
"""

import os
import sys
import ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

import cv2
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.wcs import WCS
import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import RotatedSunFrame
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from reproject import reproject_interp
from scipy.ndimage import label, center_of_mass
from scipy import ndimage as ndi

warnings.filterwarnings('ignore')

# =====================================
# 1. CONFIGURATION & DEFAULTS
# =====================================
CONFIG = {
    # Geometry defaults (can be overridden by params.txt)
    "X_START_ARCSEC": -927,
    "Y_START_ARCSEC": 290,
    "FRAME_SIZE": 400,
    "SPOT_CENTER_THRESH": 0.98, # High threshold to keep spot centered even at limb
    
    # Segmentation Thresholds
    "UMBRA_RANGE": (10, 55),
    "PENUMBRA_RANGE": (75, 120),
    "QUIET_SUN_TOL_PCT": 15.0, # Network is > 15% brighter than QS median
    "PLAGE_EXCESS_PCT": 20.0,  # Plage is > 20% brighter
    "MIN_PLAGE_AREA": 450,
    "BLUR_SIGMA": 3,
    
    # Fitting Physics
    "MIN_MU_FOR_FIT": 0.15,
    "HMI_POLY_ORDER": 2,
    "POLY_ORDER": 5,

    # Runtime place holders
    "NOAA_NUMBER": None,
    "START_DATE": None,
    "DAYS": 13,
    "CADENCE": 6,
    "EMAIL": None
}

# Mapping short names to JSOC series names
ALL_OBSERVABLES = {
    "Ic": "hmi.Ic_720s", "M" : "hmi.M_720s", "V" : "hmi.V_720s",
    "Ld": "hmi.Ld_720s", "Lw": "hmi.Lw_720s"
}
HMI_NO_LD_SERIES = "hmi.Ic_noLimbDark_720s"
AIA_SERIES       = "aia.lev1_uv_24s"

def load_config(path):
    """Loads configuration parameters from a formatted text file."""
    if not os.path.exists(path): return
    with open(path, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line or "=" not in line: continue
            key, val_str = line.split("=", 1)
            key = key.strip().upper(); val_str = val_str.strip()
            if key in CONFIG:
                try:
                    val = ast.literal_eval(val_str) # Safely evaluate numbers/tuples
                    CONFIG[key] = val
                except:
                    CONFIG[key] = val_str

# =====================================
# 2. UTILS & FILE I/O
# =====================================
def file_ok(path):
    """Verifies that a file exists and contains data."""
    return os.path.exists(path) and os.path.getsize(path) > 0

def save_list(list_path, dir_path):
    """Generates a text manifest of FITS files in a directory."""
    if not os.path.isdir(dir_path): return
    files = [f for f in sorted(os.listdir(dir_path))
             if f.endswith('.fits') and '_spotmask' not in f and '_segmentation' not in f]
    with open(list_path, 'w') as fh:
        for f in files: fh.write(f+"\n")

def read_fits(path):
    """Reads FITS data and header, robust to compressed (index 1) or standard (index 0) formats."""
    try:
        with fits.open(path) as hdul:
            if len(hdul) > 1 and hdul[1].data is not None:
                return hdul[1].data, hdul[1].header
            return hdul[0].data, hdul[0].header
    except Exception: return None, None

def center_radius_from_header(hdr):
    """Parses solar geometry (CRPIX, RSUN_OBS) from the FITS header."""
    cx = hdr.get('CRPIX1'); cy = hdr.get('CRPIX2')
    dx = hdr.get('CDELT1'); dy = hdr.get('CDELT2')
    rs = hdr.get('RSUN_OBS')
    r_pix = None
    if rs is not None and dx not in (None,0): r_pix = rs/ dx
    return (cx,cy), r_pix, dx, dy

def _parse_dateobs(s):
    """Robustly parses FITS DATE-OBS strings."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f","%Y-%m-%dT%H:%M:%S"):
        try: return datetime.strptime(s, fmt)
        except: pass
    return None

def find_closest_file(target_hdr, dir_path):
    """Identifies the file in a directory that is temporally closest to the target header."""
    if target_hdr is None or 'DATE-OBS' not in target_hdr: return None
    t_ref = _parse_dateobs(target_hdr['DATE-OBS'])
    if t_ref is None: return None
    best, best_dt = None, 9e99
    if not os.path.isdir(dir_path): return None
    for fn in sorted(os.listdir(dir_path)):
        if not fn.endswith('.fits'): continue
        _, hh = read_fits(os.path.join(dir_path, fn))
        if hh and 'DATE-OBS' in hh:
            dt = abs((_parse_dateobs(hh['DATE-OBS'])-t_ref).total_seconds())
            if dt < best_dt: best_dt, best = dt, fn
    return best

# =====================================
# 3. DOWNLOAD & GEOMETRY
# =====================================
def download_series(series, start_time, days, cadence_h, out_dir, list_path, email):
    """Queries and retrieves data series from the JSOC via SunPy Fido."""
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    steps = int(days*24/cadence_h)
    t0 = start_time; t1 = start_time + timedelta(hours=cadence_h*steps)
    print(f"   [Query] {series}: {t0.iso} to {t1.iso}")
    
    search_attrs = [a.Time(t0.iso, t1.iso), a.jsoc.Series(series),
                    a.Sample(cadence_h*u.hour), a.jsoc.Notify(email)]
    if "aia" in series: search_attrs.append(a.Wavelength(1700*u.angstrom))
    
    res = Fido.search(*search_attrs)
    if len(res) > 0: Fido.fetch(res, path=os.path.join(out_dir, "{file}"))
    save_list(list_path, out_dir)

def download_single_aia_rescue(target_time, email):
    """Rescue Mechanism: Fetches a single missing AIA file if alignment fails."""
    print(f"   [RESCUE] Downloading AIA for {target_time.iso}...")
    t_start = target_time - timedelta(minutes=1)
    t_end   = target_time + timedelta(minutes=1)
    try:
        search_attrs = [
            a.Time(t_start.iso, t_end.iso), a.jsoc.Series(AIA_SERIES),
            a.jsoc.Notify(email), a.Wavelength(1700*u.angstrom)
        ]
        res = Fido.search(*search_attrs)
        if len(res) > 0:
            files = Fido.fetch(res[0, 0], path=None) 
            if len(files) > 0: return files[0]
    except Exception as e:
        print(f"   [RESCUE FAIL] {e}")
    return None

def align_aia_to_hmi(aia_path, hmi_path, out_path):
    """
    Performs image co-registration.
    Reprojects the AIA image onto the HMI WCS coordinate system using interpolation.
    """
    if file_ok(out_path): return True
    a_dat, a_hdr = read_fits(aia_path); h_dat, h_hdr = read_fits(hmi_path)
    if a_dat is None or h_dat is None: return False
    A = WCS(a_hdr); H = WCS(h_hdr)
    reproj, _ = reproject_interp((a_dat, A), H, shape_out=h_dat.shape)
    new_hdr = h_hdr.copy(); new_hdr['HISTORY'] = 'AIA reprojected to HMI grid'
    fits.writeto(out_path, reproj, new_hdr, overwrite=True)
    return True

def radial_profile(data, center, rmax, mask=None):
    """Computes the median intensity as a function of solar radius (for limb darkening modeling)."""
    yy, xx = np.indices(data.shape)
    r = np.sqrt((xx-center[0])**2 + (yy-center[1])**2).astype(int)
    disk = r < int(rmax)
    if mask is not None: disk &= (~mask) # Exclude active regions from profile calculation
    arr = np.ma.masked_array(data, mask=~disk)
    tbin = np.bincount(r[disk], weights=arr[disk])
    prof = tbin/np.maximum(np.bincount(r[disk]),1)
    if prof.size>20:
        med = np.median(prof[:20]); # Normalize by center
        if med>0: prof = prof/med
    return prof

def profile_to_plane(profile, center, shape, rmax):
    """Reconstructs a 2D synthetic image from a 1D radial profile."""
    yy, xx = np.indices(shape)
    r = np.sqrt((xx-center[0])**2 + (yy-center[1])**2).astype(int)
    interp = interp1d(np.arange(len(profile)), profile, bounds_error=False, fill_value='extrapolate')
    out = interp(r.ravel()).reshape(shape)
    out[r>int(rmax)] = np.nan
    return out

def remove_aia_ld(aia_aligned_path, out_nolbd_path):
    """
    Corrects photometric limb darkening by dividing the observed image 
    by its empirical radial profile.
    """
    if file_ok(out_nolbd_path): return True
    data, hdr = read_fits(aia_aligned_path)
    if data is None: return False
    (cx,cy), r_pix, dx, dy = center_radius_from_header(hdr)
    if r_pix is None: return False
    mask = np.zeros_like(data, dtype=bool)
    
    # Mask the active region so it doesn't skew the profile
    xs = int(round(cx - CONFIG["X_START_ARCSEC"]/dx))
    ys = int(round(cy - CONFIG["Y_START_ARCSEC"]/dy))
    h = CONFIG["FRAME_SIZE"] // 2
    x0,x1 = max(0,xs-h), min(data.shape[1], xs+h); y0,y1 = max(0,ys-h), min(data.shape[0], ys+h)
    mask[y0:y1, x0:x1] = True
    
    prof = radial_profile(data, (cx,cy), int(r_pix), mask=mask)
    plane = profile_to_plane(prof, (cx,cy), data.shape, int(r_pix))
    
    with np.errstate(divide='ignore',invalid='ignore'):
        corr = data/np.where(np.isfinite(plane)&(plane!=0), plane, 1.0)
    
    fits.writeto(out_nolbd_path, corr, hdr, overwrite=True)
    return True

# =====================================
# 4. CENTERING & MASKS
# =====================================
def compute_centered_window_hmi(img, hdr, x_hint, y_start_arcsec):
    """
    Determines the optimal crop window centered on the sunspot.
    Uses differential rotation tracking ('x_hint') and intensity centroiding.
    """
    H, W = img.shape[:2]
    frame_size = CONFIG["FRAME_SIZE"]
    cy = hdr.get('CRPIX2'); dy = hdr.get('CDELT2')
    y_pix = int(round(cy - (y_start_arcsec / dy))) if (cy and dy) else H//2
    x_pix = int(round(x_hint))
    x0 = int(np.clip(x_pix - frame_size//2, 0, W - frame_size))
    y0 = int(np.clip(y_pix - frame_size//2, 0, H - frame_size))
    
    crop = img[y0:y0+frame_size, x0:x0+frame_size]
    med = np.nanmedian(crop)
    if med > 0:
        # Refine center using intensity threshold
        thresh = CONFIG.get("SPOT_CENTER_THRESH", 0.98) 
        mask = crop <= (thresh * med)
        lab, nlab = label(mask)
        if nlab > 0:
            sizes = np.bincount(lab.ravel()); sizes[0]=0
            cyc, cxc = center_of_mass(lab == int(np.argmax(sizes)))
            if np.isfinite(cxc) and np.isfinite(cyc):
                # Only shift if detected spot is within reasonable bounds
                shift_x = cxc - frame_size/2; shift_y = cyc - frame_size/2
                if abs(shift_x) < 100 and abs(shift_y) < 100:
                    x0 = int(np.clip(x0 + shift_x, 0, W-frame_size))
                    y0 = int(np.clip(y0 + shift_y, 0, H-frame_size))
    return x0, y0

def mu_grid_for_crop(shape, hdr, x0, y0):
    """Calculates the Mu (cosine theta) value for every pixel in the crop."""
    (cx,cy), r_pix, dx, dy = center_radius_from_header(hdr)
    if r_pix is None: return np.full(shape, np.nan)
    h,w = shape; yy,xx = np.indices((h,w))
    X = x0+xx; Y=y0+yy
    return np.sqrt(np.clip(1.0 - (np.sqrt((X-cx)**2 + (Y-cy)**2)/r_pix)**2, 0.0, 1.0))

def coords_grid_for_crop(shape, hdr, x0, y0):
    """Calculates Helioprojective coordinates (arcsec) for every pixel in the crop."""
    (cx,cy), r_pix, dx, dy = center_radius_from_header(hdr)
    if r_pix is None: return np.full(shape, np.nan), np.full(shape, np.nan)
    h, w = shape; yy, xx = np.indices((h,w))
    X_pix = x0 + xx; Y_pix = y0 + yy
    X_arc = (X_pix - cx) * dx; Y_arc = (Y_pix - cy) * dy
    return X_arc, Y_arc

def make_uint8_for_cv(img, vmin=0.0, vmax=2.0):
    """Normalizes image data to 8-bit integers for OpenCV processing."""
    arr = np.clip(img.astype(float), vmin, vmax)
    norm = (arr - vmin) / (max(vmax - vmin, 1e-6))
    return cv2.cvtColor((norm * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

def build_hmi_masks(crop_img, disk_mask_raw, disk_mask_thresh):
    """
    Performs morphological segmentation on HMI images.
    Classifies pixels into Umbra, Penumbra, and Full Spot based on intensity thresholds.
    """
    u8 = make_uint8_for_cv(crop_img, vmin=0.0, vmax=2.0)
    g  = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)
    if disk_mask_thresh is None: disk_mask_thresh = disk_mask_raw
    g[~disk_mask_thresh] = 255
    g_blur = cv2.GaussianBlur(g, (7,7), 0)
    u_range = CONFIG["UMBRA_RANGE"]; p_range = CONFIG["PENUMBRA_RANGE"]
    
    # Umbra
    u_raw = cv2.inRange(g_blur, int(u_range[0]), int(u_range[1]))
    u = cv2.erode(u_raw, np.ones((3,3),np.uint8), iterations=1)
    
    # Penumbra
    p_band = cv2.inRange(g_blur, int(p_range[0]), int(p_range[1]))
    p = cv2.bitwise_and(p_band, cv2.bitwise_not(cv2.dilate(u, np.ones((7,7),np.uint8))))
    
    # Full Spot
    f_loose = cv2.morphologyEx(cv2.inRange(g_blur,0,int(p_range[1])), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    f_spot = cv2.bitwise_or(f_loose, u)
    return (u>0)&disk_mask_raw.astype(bool), (p>0)&disk_mask_raw.astype(bool), (f_spot>0)&disk_mask_raw.astype(bool)

def clean_spot_mask(mask_bool):
    """Retains only the largest connected component (main sunspot), filtering noise."""
    if mask_bool is None or not np.any(mask_bool): return mask_bool
    lbl, n = label(mask_bool)
    if n<=1: return mask_bool
    sizes = np.bincount(lbl.ravel()); sizes[0]=0
    return (lbl == np.argmax(sizes))

def polar_slice_mask(shape, center, r_pix, rmin, rmax, th0, th1):
    """Generates a polar wedge mask for sampling Quiet Sun statistics."""
    H, W = shape; yy, xx = np.indices((H, W))
    X = xx - center[0]; Y = yy - center[1]
    r_norm = np.sqrt(X**2 + Y**2) / float(r_pix)
    th = (np.degrees(np.arctan2(Y, X)) + 360) % 360
    return (r_norm >= rmin) & (r_norm <= rmax) & (th >= th0) & (th <= th1)

def collect_I_mu(img, mu, xg, yg, mask):
    """Extracts physical values (Intensity, Mu, X, Y) from masked pixels."""
    v = img[mask]; m = mu[mask]; x = xg[mask]; y = yg[mask]
    valid = np.isfinite(v) & np.isfinite(m)
    return m[valid], v[valid], x[valid], y[valid]

# =====================================
# 6. MAIN PIPELINE LOGIC
# =====================================
def analyze_sunspot(noaa_number=None, start_time_str=None, duration_days=None, 
                    cadence_hours=None, email=None, config_file=None, 
                    steps=['download', 'align', 'aia_ld', 'hmi', 'aia'], 
                    observables_list=None, recreate_masks=False):
    """
    Executes the primary scientific data processing sequence.
    Executes selected steps: Download -> Alignment -> Correction -> Extraction.
    """
    # Load config and override
    if config_file: load_config(config_file)
    noaa = noaa_number if noaa_number else CONFIG.get("NOAA_NUMBER")
    t_str = start_time_str if start_time_str else CONFIG.get("START_DATE")
    days = duration_days if duration_days else CONFIG.get("DAYS")
    cad = cadence_hours if cadence_hours else CONFIG.get("CADENCE")
    mail = email if email else CONFIG.get("EMAIL")

    if not noaa or not t_str or not mail:
        raise ValueError("Missing required parameters (NOAA, Date, or Email).")

    start_time = Time(t_str)
    ROOT = f"NOAA_{noaa}-1700A_dt_{cad}h"
    obs_to_run = observables_list if observables_list else ALL_OBSERVABLES.keys()
    
    # Create Directories
    DIRS = {
        "hmi_nold": os.path.join(ROOT, "FITS_files_HMI_noLD"),
        "aia"     : os.path.join(ROOT, "FITS_files_AIA"),
        "mask_hmi": os.path.join(ROOT, "masks_HMI"),
        "mask_aia": os.path.join(ROOT, "masks_AIA"),
        "ld_hmi"  : os.path.join(ROOT, "Results_HMI"),
        "plg_aia" : os.path.join(ROOT, "Results_AIA"),
    }
    for k in ALL_OBSERVABLES:
        DIRS[f"fits_{k}"] = os.path.join(ROOT, f"FITS_files_HMI_{k}")
        DIRS[f"res_hmi_{k}"] = os.path.join(DIRS["ld_hmi"], k)
        DIRS[f"res_aia_{k}"] = os.path.join(DIRS["plg_aia"], k)
    for p in DIRS.values(): os.makedirs(p, exist_ok=True)
    
    LISTS = {
        "hmi_nold": os.path.join(ROOT, f"NOAA_{noaa}_{HMI_NO_LD_SERIES}_files.txt"),
        "aia"     : os.path.join(ROOT, f"NOAA_{noaa}_{AIA_SERIES}_files.txt"),
        "aia_corr": os.path.join(ROOT, f"NOAA_{noaa}_{AIA_SERIES}_corrected_files.txt"),
    }
    for k, s in ALL_OBSERVABLES.items(): LISTS[k] = os.path.join(ROOT, f"NOAA_{noaa}_{s}_files.txt")

    # --- STEP 1: DOWNLOAD ---
    if 'download' in steps:
        print("[STEP] Downloading Data...")
        download_series(HMI_NO_LD_SERIES, start_time, days, cad, DIRS['hmi_nold'], LISTS['hmi_nold'], mail)
        download_series(AIA_SERIES, start_time, days, cad, DIRS['aia'], LISTS['aia'], mail)
        for k in obs_to_run:
            download_series(ALL_OBSERVABLES[k], start_time, days, cad, DIRS[f'fits_{k}'], LISTS[k], mail)
    else:
        save_list(LISTS['hmi_nold'], DIRS['hmi_nold'])
        save_list(LISTS['aia'], DIRS['aia'])

    # --- STEP 2: ALIGNMENT ---
    if 'align' in steps:
        print("[STEP] Aligning AIA...")
        aia_files = [f.strip() for f in open(LISTS['aia']).read().splitlines()]
        for i, f in enumerate(aia_files):
            # Find closest HMI Intensity file to align to
            closest_hmi = find_closest_file(read_fits(os.path.join(DIRS['aia'], f))[1], DIRS['fits_Ic'])
            if closest_hmi:
                align_aia_to_hmi(os.path.join(DIRS['aia'], f), os.path.join(DIRS['fits_Ic'], closest_hmi),
                                 os.path.join(DIRS['aia'], f.replace('.fits','_aligned.fits')))
    
    # --- STEP 3: LIMB DARKENING REMOVAL ---
    if 'aia_ld' in steps:
        print("[STEP] Correcting AIA Limb Darkening...")
        aia_files = [f.strip() for f in open(LISTS['aia']).read().splitlines()]
        for i, f in enumerate(aia_files):
            # Must use the ALIGNED file to generate the corrected file
            aligned_file = os.path.join(DIRS['aia'], f.replace('.fits','_aligned.fits'))
            if os.path.exists(aligned_file):
                remove_aia_ld(aligned_file, os.path.join(DIRS['aia'], f.replace('.fits','_nolbd.fits')))
        save_list(LISTS['aia_corr'], DIRS['aia'])

    # --- ANALYSIS PREPARATION ---
    # Need tracking logic even if we are only running analysis
    hmi_files = [f.strip() for f in open(LISTS['hmi_nold']).read().splitlines()]
    if not hmi_files: return ROOT

    # Calculate Solar Rotation Tracking
    d0,h0 = read_fits(os.path.join(DIRS['hmi_nold'], hmi_files[0]))
    h0['cunit1']='arcsec'; h0['cunit2']='arcsec'
    mp = sunpy.map.Map(d0,h0)
    p0 = SkyCoord(CONFIG["X_START_ARCSEC"]*u.arcsec, CONFIG["Y_START_ARCSEC"]*u.arcsec, frame=mp.coordinate_frame)
    ts=[]; 
    for fn in hmi_files:
        _,h=read_fits(os.path.join(DIRS['hmi_nold'], fn))
        if h and 'DATE-OBS' in h: ts.append(Time(h['DATE-OBS']))
    ts.sort(); dts=[0.0]+[(ts[i]-ts[i-1]).value for i in range(1,len(ts))]
    rpts = SkyCoord(RotatedSunFrame(base=p0, duration=np.cumsum(np.array(dts)*u.day)))
    pts  = rpts.transform_to(mp.coordinate_frame).to_string(unit='arcsec')
    x_arc_list = np.array([float(s.split()[0]) for s in pts])
    x_tracks = h0.get('CRPIX1') - (x_arc_list/h0.get('CDELT1'))

    # Data Containers
    HMI_DATA = {key: {'U': [[],[],[],[]], 'P': [[],[],[],[]], 'F': [[],[],[],[]]} for key in obs_to_run}
    AIA_DATA = {key: {'Q': [[],[],[],[]], 'Plage': [[],[],[],[]], 'Network': [[],[],[],[]]} for key in obs_to_run}

    # --- STEP 4: HMI ANALYSIS ---
    if 'hmi' in steps:
        print("   [HMI] Processing...")
        for i, fn in enumerate(hmi_files):
            try:
                path_nold = os.path.join(DIRS['hmi_nold'], fn)
                img_nold, hdr_nold = read_fits(path_nold)
                if img_nold is None: continue

                # Center on Sunspot
                x_hint = x_tracks[i] if i < len(x_tracks) else img_nold.shape[1]//2
                x0, y0 = compute_centered_window_hmi(img_nold, hdr_nold, x_hint, CONFIG["Y_START_ARCSEC"])
                
                # Load Geometry (Mu, X, Y)
                path_ic = os.path.join(DIRS['fits_Ic'], find_closest_file(hdr_nold, DIRS['fits_Ic']))
                _, hdr_ic = read_fits(path_ic)
                mu = np.rot90(mu_grid_for_crop((CONFIG["FRAME_SIZE"], CONFIG["FRAME_SIZE"]), hdr_ic, x0, y0), 2)
                x_grid, y_grid = coords_grid_for_crop((CONFIG["FRAME_SIZE"], CONFIG["FRAME_SIZE"]), hdr_ic, x0, y0)
                x_grid = np.rot90(x_grid, 2); y_grid = np.rot90(y_grid, 2)
                
                # Generate Masks
                crop_nold = np.nan_to_num(np.rot90(img_nold[y0:y0+CONFIG["FRAME_SIZE"], x0:x0+CONFIG["FRAME_SIZE"],], 2), nan=0.0)
                disk_mask = (np.isfinite(mu) & (mu > 0))
                disk_thresh = ndi.binary_erosion(disk_mask, structure=np.ones((3,3)), iterations=1)
                u_m, p_m, f_m = build_hmi_masks(crop_nold, disk_mask, disk_thresh)
                
                # Keep only largest spot
                u_m = clean_spot_mask(u_m); p_m = clean_spot_mask(p_m); f_m = clean_spot_mask(f_m)
                
                fits.writeto(os.path.join(DIRS['mask_hmi'], fn.replace('.fits', '_spotmask.fits')), 
                             f_m.astype(np.uint8), hdr_nold, overwrite=True)

                # Extract Observables
                for k in obs_to_run:
                    f_obs = find_closest_file(hdr_nold, DIRS[f'fits_{k}'])
                    if not f_obs: continue
                    img_obs, _ = read_fits(os.path.join(DIRS[f'fits_{k}'], f_obs))
                    crop_obs = np.nan_to_num(np.rot90(img_obs[y0:y0+CONFIG["FRAME_SIZE"], x0:x0+CONFIG["FRAME_SIZE"]], 2), nan=0.0)
                    
                    for cat, msk in [('U', u_m), ('P', p_m), ('F', f_m)]:
                        m, v, x, y = collect_I_mu(crop_obs, mu, x_grid, y_grid, msk)
                        HMI_DATA[k][cat][0].append(m); HMI_DATA[k][cat][1].append(v)
                        HMI_DATA[k][cat][2].append(x); HMI_DATA[k][cat][3].append(y)
            except Exception as e: print(f"[ERR-HMI] {fn}: {e}")
        
        # Save HMI Results
        print("   [Saving] Writing HMI text files...")
        for k in obs_to_run:
            for cat in ['U', 'P', 'F']:
                d = HMI_DATA[k][cat]
                if d[0]:
                    arr = np.column_stack([np.concatenate(d[0]), np.concatenate(d[1]), np.concatenate(d[2]), np.concatenate(d[3])])
                    np.savetxt(os.path.join(DIRS[f'res_hmi_{k}'], f'lbd_{cat}_raw.txt'), arr)

    # --- STEP 5: AIA ANALYSIS ---
    if 'aia' in steps:
        print("   [AIA] Processing...")
        for i, fn_hmi in enumerate(hmi_files):
            try:
                _, hdr_hmi = read_fits(os.path.join(DIRS['hmi_nold'], fn_hmi))
                t_hmi = Time(hdr_hmi['DATE-OBS'])
                
                # Find best matching AIA file
                best_aia, best_dt = None, 9999
                aia_list = [f for f in os.listdir(DIRS['aia']) if '_nolbd' in f]
                for fa in aia_list:
                    _, ha = read_fits(os.path.join(DIRS['aia'], fa))
                    dt = abs((Time(ha['DATE-OBS']) - t_hmi).to(u.s).value)
                    if dt < best_dt: best_dt, best_aia = dt, fa
                
                # Rescue logic if gap > 5 mins
                if best_dt > 300:
                    print(f"      Missing AIA for {fn_hmi}. Rescue download...")
                    raw_file = download_single_aia_rescue(t_hmi + timedelta(minutes=5), mail)
                    if raw_file:
                        base = os.path.basename(raw_file)
                        aligned = os.path.join(DIRS['aia'], base.replace('.fits','_aligned.fits'))
                        nolbd   = os.path.join(DIRS['aia'], base.replace('.fits','_nolbd.fits'))
                        align_aia_to_hmi(raw_file, os.path.join(DIRS['hmi_nold'], fn_hmi), aligned)
                        remove_aia_ld(aligned, nolbd)
                        best_aia = os.path.basename(nolbd)
                    else: continue

                path_aia = os.path.join(DIRS['aia'], best_aia)
                img_aia, hdr_aia = read_fits(path_aia)
                
                # Recalculate Geometry for this specific timestamp
                img_hmi, _ = read_fits(os.path.join(DIRS['hmi_nold'], fn_hmi))
                x_hint = x_tracks[i] if i < len(x_tracks) else img_hmi.shape[1]//2
                x0, y0 = compute_centered_window_hmi(img_hmi, hdr_hmi, x_hint, CONFIG["Y_START_ARCSEC"])
                
                crop_aia = np.rot90(img_aia[y0:y0+CONFIG["FRAME_SIZE"], x0:x0+CONFIG["FRAME_SIZE"]], 2)
                mu_aia = np.rot90(mu_grid_for_crop((CONFIG["FRAME_SIZE"], CONFIG["FRAME_SIZE"]), hdr_aia, x0, y0), 2)
                x_grid, y_grid = coords_grid_for_crop((CONFIG["FRAME_SIZE"], CONFIG["FRAME_SIZE"]), hdr_aia, x0, y0)
                x_grid = np.rot90(x_grid, 2); y_grid = np.rot90(y_grid, 2)
                
                # Load HMI Spot Mask
                mask_path = os.path.join(DIRS['mask_hmi'], fn_hmi.replace('.fits', '_spotmask.fits'))
                spot_mask = read_fits(mask_path)[0].astype(bool) if os.path.exists(mask_path) else np.zeros_like(crop_aia, dtype=bool)
                
                # Generate "Dirty Mask" to remove general activity
                crop_hmi = np.rot90(img_hmi[y0:y0+CONFIG["FRAME_SIZE"], x0:x0+CONFIG["FRAME_SIZE"]], 2)
                c_h_disp = crop_hmi.copy()
                if np.nanmedian(c_h_disp) > 100: c_h_disp /= np.nanmedian(c_h_disp)
                dirty_u8 = make_uint8_for_cv(c_h_disp, vmin=0.0, vmax=2.0)
                dirty_mask = cv2.inRange(cv2.cvtColor(dirty_u8, cv2.COLOR_BGR2GRAY), 0, 90) > 0
                data_nospot_qs = np.where(dirty_mask, np.nan, crop_aia)
                
                # Segmentation (Plage vs Network)
                (cx, cy), r_pix, _, _ = center_radius_from_header(hdr_aia)
                full_p_mask = polar_slice_mask((img_aia.shape[0], img_aia.shape[1]), (cx, cy), r_pix, 0.6, 0.95, 70.0, 110.0)
                slice_mask = np.rot90(full_p_mask[y0:y0+CONFIG["FRAME_SIZE"], x0:x0+CONFIG["FRAME_SIZE"]], 2)
                qs_vals = data_nospot_qs[slice_mask]
                qs_med = np.nanmedian(qs_vals) if qs_vals.size > 0 else np.nanmedian(data_nospot_qs)
                
                data_nospot = data_nospot_qs.copy(); data_nospot[spot_mask] = np.nan
                blurred = cv2.GaussianBlur(np.nan_to_num(data_nospot), (CONFIG["BLUR_SIGMA"], CONFIG["BLUR_SIGMA"]), 0)
                candidates = (blurred > qs_med * (1.0 + CONFIG["PLAGE_EXCESS_PCT"]/100.0)).astype(np.uint8)
                n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
                plage_mask = np.zeros_like(crop_aia, dtype=bool)
                for l in range(1, n_lbl):
                    if stats[l, cv2.CC_STAT_AREA] >= CONFIG["MIN_PLAGE_AREA"]: plage_mask[labels==l] = True
                    
                net_thresh = qs_med * (1.0 + CONFIG["QUIET_SUN_TOL_PCT"]/100.0)
                valid_for_net = (~dirty_mask) & np.isfinite(data_nospot) & (~plage_mask) & (~spot_mask)
                net_mask = (data_nospot > net_thresh) & valid_for_net
                qs_mask = (crop_aia > 0) & np.isfinite(crop_aia) & (~spot_mask) & (~plage_mask) & (~net_mask) & (~dirty_mask)
                
                # Extract Observables
                for k in obs_to_run:
                    f_obs = find_closest_file(hdr_hmi, DIRS[f'fits_{k}'])
                    if not f_obs: continue
                    img_obs, _ = read_fits(os.path.join(DIRS[f'fits_{k}'], f_obs))
                    crop_obs = np.nan_to_num(np.rot90(img_obs[y0:y0+CONFIG["FRAME_SIZE"], x0:x0+CONFIG["FRAME_SIZE"]], 2), nan=0.0)
                    
                    for cat, msk in [('Q', qs_mask), ('Plage', plage_mask), ('Network', net_mask)]:
                        m, v, x, y = collect_I_mu(crop_obs, mu_aia, x_grid, y_grid, msk)
                        AIA_DATA[k][cat][0].append(m); AIA_DATA[k][cat][1].append(v)
                        AIA_DATA[k][cat][2].append(x); AIA_DATA[k][cat][3].append(y)
            except Exception as e: print(f"[ERR-AIA] {fn_hmi}: {e}")

        # Save AIA Results
        print("   [Saving] Writing AIA text files...")
        for k in obs_to_run:
            for cat in ['Q', 'Plage', 'Network']:
                d = AIA_DATA[k][cat]
                if d[0]:
                    arr = np.column_stack([np.concatenate(d[0]), np.concatenate(d[1]), np.concatenate(d[2]), np.concatenate(d[3])])
                    np.savetxt(os.path.join(DIRS[f'res_aia_{k}'], f'lbd_{cat}_raw.txt'), arr)

    return ROOT
