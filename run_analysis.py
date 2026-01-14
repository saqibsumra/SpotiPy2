#!/usr/bin/env python3
"""
SpotiPy-MultiObs: Solar Active Region Analysis Pipeline
=======================================================

Description:
    This script serves as the primary execution interface for the SpotiPy analysis suite.
    It automates the end-to-end workflow for studying the Center-to-Limb Variation (CLV)
    of various solar observables across the solar disk.

    The pipeline integrates data from SDO/HMI and SDO/AIA to perform a comprehensive,
    multi-wavelength statistical analysis of sunspots and their surrounding environments.

    Workflow Stages:
    1.  **Data Acquisition:** Automated retrieval of co-temporal HMI and AIA data series
        directly from the Joint Science Operations Center (JSOC).
    2.  **Pre-processing:** precise co-alignment of multi-instrument data and photometric
        correction (specifically, the removal of limb-darkening effects in UV imagery).
    3.  **Segmentation & Extraction:** Isolation of specific solar features (Umbra, Penumbra,
        Plage, Network, Quiet Sun) to extract physical quantities:
        - Continuum Intensity (Ic)
        - Magnetograms (M)
        - Dopplergrams (V)
        - Line Depth (Ld) & Line Width (Lw)
    4.  **Statistical Visualization:** Generation of spatially-resolved "Candle" (box-and-whisker)
        plots to investigate CLV trends and potential East/West hemispheric asymmetries.

Usage Examples:
    1. Interactive Mode (Standard):
       python3 run_analysis.py --config params.txt

    2. Automated Batch Mode (e.g., specific observables, no questions):
       python3 run_analysis.py --config params.txt --observables M V --auto

    3. Visualization Mode (Skip processing, regenerate plots only):
       python3 run_analysis.py --config params.txt --plot-only

    4. Manual Override (Test specific NOAA/Date without editing config):
       python3 run_analysis.py --config params.txt --noaa 12673 --days 1

Author: Muhammad Saqib Sumra
Date:   2026
"""

import argparse
import sys
from spotipy_tool import pipeline, visualization

def ask_yn(msg: str) -> bool:
    """
    Interactively asks the user for a Yes/No response.
    Used to control the flow of the pipeline step-by-step.
    """
    while True:
        ans = input(f"{msg} (y/n): ").strip().lower()
        if ans in {"y", "yes"}: return True
        if ans in {"n", "no"}: return False
        print("Please answer 'y' or 'n'.")

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="SpotiPy-MultiObs Analysis Tool")
    
    # Configuration File (Essential)
    parser.add_argument("--config", type=str, default=None, help="Path to params.txt configuration file")
    
    # Execution Flags
    parser.add_argument("--auto", action="store_true", help="Run in automated mode (skip all interactive questions)")
    parser.add_argument("--plot-only", action="store_true", help="Skip analysis and only generate plots from existing results")
    
    # Overrides (Optional command-line arguments to override config file)
    parser.add_argument("--noaa", type=int, help="Override NOAA Active Region Number")
    parser.add_argument("--date", type=str, help="Override Start Date (ISO format)")
    parser.add_argument("--email", type=str, help="Override JSOC Email")
    parser.add_argument("--days", type=int, help="Override Duration in days")
    parser.add_argument("--cadence", type=int, help="Override Cadence in hours")
    
    # Advanced Options
    parser.add_argument("--observables", nargs="+", default=None, help="Process specific observables only (e.g., M V)")
    parser.add_argument("--recreate-masks", action="store_true", help="Force regeneration of mask overlay images")

    args = parser.parse_args()
    
    print(f"--- SpotiPy Analysis ---")
    if args.config: print(f"Using Config File: {args.config}")

    steps = []

    # --- Step Selection Logic ---
    if args.plot_only:
        print("[INFO] Plot-Only Mode Enabled. Skipping data processing.")
    else:
        # 1. Download: Fetches FITS files from JSOC
        if args.auto: do_dl = True
        else: do_dl = ask_yn("Download data?")
        if do_dl: steps.append('download')

        # 2. Alignment: Reprojects AIA images to match HMI geometry
        if args.auto: do_align = True
        else: do_align = ask_yn("Align AIA?")
        if do_align: steps.append('align')

        # 3. Limb Darkening: Removes center-to-limb variation from AIA images
        if args.auto: do_ld = True
        else: do_ld = ask_yn("Correct AIA LD?")
        if do_ld: steps.append('aia_ld')

        # 4. HMI Analysis: Extracts data using HMI Intensity masks (Umbra/Penumbra)
        if args.auto: do_hmi = True
        else: do_hmi = ask_yn("Run HMI Analysis?")
        if do_hmi: steps.append('hmi')

        # 5. AIA Analysis: Extracts data using AIA masks (Plage/Network/QS)
        if args.auto: do_aia = True
        else: do_aia = ask_yn("Run AIA Analysis?")
        if do_aia: steps.append('aia')

    # Mask Recreation: Optional visualization of the segmentation masks
    recreate = args.recreate_masks
    if not args.auto and not args.plot_only and (('hmi' in steps) or ('aia' in steps)) and not recreate:
        recreate = ask_yn("Recreate/Overwrite Mask Overlays?")

    # --- Execution Phase ---
    try:
        # Call the main pipeline engine
        root_dir = pipeline.analyze_sunspot(
            noaa_number=args.noaa,
            start_time_str=args.date,
            duration_days=args.days,
            cadence_hours=args.cadence,
            email=args.email,
            config_file=args.config,
            steps=steps,
            observables_list=args.observables,
            recreate_masks=recreate
        )

        # Call the visualization engine
        print("\n--- Visualizing ---")
        visualization.generate_plots(root_dir, obs_list=args.observables)
            
        print("\n[DONE] Run complete.")

    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Tip: Ensure params.txt is correct and contains all required fields.")
        sys.exit(1)

if __name__ == "__main__":
    main()
