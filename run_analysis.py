#!/usr/bin/env python3
"""
Master Entry Point for SpotiPy-MultiObs.
"""

import argparse
from spotipy_tool import pipeline, visualization

def main():
    parser = argparse.ArgumentParser(description="SpotiPy-MultiObs Analysis Tool")
    
    # Required Basics
    parser.add_argument("--noaa", type=int, required=True, help="NOAA Active Region Number")
    parser.add_argument("--date", type=str, required=True, help="Start Date (ISO Format)")
    parser.add_argument("--email", type=str, required=True, help="JSOC Email")
    parser.add_argument("--days", type=int, default=13, help="Duration in days")
    parser.add_argument("--cadence", type=int, default=6, help="Cadence in hours")

    # Granular Controls
    parser.add_argument("--download-only", action="store_true", help="Only download data, do no processing")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, assume data exists")
    parser.add_argument("--skip-align", action="store_true", help="Skip AIA alignment")
    parser.add_argument("--only-masks", action="store_true", help="Stop after generating masks (no data extraction)")
    
    # Observable Selection
    parser.add_argument("--observables", nargs="+", default=None, 
                        help="List specific observables to run (e.g. M V Ic). Default: All")
    
    # Visualization
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate plots")
    parser.add_argument("--recreate-masks", action="store_true", help="Force regeneration of mask overlays")

    args = parser.parse_args()

    # 1. Determine Pipeline Steps
    steps = []
    if not args.skip_download: steps.append('download')
    if not args.skip_align:    steps.append('align')
    
    # Logic: If download-only, we stop there.
    if args.download_only:
        steps = ['download']
    else:
        # If not download only, we add masking
        steps.append('masks')
        # If not only-masks, we add extraction
        if not args.only_masks:
            steps.append('extract')

    # 2. Run Pipeline
    print(f"--- Starting Pipeline for NOAA {args.noaa} ---")
    print(f"Steps: {steps}")
    print(f"Observables: {args.observables if args.observables else 'ALL'}")
    
    root_dir = pipeline.analyze_sunspot(
        noaa_number=args.noaa,
        start_time_str=args.date,
        duration_days=args.days,
        cadence_hours=args.cadence,
        email=args.email,
        steps=steps,
        observables_list=args.observables,
        recreate_masks=args.recreate_masks
    )

    # 3. Run Visualization (if data was extracted)
    if 'extract' in steps and not args.skip_plots:
        print("\n--- Generating Visualization ---")
        visualization.generate_plots(root_dir, obs_list=args.observables)
    
    print("\n[DONE] Run complete.")

if __name__ == "__main__":
    main()
