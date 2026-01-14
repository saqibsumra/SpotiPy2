#!/usr/bin/env python3
"""
Master Entry Point for SpotiPy.
Interactive "Ask-and-Go" Workflow.
"""

import argparse
import sys
from spotipy_tool import pipeline, visualization

def ask_yn(msg: str) -> bool:
    while True:
        ans = input(f"{msg} (y/n): ").strip().lower()
        if ans in {"y", "yes"}: return True
        if ans in {"n", "no"}: return False
        print("Please answer 'y' or 'n'.")

def main():
    parser = argparse.ArgumentParser(description="SpotiPy-MultiObs Analysis Tool")
    parser.add_argument("--config", type=str, default=None, help="Path to params.txt")

    # Flags
    parser.add_argument("--auto", action="store_true", help="Run everything without asking")
    parser.add_argument("--plot-only", action="store_true", help="Skip processing and just generate plots")

    # Overrides
    parser.add_argument("--noaa", type=int, help="NOAA Active Region Number")
    parser.add_argument("--date", type=str, help="Start Date")
    parser.add_argument("--email", type=str, help="Email")
    parser.add_argument("--days", type=int, help="Duration")
    parser.add_argument("--cadence", type=int, help="Cadence")
    parser.add_argument("--observables", nargs="+", default=None)
    parser.add_argument("--recreate-masks", action="store_true", help="Force regenerate masks")

    args = parser.parse_args()

    print(f"--- SpotiPy Analysis ---")
    if args.config: print(f"Using Config File: {args.config}")

    steps = []

    if args.plot_only:
        print("[INFO] Plot-Only Mode Enabled. Skipping all processing steps.")
        # steps remains empty, so pipeline will just init folders and return path
    else:
        # 1. Download?
        if args.auto: do_dl = True
        else: do_dl = ask_yn("Download data?")
        if do_dl: steps.append('download')

        # 2. Align?
        if args.auto: do_align = True
        else: do_align = ask_yn("Align AIA?")
        if do_align: steps.append('align')

        # 3. Correct LD?
        if args.auto: do_ld = True
        else: do_ld = ask_yn("Correct AIA LD?")
        if do_ld: steps.append('aia_ld')

        # 4. HMI Analysis?
        if args.auto: do_hmi = True
        else: do_hmi = ask_yn("Run HMI Analysis?")
        if do_hmi: steps.append('hmi')

        # 5. AIA Analysis?
        if args.auto: do_aia = True
        else: do_aia = ask_yn("Run AIA Analysis?")
        if do_aia: steps.append('aia')

    # Ask for mask recreation if running analysis
    recreate = args.recreate_masks
    if not args.auto and not args.plot_only and (('hmi' in steps) or ('aia' in steps)) and not recreate:
        recreate = ask_yn("Recreate/Overwrite Mask Overlays?")

    # --- EXECUTE ---
    try:
        # We call pipeline to get the correct root directory path
        root_dir = pipeline.analyze_sunspot(
            noaa_number=args.noaa,
            start_time_str=args.date,
            duration_days=args.days,
            cadence_hours=args.cadence,
            email=args.email,
            config_file=args.config,
            steps=steps, # If steps is empty, it just returns path
            observables_list=args.observables,
            recreate_masks=recreate
        )

        print("\n--- Visualizing ---")
        visualization.generate_plots(root_dir, obs_list=args.observables)

        print("\n[DONE] Run complete.")

    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("Tip: Ensure params.txt is correct.")
        sys.exit(1)

if __name__ == "__main__":
    main()
