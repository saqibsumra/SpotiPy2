"""
SpotiPy-MultiObs: Statistical Visualization Module
==================================================

Description:
    This module implements the graphical rendering layer for the analysis suite, focusing on
    the statistical visualization of Center-to-Limb Variation (CLV) profiles.

    It transforms the raw extracted data into "Candle" (box-and-whisker) plots, which
    provide a clear representation of the statistical dispersion of solar observables
    as a function of viewing angle ($\mu$).

    Key Scientific Capabilities:
    1.  **Statistical Binning:** Aggregates high-volume pixel data into $\mu$-bins to
        visualize the mean trend and variance (standard deviation) simultaneously.
    2.  **Trend Quantifiction:** Overlays polynomial regression fits to quantify the
        mean limb-darkening or center-to-limb behavior for each feature class.
    3.  **Hemispheric Segmentation:** Automatically splits data into East ($x < 0$) and
        West ($x > 0$) hemispheres to investigate potential asymmetries (e.g., rotational
        Doppler shifts or trailing/leading polarity differences).
    4.  **Vector Magnitude Analysis:** For vector quantities like Magnetic Field ($M$) and
        Doppler Velocity ($V$), the module generates parallel visualizations for both
        signed (raw) values and absolute magnitudes ($|B|$, $|v|$).

Usage:
    This module is automatically invoked by `run_analysis.py` after data extraction.
    It can also be run standalone to regenerate plots from existing text files.

Author: Muhammad Saqib Sumra
Date:   2026
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Configuration Constants
# ----------------------------------------------------------------------

OBSERVABLES = ['Ic', 'M', 'V', 'Ld', 'Lw']

# Observables that should be plotted as both Raw and Absolute values
# (e.g., Magnetograms M and Dopplergrams V are vector/signed quantities)
DUAL_MODE_OBS = ['M', 'V']

POLY_ORDER_AIA = 2
POLY_ORDER_HMI = 2
MIN_MU_FOR_FIT = 0.15

# Bin Edges for the Candle plots (Mu from 0.1 to 1.0)
MU_EDGES = np.linspace(0.1, 1.0 + 1e-6, 9)

# ----------------------------------------------------------------------
# 2. Data Loading & Helper Functions
# ----------------------------------------------------------------------

def pick_existing_dir(candidates):
    """Returns the first existing directory from a list of candidates."""
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None

def load_data_4col(txt_path):
    """
    Loads Mu, I, X, Y data from text files.
    Format: [Mu, Intensity, X_arcsec, Y_arcsec]
    """
    if not os.path.exists(txt_path):
        return None, None, None, None
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Backward compatibility for old 2-col files (no spatial data)
        if data.shape[1] < 4:
            return data[:,0], data[:,1], np.zeros_like(data[:,0]), np.zeros_like(data[:,0])

        mu = data[:, 0]
        I  = data[:, 1]
        x  = data[:, 2]
        y  = data[:, 3]

        mask = np.isfinite(mu) & np.isfinite(I)
        return mu[mask], I[mask], x[mask], y[mask]
    except Exception as e:
        print(f"[ERR] Failed to load {txt_path}: {e}")
        return None, None, None, None

def filter_spatial(mu, I, x, region_code):
    """
    Filters data based on solar hemisphere.
    East: X < 0, West: X > 0.
    """
    if mu is None: return None, None
    if region_code == 'East':
        mask = (x < 0)
    elif region_code == 'West':
        mask = (x > 0)
    else:
        return mu, I
    return mu[mask], I[mask]

def bin_stats(mu, I, mu_edges):
    """Calculates Mean and Count for each Mu bin."""
    mu = np.asarray(mu); I = np.asarray(I)
    centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    means   = np.zeros_like(centers)
    counts  = np.zeros_like(centers, dtype=int)

    for i, (lo, hi) in enumerate(zip(mu_edges[:-1], mu_edges[1:])):
        m = (mu >= lo) & (mu < hi)
        vals = I[m]
        if vals.size > 0:
            means[i]  = np.mean(vals)
            counts[i] = vals.size
        else:
            means[i]  = np.nan
            counts[i] = 0
    return centers, means, counts

def compute_candle_binning(mu, I, mu_edges):
    """Groups data arrays into bins for the boxplot function."""
    mu = np.asarray(mu); I = np.asarray(I)
    centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    per_bin = []
    for lo, hi in zip(mu_edges[:-1], mu_edges[1:]):
        m = (mu >= lo) & (mu < hi)
        per_bin.append(I[m])
    return centers, per_bin

def fit_from_candles(mu_centers, means, min_mu, max_order):
    """Fits a polynomial to the binned means."""
    mask = np.isfinite(mu_centers) & np.isfinite(means) & (mu_centers >= min_mu)
    x = mu_centers[mask]
    y = means[mask]

    if x.size < 2: return None, None

    order = min(int(max_order), int(x.size - 1))
    coeffs_raw = np.polyfit(x, y, order)

    val1 = np.polyval(coeffs_raw, 1.0)
    coeffs_norm = None if abs(val1) < 1e-9 else coeffs_raw / val1

    return coeffs_norm, coeffs_raw

# ----------------------------------------------------------------------
# 3. Plotting Functions
# ----------------------------------------------------------------------

def plot_component(component, out_path, obs, mode_label):
    """Plots a single component (e.g. Umbra) with candles and fit."""
    name, mu, I, color, order = component["name"], component["mu"], component["I"], component["color"], component["order"]
    if mu is None or len(mu) < 5: return

    centers, means, _ = bin_stats(mu, I, MU_EDGES)
    _, coeffs_raw = fit_from_candles(centers, means, MIN_MU_FOR_FIT, order)
    centers_candles, bin_lists = compute_candle_binning(mu, I, MU_EDGES)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mu, I, s=3, alpha=0.15, color="gray", zorder=1)
    
    bp = ax.boxplot(bin_lists, positions=centers_candles, widths=0.05,
                    patch_artist=True, showfliers=False, showmeans=True, meanline=True, manage_ticks=False)

    for box in bp["boxes"]:
        box.set_facecolor(color); box.set_alpha(0.5)
    for mean in bp["means"]:
        mean.set_color("black"); mean.set_linewidth(1.5)
    for med in bp["medians"]:
        med.set_color("white"); med.set_linewidth(1.0)

    if coeffs_raw is not None:
        xs = np.linspace(max(MIN_MU_FOR_FIT, np.nanmin(mu)), 1.0, 300)
        ys = np.polyval(coeffs_raw, xs)
        ax.plot(xs, ys, color=color, linewidth=2.5, label=f"{name} Fit", zorder=10)

    ax.set_xlabel(r"$\mu = \cos\theta$")
    ax.set_ylabel(f"Value ({obs} - {mode_label})")
    ax.set_title(f"{name} | {obs} | {mode_label}")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_combined(components, out_norm, out_raw, obs, mode_label):
    """Plots all components on a single graph (Normalized and Raw)."""
    xs = np.linspace(MIN_MU_FOR_FIT, 1.0, 400)
    fig_n, ax_n = plt.subplots(figsize=(7,5))
    fig_r, ax_r = plt.subplots(figsize=(7,5))

    has_data = False
    for c in components:
        if c["mu"] is None or len(c["mu"]) < 5: continue
        centers, means, _ = bin_stats(c["mu"], c["I"], MU_EDGES)
        c_norm, c_raw = fit_from_candles(centers, means, MIN_MU_FOR_FIT, c["order"])

        if c_norm is not None:
            ax_n.plot(xs, np.polyval(c_norm, xs), color=c["color"], lw=2, label=c["name"])
        if c_raw is not None:
            ax_r.plot(xs, np.polyval(c_raw, xs), color=c["color"], lw=2, label=c["name"])
            has_data = True

    if has_data:
        ax_n.set_title(f"Combined {obs} (Norm) - {mode_label}")
        ax_n.set_xlabel(r"$\mu$"); ax_n.set_ylabel("Normalized Intensity")
        ax_n.legend()
        fig_n.savefig(out_norm, dpi=150)

        ax_r.set_title(f"Combined {obs} (Raw) - {mode_label}")
        ax_r.set_xlabel(r"$\mu$"); ax_r.set_ylabel(f"Value ({obs})")
        ax_r.legend()
        fig_r.savefig(out_raw, dpi=150)
        print(f"   -> Saved combined plots for {mode_label}")

    plt.close(fig_n)
    plt.close(fig_r)

# ----------------------------------------------------------------------
# 4. Main Driver
# ----------------------------------------------------------------------

def generate_plots(root_dir, obs_list=None):
    """
    Entry point for visualization.
    Iterates through all Observables -> Modes -> Regions.
    """
    print(f"[INFO] Plotting Root: {root_dir}")

    # Auto-detect results directories inside the given root_dir
    base_hmi = pick_existing_dir([os.path.join(root_dir, "Results_HMI"), os.path.join(root_dir, "LimbDarkening_results_HMI")])
    base_aia = pick_existing_dir([os.path.join(root_dir, "Results_AIA"), os.path.join(root_dir, "LimbDarkening_results_AIA")])

    if not base_hmi or not base_aia:
        print("[ERR] Missing Results directories in the output folder.")
        return

    # Use specific list if provided via --observables flag
    target_observables = obs_list if obs_list else OBSERVABLES

    for obs in target_observables:
        print(f"\n=== Processing {obs} ===")

        hmi_obs_dir = os.path.join(base_hmi, obs)
        aia_obs_dir = os.path.join(base_aia, obs)

        if not os.path.isdir(hmi_obs_dir):
            print(f"   [SKIP] Missing folder: {hmi_obs_dir}")
            continue

        # Load RAW data (4 columns: mu, I, x, y)
        mu_U, I_U, x_U, _ = load_data_4col(os.path.join(hmi_obs_dir, "lbd_U_raw.txt"))
        mu_P, I_P, x_P, _ = load_data_4col(os.path.join(hmi_obs_dir, "lbd_P_raw.txt"))
        mu_F, I_F, x_F, _ = load_data_4col(os.path.join(hmi_obs_dir, "lbd_F_raw.txt"))

        mu_Q, I_Q, x_Q, _ = load_data_4col(os.path.join(aia_obs_dir, "lbd_Q_raw.txt"))
        mu_Pl, I_Pl, x_Pl, _ = load_data_4col(os.path.join(aia_obs_dir, "lbd_Plage_raw.txt"))
        mu_N, I_N, x_N, _ = load_data_4col(os.path.join(aia_obs_dir, "lbd_Network_raw.txt"))

        # Determine Analysis Modes: "Raw" is always processed.
        # "Abs" (Absolute Value) is added for vector quantities like M and V.
        modes = ["Raw"]
        if obs in DUAL_MODE_OBS:
            modes.append("Abs")

        for mode in modes:
            print(f"   >> Mode: {mode}")

            # Helper to convert data based on mode
            def get_data(mu_in, I_in, x_in):
                if mu_in is None: return None, None, None
                I_out = np.abs(I_in) if mode == "Abs" else I_in.copy()
                return mu_in, I_out, x_in

            # Iterate Spatial Regions
            for region in ["Full", "East", "West"]:
                out_dir = os.path.join(root_dir, "Post_CLV_candles", obs, mode, region)
                os.makedirs(out_dir, exist_ok=True)

                comps_to_plot = []

                # Define Solar Components
                raw_comps = [
                    ("Quiet Sun", mu_Q, I_Q, x_Q, "black", POLY_ORDER_AIA),
                    ("Plage",     mu_Pl, I_Pl, x_Pl, "red",   POLY_ORDER_AIA),
                    ("Network",   mu_N, I_N, x_N, "orange", POLY_ORDER_AIA),
                    ("Spot",      mu_F, I_F, x_F, "cyan",   POLY_ORDER_HMI),
                    ("Umbra",     mu_U, I_U, x_U, "magenta", 2), # Lower order for Umbra due to sparsity
                    ("Penumbra",  mu_P, I_P, x_P, "green",  POLY_ORDER_HMI)
                ]

                # Process Components
                for name, m, i, x, col, order in raw_comps:
                    m_mode, i_mode, x_mode = get_data(m, i, x)
                    m_final, i_final = filter_spatial(m_mode, i_mode, x_mode, region)

                    if m_final is not None and len(m_final) > 10:
                        comp_dict = {
                            "name": name, "mu": m_final, "I": i_final,
                            "color": col, "order": order
                        }
                        comps_to_plot.append(comp_dict)
                        fname = f"{name.replace(' ', '')}.png"
                        plot_component(comp_dict, os.path.join(out_dir, fname), obs, f"{mode}-{region}")

                # Generate Combined Summary
                if comps_to_plot:
                    plot_combined(comps_to_plot,
                                  os.path.join(out_dir, "Combined_Norm.png"),
                                  os.path.join(out_dir, "Combined_Raw.png"),
                                  obs, f"{mode}-{region}")

    print("\n[DONE] All plots generated.")

if __name__ == "__main__":
    generate_plots(os.getcwd())
