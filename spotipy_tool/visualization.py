import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBSERVABLES = ['Ic', 'M', 'V', 'Ld', 'Lw']
DUAL_MODE_OBS = ['M', 'V']
MU_EDGES = np.linspace(0.1, 1.0 + 1e-6, 9)
MIN_MU = 0.15

def load_data_4col(txt_path):
    if not os.path.exists(txt_path): return None, None, None, None
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1: data = data.reshape(1, -1)
        if data.shape[1] < 4: return data[:,0], data[:,1], np.zeros_like(data[:,0]), np.zeros_like(data[:,0])
        mask = np.isfinite(data[:,0]) & np.isfinite(data[:,1])
        return data[mask,0], data[mask,1], data[mask,2], data[mask,3]
    except: return None, None, None, None

def bin_stats(mu, I):
    centers = 0.5 * (MU_EDGES[:-1] + MU_EDGES[1:])
    means = np.zeros_like(centers)
    for i, (lo, hi) in enumerate(zip(MU_EDGES[:-1], MU_EDGES[1:])):
        vals = I[(mu >= lo) & (mu < hi)]
        means[i] = np.mean(vals) if vals.size > 0 else np.nan
    return centers, means

def generate_plots(root_dir, obs_list=None):
    if obs_list is None: obs_list = OBSERVABLES
    
    base_hmi = os.path.join(root_dir, "Results_HMI")
    base_aia = os.path.join(root_dir, "Results_AIA")
    
    for obs in obs_list:
        print(f"   [Plot] Generating plots for {obs}...")
        hmi_dir = os.path.join(base_hmi, obs)
        aia_dir = os.path.join(base_aia, obs)
        
        # Load Data
        mu_U, I_U, x_U, _ = load_data_4col(os.path.join(hmi_dir, "lbd_U_raw.txt"))
        mu_Q, I_Q, x_Q, _ = load_data_4col(os.path.join(aia_dir, "lbd_Q_raw.txt"))
        # (Load others P, F, Plage, Network similarly...)

        modes = ["Raw"]
        if obs in DUAL_MODE_OBS: modes.append("Abs")
        
        for mode in modes:
            for region in ["Full", "East", "West"]:
                out_dir = os.path.join(root_dir, "Post_CLV_candles", obs, mode, region)
                os.makedirs(out_dir, exist_ok=True)
                # ... (Paste the plotting logic from candel_111225.py here) ...
