# SpotiPy-MultiObs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**SpotiPy-MultiObs** is a Python tool for the automated analysis of sunspot center-to-limb variations (CLV). It extends the functionality of the original [SpotiPy](https://github.com/Emily-Joe/SpotiPy) framework to support multi-wavelength and multi-observable analysis.

This tool automatically downloads, aligns, and extracts data for **5 solar observables** (Intensity, Magnetogram, Dopplergram, Line Depth, Line Width) and performs spatially resolved statistical analysis (East vs. West hemisphere asymmetry).

## 🚀 Key Features

* **Multi-Observable Pipeline:** Automatically processes SDO/HMI (Intensity, Magnetogram, Doppler) and SDO/AIA (1700 Å) data.
* **Spatial Asymmetry:** Splits sunspot data into "Leading" (West) and "Following" (East) hemispheres to study rotational effects and physical asymmetries.
* **Statistical "Candle" Plots:** Generates box-plot distributions of physical values across $\mu$ (cosine theta), allowing for robust CLV fitting.
* **Raw vs. Absolute Physics:** Handles signed values (velocity/magnetic field) correctly, offering both raw (signed) and absolute-magnitude analysis modes.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/SpotiPy_MultiObs.git](https://github.com/USERNAME/SpotiPy_MultiObs.git)
    cd SpotiPy_MultiObs
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: This tool requires `sunpy`, `astropy`, `reproject`, `opencv-python`, and `scikit-image`.*

## 🛠️ Usage

You can run the full analysis pipeline directly from the command line.

### Basic Run
Analyze a specific active region (e.g., NOAA 12218) for a set duration:

```bash
python run_analysis.py --noaa 12218 --date "2014-11-23T20:00:00" --days 13 --email "write_email"
