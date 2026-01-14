# SpotiPy-MultiObs: Solar Active Region Analysis Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Publication--Ready-orange)]()

**SpotiPy-MultiObs** is a Python framework for the automated analysis of sunspot center-to-limb variations (CLV). It extends the functionality of the original [SpotiPy](https://github.com/Emily-Joe/SpotiPy) tool to support multi-wavelength and multi-observable analysis.

This framework automatically downloads, aligns, and extracts data for **5 solar observables** (Intensity, Magnetogram, Dopplergram, Line Depth, Line Width) and performs spatially resolved statistical analysis (East vs. West hemisphere asymmetry).

## 🚀 Key Features

* **Multi-Observable Pipeline:** Automatically processes SDO/HMI (Intensity, Magnetogram, Doppler) and SDO/AIA (1700 Å) data.
* **Spatial Asymmetry:** Splits sunspot data into "Leading" (West) and "Following" (East) hemispheres to study rotational effects and physical asymmetries.
* **Statistical "Candle" Plots:** Generates box-plot distributions of physical values across $\mu$ (cosine theta), allowing for robust CLV fitting.
* **Raw vs. Absolute Physics:** Handles signed values (velocity/magnetic field) correctly, offering both raw (signed) and absolute-magnitude analysis modes.
* **Interactive & Batch Modes:** Supports step-by-step interactive execution or fully automated batch processing.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/saqibsumra/SpotiPy_MultiObs.git](https://github.com/saqibsumra/SpotiPy_MultiObs.git)
    cd SpotiPy_MultiObs
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install numpy matplotlib scipy astropy sunpy[all] opencv-python reproject
    ```

## ⚙️ Configuration

All physics and run parameters are controlled via the `params.txt` file. You must edit this file before running the code to match your target Active Region.

**Key Parameters in `params.txt`:**
* `NOAA_NUMBER`: The Active Region number to track (e.g., `12673`).
* `START_DATE`: The starting timestamp (ISO format, e.g., `2017-09-06T12:00:00`).
* `EMAIL`: Your email address for JSOC data queries (**Required**).
* `CADENCE`: Time step in hours between observations (e.g., `24`).
* `DAYS`: Duration of the observation window.

## 🛠️ Usage

The framework is executed via the `run_analysis.py` interface. It supports **4 distinct modes of operation** to suit different research needs.

### 1. Interactive Mode (Standard)
Best for first-time runs. The script guides you step-by-step with Yes/No prompts for every stage (Download, Alignment, Extraction, etc.).
```bash
python3 run_analysis.py --config params.txt
