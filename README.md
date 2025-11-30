# rstarBrookings2017

This fork builds off of the replication files for
[*Safety, Liquidity, and the Natural Rate of Interest*](https://www.brookings.edu/bpea-articles/safety-liquidity-and-the-natural-rate-of-interest/)
by Marco del Negro, Domenico Giannone, Marc Giannoni, and Andrea Tambalotti,
*Brookings Papers on Economic Activity*, Spring 2017: 235-294.

The primary goal here is to translate the original MATLAB toolchain
into a Python-first workflow while keeping output compatibility for the
Brookings figures and tables. The work-in-progress port mirrors the
MATLAB estimation and plotting scripts so both ecosystems can run side
by side during the transition.

## Repository layout

- `python/` – modernized Python package containing the TVAR port.
- `tvar/` – original MATLAB scripts, routines, data inputs, and legacy outputs.
- `requirements.txt` – runtime dependencies for the Python environment.

## Python environment

The Python port now lives under the top-level `python/` package so it can evolve
independently of the MATLAB scripts. To reproduce the TVAR estimation without
depending on the system Python packages:

1. Create a virtual environment (Python 3.9+ recommended):
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the estimator from the repo root (optionally adjust `python/main_model1.py`
   controls before launching):
   ```
   python -m python.main_model1
   ```

   You can override the heavy defaults without editing code by supplying
   environment variables, e.g.
   ```
   RSTAR_NDRAWS=200 RSTAR_NCHAINS=1 RSTAR_THIN=2 python -m python.main_model1
   ```
   Supported toggles: `RSTAR_RUN_ESTIMATION`, `RSTAR_OUTPUT_NAME`,
   `RSTAR_NDRAWS`, `RSTAR_NCHAINS`, `RSTAR_THIN`, `RSTAR_NBENCH`.

   The Python port still reads inputs from the legacy MATLAB directory
   (`tvar/DataCompleteLatest.xls`) and drops its MAT/PDF/CSV outputs into
   the same `tvar` subfolders so both toolchains stay in sync.

The runtime expectations in `main_model1.py` still apply (100k draws × 8
chains by default). Consider lowering `Ndraws` or toggling `RunEstimation`
while testing.
