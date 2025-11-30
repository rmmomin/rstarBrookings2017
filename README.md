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

## Python environment

The Python port currently lives under `tvar/python`. To reproduce the TVAR
estimation without depending on the system Python packages:

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
3. Run the estimator from the repo root (optionally adjust `main_model1.py`
   controls before launching):
   ```
   python -m tvar.python.main_model1
   ```

The runtime expectations in `main_model1.py` still apply (100k draws × 8
chains by default). Consider lowering `Ndraws` or toggling `RunEstimation`
while testing.
