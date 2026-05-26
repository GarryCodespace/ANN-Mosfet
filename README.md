# ANN for MOSFET SPICE Model Extraction

This project explores how an artificial neural network can learn MOSFET
electrical behaviour from simulated I-V data and act as a first step toward
SPICE model extraction.

The current implementation is intentionally lightweight:

- Generates MOSFET drain current sweeps using a compact Shichman-Hodges style
  model with process/device variation.
- Trains a PyTorch neural network surrogate to predict `log10(|Id| + eps)` from bias
  and geometry inputs.
- Evaluates generalisation on held-out operating points.
- Tests robustness to imperfect input data by injecting bias noise.

## Project Layout

```text
mosfet_ann/
  data.py          Synthetic MOSFET data generation
  model.py         ANN training, prediction, evaluation, persistence
  robustness.py    Noise-sensitivity experiment
scripts/
  generate_data.py
  train_ann.py
  predict_current.py
  evaluate_robustness.py
docs/
  report_outline.md
  literature_notes.md
tests/
  test_pipeline.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

Generate a dataset:

```bash
python scripts/generate_data.py --samples 8000 --output data/mosfet_iv.csv
```

Train the ANN:

```bash
python scripts/train_ann.py --data data/mosfet_iv.csv --model artifacts/mosfet_ann.pt
```

Predict drain current for one MOSFET bias/device point:

```bash
python scripts/predict_current.py \
  --model artifacts/mosfet_ann.pt \
  --vgs 0.8 \
  --vds 0.8 \
  --width-um 1.0 \
  --length-um 0.18
```

Evaluate robustness to noisy bias inputs:

```bash
python scripts/evaluate_robustness.py \
  --data data/mosfet_iv.csv \
  --model artifacts/mosfet_ann.pt \
  --output artifacts/robustness.csv
```

## Current Model Inputs

The ANN uses:

- `vgs`: gate-source voltage
- `vds`: drain-source voltage
- `vbs`: body-source voltage
- `width_um`: MOSFET width in micrometres
- `length_um`: MOSFET length in micrometres
- `temperature_c`: temperature in Celsius
- `vth0`: nominal threshold voltage
- `kp`: transconductance parameter
- `lambda_`: channel-length modulation
- `gamma`: body-effect coefficient
- `phi`: surface potential

The target is `log10(abs(id_a) + 1e-15)`, which is much easier for an ANN to
learn than raw drain current across many orders of magnitude.

## Next Technical Steps

1. Replace or complement the analytic generator with `ngspice` simulations.
2. Add C-V data targets, not just I-V.
3. Compare architectures: MLP, residual MLP, monotonic networks, and
   physics-informed loss terms.
4. Export a SPICE-compatible behavioural source or Verilog-A approximation.
5. Validate against measured MOSFET characterisation data when available.
