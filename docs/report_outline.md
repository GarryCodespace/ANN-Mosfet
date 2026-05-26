# Report Outline: Suitability of ANN for MOSFET SPICE Model Generation

## 1. Executive Summary

Summarise whether ANN models are suitable for MOSFET SPICE model extraction,
where they perform well, and what limitations remain.

## 2. Background

- MOSFET I-V and C-V behaviour
- Compact models in SPICE and commercial PDKs
- Traditional model extraction workflow
- Motivation for ANN-assisted extraction

## 3. Literature Review

- ANN-based MOSFET characteristic prediction
- Neural surrogate models for SPICE acceleration
- Machine-learning-assisted compact-model parameter extraction
- Physics-informed or constrained neural compact models

## 4. Data Generation

- Device geometry ranges
- Bias sweep ranges
- Temperature and process variation
- SPICE simulator setup or analytic fallback model
- Data cleaning and normalisation

## 5. ANN Methodology

- Input feature selection
- Target selection: raw current, log current, compact-model parameters, or C-V
- Architecture candidates
- Training, validation, and test split
- Error metrics

## 6. Experiments

- Baseline MLP surrogate
- Architecture comparison
- Input selection comparison
- Robustness to imperfect input data
- Extrapolation outside the training range

## 7. Results

- Prediction error tables
- I-V sweep plots
- Robustness plots
- Failure case analysis

## 8. Suitability Assessment

- Benefits for extraction speed and automation
- Risks for circuit simulation accuracy
- Requirements for commercial PDK use
- Recommended next steps

## 9. Conclusion

State whether ANN modelling is suitable as a standalone model, an extraction
assistant, or a hybrid physics/ML compact modelling method.
