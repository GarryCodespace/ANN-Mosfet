# Literature and Resource Notes

## Starting Resources

- Subir Maity, "VLSI MOSFET Characteristics Prediction using Neural Network"
  explains a practical ML framing for predicting MOSFET characteristics from
  device parameters and bias values.
- Nathan Iyer, "Accelerating and Enhancing SPICE Simulations with Neural
  Models" discusses neural surrogates for speeding simulation and improving
  modelling workflows.

## Project-Relevant Themes

- Neural models are most useful when trained on a wide, well-sampled operating
  space covering geometry, bias, process corners, and temperature.
- Predicting current directly is difficult because MOSFET current spans many
  orders of magnitude. A log-current target usually trains more reliably.
- Input scaling is essential. Biases, dimensions, and model parameters live on
  very different numeric scales.
- Robustness matters because measured characterisation data can include
  instrument noise, bias error, temperature drift, and extraction artefacts.
- A pure ANN surrogate may be accurate but can violate physical expectations
  such as monotonicity or smooth derivatives. Physics-informed constraints are
  important for later SPICE integration.

## Suggested Academic Search Terms

- neural network MOSFET compact model
- artificial neural network SPICE model extraction
- physics-informed neural network compact transistor model
- MOSFET IV CV modelling neural network
- Verilog-A neural network compact model
- BSIM parameter extraction machine learning

## Open Questions for the Report

- Which input features are truly available during extraction: raw measured I-V
  curves, known geometry/process parameters, or initial compact-model guesses?
- Is the desired output a direct current predictor, extracted BSIM parameters,
  or a SPICE-compatible behavioural model?
- How should model quality be measured: pointwise current error, circuit-level
  simulation error, derivative smoothness, or parameter extraction stability?
