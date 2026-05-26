import numpy as np

from mosfet_ann.data import generate_dataset
from mosfet_ann.model import train_model
from mosfet_ann.robustness import evaluate_bias_noise


def test_dataset_contains_finite_targets():
    frame = generate_dataset(samples=128, seed=1)
    assert len(frame) == 128
    assert np.isfinite(frame["id_a"]).all()
    assert np.isfinite(frame["log_id"]).all()


def test_training_and_robustness_pipeline_runs():
    frame = generate_dataset(samples=96, seed=2)
    model, _, test_metrics = train_model(frame, hidden_layer_sizes=(8,), max_iter=3)
    assert np.isfinite(test_metrics.mae_log10_a)

    result = evaluate_bias_noise(model, frame.head(32), noise_sigmas_mv=[0, 10])
    assert list(result["bias_noise_sigma_mv"]) == [0, 10]
    assert np.isfinite(result["mean_abs_delta_log10_a"]).all()

    prediction = model.predict_current(frame.head(1))
    assert prediction.shape == (1,)
    assert prediction[0] > 0
