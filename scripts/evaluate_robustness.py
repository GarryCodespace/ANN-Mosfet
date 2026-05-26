from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mosfet_ann.model import MosfetAnnModel
from mosfet_ann.robustness import evaluate_bias_noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mosfet_iv.csv")
    parser.add_argument("--model", default="artifacts/mosfet_ann.pt")
    parser.add_argument("--output", default="artifacts/robustness.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    model = MosfetAnnModel.load(args.model)
    result = evaluate_bias_noise(model, frame, noise_sigmas_mv=[0, 1, 2, 5, 10, 20])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote robustness results to {output}")


if __name__ == "__main__":
    main()
