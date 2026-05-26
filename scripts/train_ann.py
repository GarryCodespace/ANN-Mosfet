from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mosfet_ann.model import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mosfet_iv.csv")
    parser.add_argument("--model", default="artifacts/mosfet_ann.pt")
    parser.add_argument("--max-iter", type=int, default=800)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    model, train_metrics, test_metrics = train_model(frame, max_iter=args.max_iter)
    model.save(args.model)

    print("Train:", train_metrics)
    print("Test: ", test_metrics)
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
