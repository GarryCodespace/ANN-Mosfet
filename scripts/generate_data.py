from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mosfet_ann.data import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="data/mosfet_iv.csv")
    args = parser.parse_args()

    frame = generate_dataset(samples=args.samples, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} samples to {output}")


if __name__ == "__main__":
    main()
