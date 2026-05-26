from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mosfet_ann.data import CURRENT_EPS, MosfetParameters, drain_current
from mosfet_ann.model import FEATURE_COLUMNS, MosfetAnnModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict MOSFET drain current with a trained ANN.")
    parser.add_argument("--model", default="artifacts/mosfet_ann.pt")
    parser.add_argument("--vgs", type=float, default=0.8, help="Gate-source voltage in V")
    parser.add_argument("--vds", type=float, default=0.8, help="Drain-source voltage in V")
    parser.add_argument("--vbs", type=float, default=0.0, help="Body-source voltage in V")
    parser.add_argument("--width-um", type=float, default=1.0, help="MOSFET width in micrometres")
    parser.add_argument("--length-um", type=float, default=0.18, help="MOSFET length in micrometres")
    parser.add_argument("--temperature-c", type=float, default=27.0, help="Temperature in Celsius")
    parser.add_argument("--vth0", type=float, default=0.42)
    parser.add_argument("--kp", type=float, default=220e-6)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.06)
    parser.add_argument("--gamma", type=float, default=0.45)
    parser.add_argument("--phi", type=float, default=0.65)
    args = parser.parse_args()

    row = pd.DataFrame(
        [
            {
                "vgs": args.vgs,
                "vds": args.vds,
                "vbs": args.vbs,
                "width_um": args.width_um,
                "length_um": args.length_um,
                "temperature_c": args.temperature_c,
                "vth0": args.vth0,
                "kp": args.kp,
                "lambda_": args.lambda_,
                "gamma": args.gamma,
                "phi": args.phi,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    model = MosfetAnnModel.load(args.model)
    predicted_log_id = model.predict_log_current(row)[0]
    predicted_id = model.predict_current(row)[0]

    reference_id = drain_current(
        vgs=row["vgs"].to_numpy(),
        vds=row["vds"].to_numpy(),
        vbs=row["vbs"].to_numpy(),
        width_um=row["width_um"].to_numpy(),
        length_um=row["length_um"].to_numpy(),
        temperature_c=row["temperature_c"].to_numpy(),
        params=MosfetParameters(
            vth0=args.vth0,
            kp=args.kp,
            lambda_=args.lambda_,
            gamma=args.gamma,
            phi=args.phi,
        ),
    )[0]

    print("Input:")
    for column in FEATURE_COLUMNS:
        print(f"  {column}: {row[column].iloc[0]:.6g}")
    print()
    print(f"ANN predicted log10(Id): {predicted_log_id:.6f}")
    print(f"ANN predicted Id:        {predicted_id:.6e} A")
    print(f"Generator reference Id:  {reference_id:.6e} A")
    print(f"Reference log10(Id):     {np.log10(abs(reference_id) + CURRENT_EPS):.6f}")


if __name__ == "__main__":
    main()
