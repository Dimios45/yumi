"""Operator-run ruler calibration: touching jaws, two intermediate gaps, maximum."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def finish(config, samples_path, output, maximum_mm):
    from yumi.calibration import fit_piecewise_image_width

    samples = json.loads(samples_path.read_text())
    openings = np.sort(np.asarray(samples["jaw_opening_m"], float))
    if (
        len(openings) != 4
        or not np.isclose(openings[0], 0, atol=1e-9)
        or not np.isclose(openings[-1], maximum_mm / 1000, atol=1e-9)
    ):
        raise ValueError(
            "Save exactly four settings: 0, two intermediate gaps, and the entered maximum"
        )
    fit = fit_piecewise_image_width(samples, minimum_samples=4)
    model = fit["image_width_calibration"]
    model.update(
        source_samples=str(samples_path.resolve()),
        source_sha256=hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        measurement_method="ruler at three nonzero gaps; physical jaw contact at zero",
        independent_accuracy_validated=False,
    )
    config["markers"].update(fit)
    config["calibration_verified"] = False
    config["timing_verified"] = False
    with output.open("x") as file:
        json.dump(config, file, indent=2, allow_nan=False)
        file.write("\n")
    return fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/umi.yam-collection.json"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--control-port", type=int, default=8081)
    args = parser.parse_args()
    maximum_mm = float(
        input("Measure the MAXIMUM inner jaw gap with your ruler. Enter mm: ")
    )
    if not np.isfinite(maximum_mm) or maximum_mm < 20:
        parser.error("Enter a finite maximum of at least 20 mm")
    config = json.loads(args.config.read_text())
    config["markers"]["width_range_m"] = [0.0, maximum_mm / 1000]
    config["markers"]["image_width_calibration"] = None
    config["markers"]["width_method"] = "image_calibrated"
    config["calibration_verified"] = False
    config["timing_verified"] = False
    root = (
        ROOT
        / "data/calibration"
        / ("four-point-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S"))
    )
    root.mkdir(parents=True, exist_ok=False)
    samples = root / "samples.json"
    output = root / "umi.ruler.json"
    (root / "capture-config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Open http://localhost:{args.port}. Save exactly FOUR settings in the browser:"
    )
    print("1. Jaws touching: enter 0 (no ruler measurement).")
    print("2–3. Two ruler-measured gaps, roughly one-third and two-thirds open.")
    print(f"4. Maximum opening: enter {maximum_mm:g} mm.")
    print(
        "Hold each gap steady; wait for READY; use 'Measured jaw gap (mm)' and 'Save measured gap'."
    )
    print(
        "Leave 'Validation sample' unchecked. After all four saves, Ctrl+C HERE to fit and save."
    )
    print(
        "Camera only; no robot access. A new config is written; your existing config is retained.",
        flush=True,
    )
    from yumi.live_calibration import serve

    serve(config, samples, args.port, args.control_port)
    if not samples.exists():
        print(f"No measurements saved. Session folder: {root}")
        return
    try:
        finish(config, samples, output, maximum_mm)
    except ValueError as error:
        print(f"Fit not accepted: {error}\nMeasurements retained at {samples}")
        return
    print(
        f"SAVED CALIBRATION: {output}\nFour-point ruler interpolation; physical accuracy remains unvalidated."
    )


if __name__ == "__main__":
    main()
