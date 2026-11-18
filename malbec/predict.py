"""CPU inference for a preprocessed MALBEC waveform. Input units: microvolts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM / "src"))
from h5dataset import outcome_labels
from hparams import hparams_dict
from model import LitResNet
# Old joblib models may refer to this class via __main__.
from recalibrate import MergedIsotonicModel

CKPT = "finetuned_18d5a8abc508439a82786289b8f67ccc.ckpt"
SHA256 = "293f5f9b5317c279172d9f1f3b8dadc67d23513ccc058e93db15352d8b5fc986"
MEAN_UV = -0.5810575960776053
STD_UV = 328.8951471302853


def load_model(path):
    with Path(path).open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SHA256:
            raise ValueError("Checkpoint differs from the verified Zenodo release")
    return LitResNet.load_from_checkpoint(str(path), hparams=hparams_dict, map_location="cpu").eval()


def calibrate(name, value):
    calibration = joblib.load(UPSTREAM / "models" / f"model_{name}_hd.pkl")
    return float(calibration.predict([value])[0])


@torch.inference_mode()
def predict(model, uv, *, stemi_excluded=False):
    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape != (12, 4096) or not np.isfinite(uv).all():
        raise ValueError("Expected twelve complete, finite 4096-sample leads in microvolts")
    if (np.ptp(uv, axis=1) == 0).any():
        raise ValueError("One or more leads have no signal")
    x = torch.from_numpy((uv - MEAN_UV) / STD_UV).float().unsqueeze(0)
    scores = torch.sigmoid(model(x)).squeeze(0).tolist()
    raw = dict(zip(outcome_labels, scores, strict=True))
    calibrated = {"stemi": calibrate("stemi", raw["stemi_hd_v2"])}
    if stemi_excluded:
        # The notebook uses recalibrated STEMI in the max, and conditions the
        # NSTEMI analysis on STEMI exclusion; these are not mutually exclusive scores.
        for name in ("nstemi", "nstemi_type1", "nstemi_type2"):
            value = raw[f"{name}_hd_v2"]
            if name in ("nstemi", "nstemi_type1"):
                value = max(value, calibrated["stemi"])
            calibrated[name] = calibrate(name, value)
    return {"raw_sigmoid": raw, "calibrated": calibrated}


def read_h5(path):
    with h5py.File(path, "r") as handle:
        if len(handle["data"]) != len(handle["hash"]):
            raise ValueError("Waveform and record counts differ")
        for key, signal in zip(handle["hash"].asstr(), handle["data"], strict=True):
            if signal.shape == (1, 12, 4096):
                signal = signal[0]
            yield key, signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="MALBEC-format HDF5 with data and hash arrays")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stemi-excluded", action="store_true",
                        help="Evaluate NSTEMI only for a cohort with STEMI independently excluded")
    args = parser.parse_args()
    model = load_model(args.checkpoint)
    for key, signal in read_h5(args.h5):
        print(json.dumps({"id": key, **predict(model, signal, stemi_excluded=args.stemi_excluded)}, allow_nan=False))
