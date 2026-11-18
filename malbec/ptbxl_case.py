"""Select a public case without model scores, reproduce controls, then infer on PTB-XL."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, filtfilt, iirnotch

from predict import UPSTREAM, MergedIsotonicModel, load_model, predict, read_h5

PTB_VERSION = "1.0.3"
SOURCES = [f"https://physionet.org/files/ptb-xl/{PTB_VERSION}/",
           f"https://physionet-open.s3.amazonaws.com/ptb-xl/{PTB_VERSION}/"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", *(f"V{i}" for i in range(1, 7))]
ANTERIOR_CODES = {"AMI", "ASMI", "ALMI"}
# Rounded raw STEMI probabilities displayed in the authors' frozen notebook.
REFERENCE_STEMI = {"non": .000003, "stemi": .999984, "nstemi_type2": .000963,
                   "nstemi_type1": .080542, "omi": .961268}
REFERENCE_CAL_STEMI = {"non": .000107, "stemi": 1., "nstemi_type2": .002333,
                       "nstemi_type1": .102147, "omi": .910085}


def fetch_file(relative, dest):
    for base in SOURCES:
        try:
            with urlopen(base + relative, timeout=90) as response:
                data = response.read()
            if not data or data.lstrip().startswith(b"<"):
                raise ValueError("Received an HTML or empty response")
            dest.write_bytes(data)
            return hashlib.sha256(data).hexdigest()
        except (URLError, TimeoutError, ValueError) as exc:
            failure = exc
    raise RuntimeError(f"Could not fetch {relative}") from failure


def select_case(csv_path):
    frame = pd.read_csv(csv_path)
    # Chosen by human review of PTB-XL reports, before inference on this
    # record. Review list: GitHub Actions run 44300055860 (no model used).
    selected = frame.loc[frame.ecg_id == 13913]
    if len(selected) != 1:
        raise ValueError("Expected exactly one PTB-XL record 13913")
    r = selected.iloc[0]
    codes = ast.literal_eval(r.scp_codes)
    expected_findings = ("especially marked in v2,3,4", "depressed in iii",
                         "recent anterior and lateral myocardial")
    normalized_report = " ".join(r.report.lower().split())
    if (str(r.validated_by_human).lower() != "true" or r.infarction_stadium1 != "Stadium I"
            or not any(codes.get(key) == 100 for key in ANTERIOR_CODES)
            or not all(finding in normalized_report for finding in expected_findings)):
        raise ValueError("Selected record no longer matches the verified annotations")
    return {"ecg_id": int(r.ecg_id), "report": r.report, "stage": r.infarction_stadium1,
            "scp_codes": codes, "fold": int(r.strat_fold), "human_validated": True,
            "filename_hr": r.filename_hr, "dataset_version": PTB_VERSION,
            "selection_basis": "Stage I; pronounced ST elevation V2–V4 and reciprocal ST depression III; recent anterior/lateral infarction in the reviewed ECG report. ID fixed before model evaluation."}



def prepare(record):
    names = [name.upper() for name in record.sig_name]
    if len(set(names)) != 12 or set(names) != set(LEADS):
        raise ValueError(f"Unexpected lead set {names}")
    if record.fs != 500 or record.sig_len != 5000:
        raise ValueError("Expected the 500 Hz, ten-second PTB-XL version")
    indexes = [names.index(name) for name in LEADS]
    if any(record.units[i].lower() != "mv" for i in indexes):
        raise ValueError(f"Unexpected WFDB units {record.units}")
    mv = record.p_signal[:, indexes].T
    if not np.isfinite(mv).all():
        raise ValueError("Nonfinite measurements in this record")
    uv = mv * 1000.0
    # Match the digital filters and linear interpolation in upstream/ecg_processing.py.
    fs = record.fs
    for freq in (50, 60):
        for k in range(1, int(fs / 2 / freq - 1) + 1):
            b, a = iirnotch(k * freq / (fs / 2), 30)
            uv = filtfilt(b, a, uv, axis=-1)
    b, a = butter(4, .05 / (fs / 2), btype="high")
    uv = filtfilt(b, a, uv, axis=-1)
    positions = np.arange(4096) * (fs * 10 / 4096)
    preprocessed = np.stack([np.interp(positions, np.arange(len(lead)), lead) for lead in uv])
    return mv, preprocessed


def plot_ecg(mv, case, dest):
    fig, axes = plt.subplots(3, 4, figsize=(15, 7), sharex=True, sharey=True, layout="constrained")
    for row, col in np.ndindex(3, 4):
        i = col * 3 + row
        ax = axes[row, col]
        # Same 0-2.5 s interval in every display panel. Inference uses all 10 s.
        ax.plot(np.arange(1250) / 500, mv[i, :1250], color="#111111", lw=.9)
        ax.set_title(LEADS[i], loc="left", fontsize=10)
        ax.set_xticks(np.arange(0, 2.51, .2))
        ax.set_xticks(np.arange(0, 2.51, .04), minor=True)
        ax.set_yticks(np.arange(-4, 4.1, .5))
        ax.set_yticks(np.arange(-4, 4.1, .1), minor=True)
        limit = max(1.5, np.ceil(np.abs(mv[:, :1250]).max() * 2) / 2)
        ax.set(xlim=(0, 2.5), ylim=(-limit, limit))
        ax.grid(which="major", color="#e3a9ac", lw=.65)
        ax.grid(which="minor", color="#f4dadb", lw=.35)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"PTB-XL {PTB_VERSION}  |  ECG {case['ecg_id']:05d}  |  first 2.5 s of each lead")
    fig.supxlabel("Major grid: 0.2 s horizontally, 0.5 mV vertically.   Source: Wagner et al., PhysioNet (CC BY 4.0).")
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checksums = {}
    table = output / "ptbxl_database.csv"
    checksums["ptbxl_database.csv"] = fetch_file("ptbxl_database.csv", table)
    case = select_case(table)
    stem = case["filename_hr"]
    for suffix in (".hea", ".dat"):
        local_file = output / (Path(stem).name + suffix)
        checksums[stem + suffix] = fetch_file(stem + suffix, local_file)
    case["input_sha256"] = checksums
    (output / "selection.json").write_text(json.dumps(case, indent=2))
    print("Selected before inference:", json.dumps(case), flush=True)
    record = wfdb.rdrecord(str(output / Path(stem).name))
    mv, signal = prepare(record)
    plot_ecg(mv, case, output / "ptbxl_ecg.png")
    torch.set_num_threads(2)
    model = load_model(args.checkpoint)
    controls = {}
    for key, uv in read_h5(UPSTREAM / "data" / "ecgs_dummy.h5"):
        result = predict(model, uv, stemi_excluded=key in ("non", "nstemi_type1", "nstemi_type2"))
        np.testing.assert_allclose(result["raw_sigmoid"]["stemi_hd_v2"], REFERENCE_STEMI[key], atol=.0001, rtol=0)
        np.testing.assert_allclose(result["calibrated"]["stemi"], REFERENCE_CAL_STEMI[key], atol=.0001, rtol=0)
        controls[key] = result
    if set(controls) != set(REFERENCE_STEMI):
        raise ValueError("Missing reference examples")
    results = {"ptb_xl": {"id": case["ecg_id"], **predict(model, signal)}, "zenodo_controls": controls}
    (output / "predictions.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print("Verified 5 official controls. PTB-XL output:", json.dumps(results["ptb_xl"]), flush=True)


if __name__ == "__main__":
    main()
