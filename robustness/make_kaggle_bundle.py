"""Optional. Builds an upload folder for running the robustness notebooks on Kaggle.

Usage: python robustness/make_kaggle_bundle.py
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm

BUNDLE = os.path.join(cm.ROOT, "kaggle_bundle")
DST_PV2 = os.path.join(BUNDLE, "los_nlos")


def put(src, rel):
    dst = os.path.join(DST_PV2, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return os.path.getsize(dst)


def main():
    idx_path = os.path.join(cm.SNR_DATA_DIR, "clusters", "index.json")
    if not os.path.exists(idx_path):
        print("building the SNR-equalised dataset (measuring the noise floor of all clusters) ...")
        t0 = time.time()
        snr = cm.measure_snr()
        info = cm.build_snr_equalised_dataset(snr)
        print(f"  built in {time.time() - t0:.0f} s; target SNR {info['target_snr_db']:.1f} dB")
    else:
        print("using the existing SNR-equalised dataset:", cm.SNR_DATA_DIR)

    if os.path.isdir(BUNDLE):
        shutil.rmtree(BUNDLE)
    size = 0

    with open(os.path.join(cm.ORIG_DATA_DIR, "clusters", "index.json")) as f:
        oidx = json.load(f)
    size += put(os.path.join(cm.ORIG_DATA_DIR, "clusters", "index.json"), "output_raw_full/clusters/index.json")
    for c in cm.ALL_CLUSTERS:
        ff = oidx["clusters"][c]["features_file"]
        size += put(os.path.join(cm.ORIG_DATA_DIR, "clusters", ff), f"output_raw_full/clusters/{ff}")

    with open(idx_path) as f:
        sidx = json.load(f)
    size += put(idx_path, "robustness/data/snr_equalised/clusters/index.json")
    for c in cm.LOS_CLUSTERS:
        ff = sidx["clusters"][c]["features_file"]
        size += put(os.path.join(cm.SNR_DATA_DIR, "clusters", ff), f"robustness/data/snr_equalised/clusters/{ff}")

    for rel in ("robustness/common.py", "robustness/run_deep_split.py",
                "robustness/01_snr_equalisation/snr_equalisation.ipynb",
                "robustness/02_frame_vs_window/frame_vs_window.ipynb",
                "CNN_LSTM/cnn_lstm_core.py", "CNN_Transformer/cnn_transformer_core.py"):
        size += put(os.path.join(cm.PV2, rel), rel)

    for model in ("SVM", "RF", "CNN_LSTM", "CNN_Transformer"):
        for p in cm.PROTOCOLS:
            size += put(os.path.join(cm.PV2, model, "results", f"splits_{p}.jsonl"), f"{model}/results/splits_{p}.jsonl")
        if model.startswith("CNN"):
            size += put(os.path.join(cm.PV2, model, "results", "part_b_config.json"), f"{model}/results/part_b_config.json")

    meta = {"title": "LOS-NLOS robustness", "id": "YOUR-KAGGLE-USERNAME/los-nlos-robustness",
            "licenses": [{"name": "other"}]}
    with open(os.path.join(BUNDLE, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    import zipfile
    zpath = BUNDLE + ".zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for dirpath, _, files in os.walk(DST_PV2):
            for fn in files:
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.relpath(full, BUNDLE))

    print(f"\nbundle written to {BUNDLE}  ({size / 1e9:.2f} GB)")
    print(f"zip for upload:  {zpath}  ({os.path.getsize(zpath) / 1e9:.2f} GB)")
    print("next: upload the zip as a private Kaggle dataset")


if __name__ == "__main__":
    main()
