# LOS/NLOS classification in unseen environments from massive-MIMO CSI

Code and results for line-of-sight (LOS) and non-line-of-sight (NLOS) classification on the open Argos massive-MIMO dataset (4 LOS and 5 NLOS clusters, 64 base-station antennas, 52 subcarriers).

Every model is tested only on clusters it never saw in training, over 80 splits:

- **LOPO** (`lopo`): 20 splits that hold out one LOS and one NLOS cluster.
- **L2PO** (`l2po`): 60 splits that hold out two LOS and two NLOS clusters.

## Results

Mean accuracy (%) on the held-out clusters, from the stored results in this repository. The SVM and RF classify single frames. The sequence models and the Nguyen et al. models classify windows of 50 frames.

| Model | LOPO (20) | L2PO (60) | All 80 | Worst split |
|---|---|---|---|---|
| CNN-Transformer | 99.71 | 96.58 | 97.36 | 75.33 |
| CNN-LSTM | 99.56 | 96.01 | 96.90 | 74.12 |
| RF, tuned | 97.54 | 93.56 | 94.55 | 67.86 |
| SVM, tuned | 96.15 | 92.72 | 93.57 | 61.02 |
| Nguyen et al. (2018), our input | 97.01 | 91.79 | 93.10 | 32.66 |
| Zheng et al. (2023) | 95.41 | 92.17 | 92.98 | 77.48 |
| RF | 94.32 | 90.39 | 91.37 | 64.53 |
| SVM | 90.86 | 90.08 | 90.27 | 74.08 |
| Nguyen et al. (2018), their input | 65.41 | 65.25 | 65.29 | 57.37 |

The full statistics are in `analysis/output/` and the figures in `analysis/figures/`. They include 95% bootstrap confidence intervals, Wilcoxon signed-rank tests with Holm correction, Friedman ranks with Nemenyi critical differences, per-cluster accuracy, confusion matrices, the SNR-equalisation and frame-versus-window results, and the reports on the tuned and re-implemented models.

## Statistics and figures from the stored results

These scripts need neither the dataset nor any training, and they take under a minute.

```bash
conda activate los_nlos
python analysis/overall.py
python analysis/robustness.py
python analysis/reported.py
python analysis/tuned.py
python analysis/zheng.py
```

| Script | Output |
|---|---|
| `overall.py` | `overall.md`, `pairwise.md`, `per_cluster.md`, `nguyen_comparison.md`, `cost.md`, `overall_summary.json` and the main figures |
| `robustness.py` | `snr.md`, `snr_tests.md`, `frame_vs_window.md`, `robustness_summary.json` and two figures |
| `reported.py` | `reported.md`, `reported_summary.json` (accuracy reported by Nguyen et al. and Zheng et al., in the same environment and on unseen clusters) |
| `tuned.py` | `tuned.md`, `tuned_summary.json` (checks, tests, selected settings and SNR condition B of the tuned SVM and RF) |
| `zheng.py` | `zheng.md`, `zheng_summary.json` and a critical-difference figure |

## Repository layout

| Folder | Content |
|---|---|
| `preprocess_raw.ipynb` | Raw I/Q traces to calibrated CSI and per-cluster features |
| `SVM/`, `RF/` | Frame classifiers on the 3328 per-antenna-normalised amplitudes of a frame |
| `CNN_LSTM/`, `CNN_Transformer/` | Sequence models on 50-frame windows of 52 antenna-averaged amplitudes |
| `Tuned_SVM_RF/` | SVM and RF with nested, cluster-wise hyperparameter search |
| `Nguyen2018_CNNLSTM/` | Re-implementation of the CNN-LSTM of Nguyen et al., *Sensors* 18(11), 4057, 2018 |
| `Zheng2023_SVM/` | Re-implementation of the SVM detector of Zheng et al., *Remote Sensing* 15(10), 2676, 2023 |
| `robustness/01_snr_equalisation/` | Training and testing with the LOS clusters noise-equalised to the NLOS SNR |
| `robustness/02_frame_vs_window/` | SVM and RF frame decisions voted over the same 50-frame windows |
| `data_exploration/` | Plots of the CSI across subcarriers, antennas and time |
| `analysis/` | Statistics, tables and figures over all models, computed from the stored results |
| `raw_traces/` | Place for the raw dataset files (not included) |

Each experiment folder has a `results/` folder. The files `splits_lopo.jsonl` and `splits_l2po.jsonl` hold one JSON line per split, with the training and test clusters, accuracy, macro-F1, ROC-AUC, recall per class and the accuracy of each held-out cluster. `splits_summary.json` and `summary.json` hold the means over the splits.

## Installation

Two environments match the original setup. The sequence models ran with Python 3.11, and the rest with Python 3.13.

```bash
conda create -n los_nlos python=3.13
conda activate los_nlos
pip install -r requirements.txt

conda create -n los_nlos_deep python=3.11
conda activate los_nlos_deep
pip install -r requirements-deep.txt
```

Use `los_nlos_deep` for `CNN_LSTM`, `CNN_Transformer`, `Nguyen2018_CNNLSTM` and `robustness/01_snr_equalisation`, and `los_nlos` for everything else.

## Data

Download the Argos dataset of Du and Sabharwal (IEEE Transactions on Vehicular Technology, 2021, doi:10.1109/TVT.2021.3131606) from [DATASET_URL]. Place the nine raw trace files `channe14_LOS_cluster1.hdf5` to `channe14_NLOS_cluster5.hdf5` (about 17 GB) in `raw_traces/`.

## Running

Run each notebook from its own folder, with Run All in Jupyter or from the command line.

```bash
jupyter nbconvert --to notebook --execute --inplace preprocess_raw.ipynb

cd SVM
jupyter nbconvert --to notebook --execute --inplace svm_los_nlos.ipynb
```

The order and the approximate run time on a laptop CPU are as follows.

| Step | Notebook or script | Time |
|---|---|---|
| 1 | `preprocess_raw.ipynb` | under 1 min, writes about 2 GB to `output_raw_full/clusters/` |
| 2 | `SVM/svm_los_nlos.ipynb`, `RF/rf_los_nlos.ipynb` | about 15 and 30 min |
| 3 | `CNN_LSTM/cnn_lstm_los_nlos.ipynb`, `CNN_Transformer/cnn_transformer_los_nlos.ipynb` | about 2 and 4 h |
| 4 | `Nguyen2018_CNNLSTM/nguyen2018_cnnlstm.ipynb` | 2 to 4 h per variant |
| 5 | `robustness/02_frame_vs_window/frame_vs_window.ipynb` | about 2 h |
| 6 | `robustness/01_snr_equalisation/snr_equalisation.ipynb` | 2 to 3 h for SVM and RF, about 6 h per condition for the sequence models on a GPU |
| 7 | `python Zheng2023_SVM/prepare_inputs.py`, then `Zheng2023_SVM/zheng2023_svm.ipynb` | 15 min, then several hours |
| 8 | `Tuned_SVM_RF/tuned_svm_rf.ipynb` | about 6 h for the SVM and 9 h for the RF on 4 cores |
| 9 | The `analysis/` scripts above | under 1 min |

Every split is written to the results file as soon as it finishes, and a stopped notebook continues with the missing splits when it runs again. Delete the `splits_*.jsonl` files of a folder before running it with changed settings. Steps 6 and 7 need the raw traces, and Steps 7 and 8 use the noise-equalised data built in Step 6. Steps 6 to 8 also run on Kaggle, where the notebooks find their inputs under `/kaggle/input` automatically, and `robustness/make_kaggle_bundle.py` builds the upload folder.

The deep models use seed 42 for the final model of every split and 42 + k for inner fold k. Results on other hardware or on a GPU can differ slightly from the stored CPU results.

## Dataset citation

X. Du and A. Sabharwal, "Massive MIMO Channels with Inter-User Angle Correlation: Open-Access Dataset, Analysis and Measurement-Based Validation," *IEEE Transactions on Vehicular Technology*, 2021, doi:10.1109/TVT.2021.3131606.

## License

[LICENSE]
