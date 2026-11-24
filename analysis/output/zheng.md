# Zheng et al. (2023) re-implementation: report

Every number below is also in `zheng_summary.json`.

## Main results (accuracy %, mean ± sd over splits)

| Method | Unit | LOPO (20) | L2PO (60) | All (80) | 95% CI (all) | Median | Worst |
|---|---|---|---|---|---|---|---|
| CNN-Transformer | window | 99.71 ± 0.55 | 96.58 ± 6.34 | 97.36 ± 5.67 | [96.02, 98.52] | 99.85 | 75.33 |
| CNN-LSTM | window | 99.56 ± 0.74 | 96.01 ± 6.24 | 96.90 ± 5.63 | [95.63, 98.05] | 99.65 | 74.12 |
| RF | frame | 94.32 ± 8.15 | 90.39 ± 10.88 | 91.37 ± 10.41 | [89.03, 93.60] | 96.19 | 64.53 |
| SVM | frame | 90.86 ± 9.06 | 90.08 ± 7.06 | 90.27 ± 7.61 | [88.60, 91.90] | 91.67 | 74.08 |
| Nguyen (our input) | window | 97.01 ± 5.00 | 91.79 ± 14.80 | 93.10 ± 13.25 | [89.97, 95.71] | 99.36 | 32.66 |
| Nguyen (complex + RSSI) | window | 65.41 ± 3.98 | 65.25 ± 2.63 | 65.29 ± 3.03 | [64.62, 65.93] | 67.37 | 57.37 |
| Zheng SVM | frame | 95.41 ± 4.29 | 92.17 ± 5.33 | 92.98 ± 5.28 | [91.80, 94.08] | 93.35 | 77.48 |
| Zheng SVM (window vote) | window | 97.87 ± 3.22 | 94.96 ± 4.79 | 95.69 ± 4.63 | [94.63, 96.68] | 96.48 | 80.84 |
| Rician-K | frame | 90.14 ± 7.38 | 89.74 ± 4.58 | 89.84 ± 5.42 | [88.63, 91.03] | 89.91 | 75.32 |
| Skewness | frame | 66.85 ± 4.50 | 65.73 ± 3.77 | 66.01 ± 3.99 | [65.12, 66.87] | 65.67 | 58.12 |
| Gain only | frame | 54.44 ± 21.86 | 49.79 ± 13.25 | 50.95 ± 15.97 | [47.40, 54.42] | 50.40 | 11.64 |

## Wilcoxon signed-rank (Zheng SVM against the four models, 80 splits, Holm over 4)

| Zheng SVM | vs. | Δ (pp) | W/L/T | r | p | p (Holm) |
|---|---|---|---|---|---|---|
| Zheng SVM | CNN-Transformer | -4.38 | 7/73/0 | -0.85 | 5.16e-11 | 2.07e-10 |
| Zheng SVM | CNN-LSTM | -3.92 | 10/70/0 | -0.79 | 6.71e-10 | 2.01e-09 |
| Zheng SVM | RF | +1.61 | 46/34/0 | +0.19 | 1.44e-01 | 1.44e-01 |
| Zheng SVM | SVM | +2.71 | 55/25/0 | +0.43 | 7.60e-04 | 1.52e-03 |

**Friedman (a) four models + Zheng SVM:** χ² = 150.64, p = 1.48e-31, CD = 0.68 (k = 5, n = 80).
Mean ranks: CNN-Transformer 1.79, CNN-LSTM 1.98, RF 3.44, Zheng SVM 3.52, SVM 4.28

**Friedman (b) four models + Zheng SVM + Rician-K + skewness:** χ² = 310.54, p = 4.52e-64, CD = 1.01 (k = 7, n = 80).
Mean ranks: CNN-Transformer 1.93, CNN-LSTM 2.05, RF 3.69, Zheng SVM 3.89, SVM 4.71, Rician-K 4.79, Skewness 6.94

Critical-difference plot: `analysis/figures/zheng_critical_difference.png`

## All kernels, in the style of their Table 3 (their method uses RBF)

| Protocol | Kernel | NLOS rate (their TP) | LOS rate (their TN) | Accuracy (their precision) | Fit time (s) |
|---|---|---|---|---|---|
| LOPO | rbf | 98.34 | 93.91 | 95.41 | 11 |
| LOPO | linear | 95.81 | 96.82 | 96.51 | 13 |
| LOPO | poly | 97.66 | 94.72 | 95.70 | 6 |
| LOPO | sigmoid | 42.52 | 61.17 | 54.59 | 166 |
| L2PO | rbf | 97.96 | 89.15 | 92.17 | 5 |
| L2PO | linear | 94.97 | 93.58 | 94.04 | 5 |
| L2PO | poly | 97.08 | 90.24 | 92.58 | 3 |
| L2PO | sigmoid | 48.57 | 60.33 | 56.11 | 73 |

## Same environment vs. unseen environments (frame accuracy %)

| Method | Chronological (first 70% / last 15%) | Random 70/30 (optimistic) | Unseen LOPO | Unseen L2PO | Unseen all 80 | Gap (chrono − all) |
|---|---|---|---|---|---|---|
| Zheng SVM | 99.51 | 99.65 | 95.41 | 92.17 | 92.98 | +6.53 |
| SVM | 99.85 | 100.00 | 90.86 | 90.08 | 90.27 | +9.58 |
| Rician-K | 91.45 | 90.95 | 90.14 | 89.74 | 89.84 | +1.61 |
| Skewness | 66.05 | 64.17 | 66.85 | 65.73 | 66.01 | +0.04 |
| Gain only | 56.49 | 61.71 | 54.44 | 49.79 | 50.95 | +5.54 |

## SNR condition B (trained on original data, LOS test clusters noise-equalised)

| Method | Protocol | Original | (B) | LOS rec. orig. | LOS rec. (B) |
|---|---|---|---|---|---|
| Zheng SVM | LOPO | 95.41 ± 4.29 | 63.72 ± 10.63 | 93.9 | 45.3 |
| Zheng SVM | L2PO | 92.17 ± 5.33 | 60.91 ± 7.32 | 89.1 | 41.1 |
| Rician-K | LOPO | 90.14 ± 7.38 | 85.84 ± 8.60 | 91.4 | 84.8 |
| Rician-K | L2PO | 89.74 ± 4.58 | 85.22 ± 6.19 | 91.0 | 84.0 |
| Skewness | LOPO | 66.85 ± 4.50 | 68.12 ± 3.86 | 90.4 | 92.4 |
| Skewness | L2PO | 65.73 ± 3.77 | 67.32 ± 3.20 | 87.1 | 89.5 |
| Gain only | LOPO | 54.44 ± 21.86 | 54.52 ± 21.72 | 74.3 | 74.4 |
| Gain only | L2PO | 49.79 ± 13.25 | 49.88 ± 13.29 | 60.1 | 60.2 |

| Method | Δ all 80 (pp) | median Δ | p | p (Holm over 5) |
|---|---|---|---|---|
| Zheng SVM | -31.37 | -29.49 | 7.85e-15 | 3.14e-14 |
| Rician-K | -4.47 | -3.98 | 2.46e-14 | 7.39e-14 |
| Skewness | +1.51 | +1.43 | 4.96e-12 | 9.92e-12 |
| Gain only | +0.08 | +0.03 | 9.57e-03 | 9.57e-03 |

## Hampel filter: share of values replaced (antenna 0)

| Cluster | Original | SNR-equalised |
|---|---|---|
| LOS_cluster1 | 2.53 % | 4.45 % |
| LOS_cluster2 | 1.75 % | 4.18 % |
| LOS_cluster3 | 2.84 % | 4.67 % |
| LOS_cluster4 | 2.78 % | 4.55 % |
| NLOS_cluster1 | 2.29 % |  |
| NLOS_cluster2 | 2.94 % |  |
| NLOS_cluster3 | 3.08 % |  |
| NLOS_cluster4 | 3.72 % |  |
| NLOS_cluster5 | 2.52 % |  |

## Antenna sensitivity (LOPO, accuracy %)

| Antenna | Position | Zheng SVM | Rician-K | Skewness | Gain only |
|---|---|---|---|---|---|
| 0 | corner | 95.41 | 90.14 | 66.85 | 54.44 |
| 7 | corner | 89.51 | 91.72 | 69.43 | 49.67 |
| 27 | centre | 96.84 | 87.36 | 65.01 | 62.89 |
| 32 | edge | 96.46 | 83.81 | 66.05 | 66.12 |
| 63 | corner | 98.90 | 96.12 | 78.56 | 75.53 |

Spread of the SVM mean accuracy over the 5 antennas: 89.51 to 98.90 % (range 9.39 pp).

## Settings not stated by Zheng et al.

{"hampel": {"half_width": 3, "n_sigmas": 3.0, "mad_scale": 1.4826, "axis": "time, per subcarrier"}, "svc": {"C": 1.0, "gamma": "scale", "degree": 3, "coef0": 0.0}}
