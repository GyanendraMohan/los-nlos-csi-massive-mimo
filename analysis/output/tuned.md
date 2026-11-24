# Tuned SVM and RF: report

Every number below is also in `tuned_summary.json`. Accuracy in %, mean ± sd over the splits.

## Checks

- **SVM (tuned)**: run complete = True, test frames equal to the stored model, stored default equal to `SVM/results`, search complete (960 LOPO and 2160 L2PO tasks, no test cluster in any inner fold), default chosen on 3 of 80 splits (max. difference to the stored model 0.0e+00).
- **RF (tuned)**: run complete = True, test frames equal to the stored model, stored default equal to `RF/results`, search complete (720 LOPO and 1620 L2PO tasks, no test cluster in any inner fold), default chosen on 0 of 80 splits.

## Main results

| Method | Unit | LOPO (20) | L2PO (60) | All (80) | 95% CI (all) | Median | Worst | Macro-F1 | LOS rec. | NLOS rec. |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN-Transformer | window | 99.71 ± 0.55 | 96.58 ± 6.34 | 97.36 ± 5.67 | [96.02, 98.52] | 99.85 | 75.33 | 97.18 | 97.31 | 97.61 |
| CNN-LSTM | window | 99.56 ± 0.74 | 96.01 ± 6.24 | 96.90 ± 5.63 | [95.63, 98.05] | 99.65 | 74.12 | 96.69 | 96.69 | 97.47 |
| Zheng SVM | frame | 95.41 ± 4.29 | 92.17 ± 5.33 | 92.98 ± 5.28 | [91.79, 94.11] | 93.35 | 77.48 | 92.61 | 90.34 | 98.05 |
| SVM (tuned) | frame | 96.15 ± 3.78 | 92.72 ± 8.45 | 93.57 ± 7.70 | [91.81, 95.17] | 96.32 | 61.02 | 93.06 | 93.69 | 93.10 |
| SVM | frame | 90.86 ± 9.06 | 90.08 ± 7.06 | 90.27 ± 7.61 | [88.61, 91.91] | 91.67 | 74.08 | 89.71 | 87.91 | 94.41 |
| RF (tuned) | frame | 97.54 ± 4.13 | 93.56 ± 9.13 | 94.55 ± 8.35 | [92.65, 96.32] | 99.25 | 67.86 | 94.39 | 91.68 | 99.99 |
| RF | frame | 94.32 ± 8.15 | 90.39 ± 10.88 | 91.37 ± 10.41 | [89.04, 93.59] | 96.19 | 64.53 | 91.15 | 86.78 | 100.00 |

SVM (tuned) with a 50-frame window vote: 94.10 % (all 80).

RF (tuned) with a 50-frame window vote: 94.90 % (all 80).

## Wilcoxon signed-rank (80 splits, Holm within each tuned model)

| Tuned | vs. | Δ (pp) | W/L/T | r | p | p (Holm) |
|---|---|---|---|---|---|---|
| SVM (tuned) | CNN-Transformer | -3.79 | 7/73/0 | -0.79 | 7.35e-10 | 2.94e-09 |
| SVM (tuned) | CNN-LSTM | -3.32 | 12/68/0 | -0.73 | 1.25e-08 | 2.49e-08 |
| SVM (tuned) | Zheng SVM | +0.59 | 44/36/0 | +0.18 | 1.59e-01 | 1.59e-01 |
| SVM (tuned) | SVM | +3.30 | 56/21/3 | +0.76 | 7.67e-09 | 2.30e-08 |
| RF (tuned) | CNN-Transformer | -2.81 | 19/53/8 | -0.59 | 1.59e-05 | 4.78e-05 |
| RF (tuned) | CNN-LSTM | -2.35 | 23/47/10 | -0.46 | 7.90e-04 | 1.58e-03 |
| RF (tuned) | Zheng SVM | +1.57 | 59/21/0 | +0.38 | 3.38e-03 | 3.38e-03 |
| RF (tuned) | RF | +3.18 | 72/3/5 | +0.90 | 1.08e-11 | 4.30e-11 |

**Friedman (CNN-Transformer, CNN-LSTM, SVM (tuned), RF (tuned), Zheng SVM):** χ² = 125.72, p = 3.20e-26, CD = 0.68 (k = 5, n = 80).
Mean ranks: CNN-Transformer 1.90, CNN-LSTM 2.19, RF (tuned) 2.95, SVM (tuned) 3.89, Zheng SVM 4.08

## Chosen settings

**SVM (tuned)**

| Setting | LOPO | L2PO |
|---|---|---|
| C = 1, γ = 0.1 × scale | 5 | 19 |
| C = 10, γ = 0.1 × scale | 5 | 14 |
| C = 100, γ = 0.1 × scale | 8 | 10 |
| C = 0.1, γ = 0.1 × scale | 2 | 7 |
| C = 0.1, γ = 1 × scale | 0 | 5 |
| C = 1, γ = 1 × scale | 0 | 3 |
| C = 10, γ = 1 × scale | 0 | 1 |
| C = 1, γ = 10 × scale | 0 | 1 |

**RF (tuned)**

| Setting | LOPO | L2PO |
|---|---|---|
| max_features = 0.05, min_samples_leaf = 20 | 16 | 41 |
| max_features = 0.05, min_samples_leaf = 1 | 4 | 16 |
| max_features = log2, min_samples_leaf = 1 | 0 | 2 |
| max_features = log2, min_samples_leaf = 20 | 0 | 1 |

## SNR condition B (trained on original data, LOS test clusters noise-equalised)

| Method | Protocol | Original | (B) | LOS rec. orig. | LOS rec. (B) |
|---|---|---|---|---|---|
| CNN-Transformer | LOPO | 98.90 | 97.38 ± 4.98 | 98.8 | 96.4 |
| CNN-Transformer | L2PO | 96.16 | 95.57 ± 8.12 | 95.9 | 95.0 |
| CNN-LSTM | LOPO | 99.73 | 99.47 ± 0.69 | 99.9 | 99.5 |
| CNN-LSTM | L2PO | 96.09 | 94.78 ± 7.47 | 95.9 | 93.9 |
| SVM (tuned) | LOPO | 96.15 | 93.83 ± 4.41 | 97.4 | 93.8 |
| SVM (tuned) | L2PO | 92.72 | 89.76 ± 11.05 | 92.5 | 87.9 |
| SVM | LOPO | 90.86 | 85.10 ± 9.63 | 89.0 | 80.2 |
| SVM | L2PO | 90.08 | 83.28 ± 7.57 | 87.5 | 77.0 |
| RF (tuned) | LOPO | 97.54 | 70.54 ± 16.32 | 96.3 | 54.9 |
| RF (tuned) | L2PO | 93.56 | 65.44 ± 14.70 | 90.2 | 47.0 |
| RF | LOPO | 94.32 | 62.94 ± 20.65 | 91.3 | 43.2 |
| RF | L2PO | 90.40 | 58.02 ± 14.61 | 85.3 | 35.6 |

| Method | Original (80) | (B) (80) | Δ (pp) | p | p (Holm) |
|---|---|---|---|---|---|
| CNN-Transformer | 96.84 | 96.02 | -0.82 | 1.73e-06 | 1.73e-06 |
| CNN-LSTM | 97.00 | 95.95 | -1.05 | 7.55e-10 | 1.51e-09 |
| SVM (tuned) | 93.57 | 90.78 | -2.80 | 1.15e-14 | 4.71e-14 |
| SVM | 90.27 | 83.73 | -6.54 | 7.85e-15 | 4.71e-14 |
| RF (tuned) | 94.55 | 66.71 | -27.84 | 7.85e-15 | 4.71e-14 |
| RF | 91.38 | 59.25 | -32.13 | 7.85e-15 | 4.71e-14 |

| Comparison under (B) | Δ (pp) | p |
|---|---|---|
| SVM (tuned) vs SVM | +7.04 | 6.14e-13 |
| SVM (tuned) vs CNN-Transformer | -5.24 | 5.59e-10 |
| SVM (tuned) vs CNN-LSTM | -5.18 | 1.04e-10 |
| RF (tuned) vs RF | +7.46 | 1.73e-14 |
| RF (tuned) vs CNN-Transformer | -29.31 | 9.13e-15 |
| RF (tuned) vs CNN-LSTM | -29.24 | 8.15e-15 |
