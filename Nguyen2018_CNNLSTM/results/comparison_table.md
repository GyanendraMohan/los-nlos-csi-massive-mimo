| Model | Data / protocol | Threshold | LOPO accuracy (%) | L2PO accuracy (%) |
|---|---|---|---|---|
| Nguyen CNN-LSTM (complex CSI + RSSI), as reported | their building, random 70/15/15 split | tuned on their data | 96.53 (single split) | — |
| Nguyen CNN-LSTM* (amplitude + RSSI), as reported | their building, random 70/15/15 split | tuned on their data | 95.97 (single split) | — |
| CNN-Transformer | Argos, unseen clusters (same 80 splits) | 0.5 | 99.71 ± 0.55 | 96.58 ± 6.34 |
| CNN-LSTM | Argos, unseen clusters (same 80 splits) | 0.5 | 99.56 ± 0.74 | 96.01 ± 6.24 |
| RF | Argos, unseen clusters (same 80 splits) | 0.5 | 94.32 ± 8.15 | 90.39 ± 10.88 |
| SVM | Argos, unseen clusters (same 80 splits) | 0.5 | 90.86 ± 9.06 | 90.08 ± 7.06 |
| Nguyen CNN-LSTM (complex CSI + RSSI) (re-implemented) | Argos, unseen clusters (same 80 splits) | 0.5 | 65.41 ± 3.98 | 65.25 ± 2.63 |
| Nguyen CNN-LSTM (our input) (re-implemented) | Argos, unseen clusters (same 80 splits) | 0.5 | 97.01 ± 5.00 | 91.79 ± 14.80 |
| Nguyen CNN-LSTM (complex CSI + RSSI) (re-implemented) | Argos, unseen clusters | tuned on the test set (optimistic) | 69.19 | 62.77 |
| Nguyen CNN-LSTM (our input) (re-implemented) | Argos, unseen clusters | tuned on the test set (optimistic) | 99.49 | 97.31 |
