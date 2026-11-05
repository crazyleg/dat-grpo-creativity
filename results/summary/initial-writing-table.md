| Metric | Baseline | LoRA1500 | Full250 |
| --- | ---: | ---: | ---: |
| WritingBench overall, higher better | 8.06 | 8.11 | 8.05 |
| WritingBench style, higher better | 8.46 | 8.60 | 8.40 |
| SteerBench steering error, lower better | 0.5387 | 0.5503 | 0.5633 |
| SteerBench miscalibration, lower better | 0.7769 | 0.7860 | 0.8614 |
| SteerBench side-effect error, lower better | 0.5244 | 0.5270 | 0.4638 |
| SteerBench outputs with no measured movement | 32/128 | 34/128 | 46/128 |
| Side-effect error among moving outputs | 0.6991 | 0.7175 | 0.7238 |

WritingBench means use 62/62/63 fully scored cases; style uses all 64. SteerBench uses 128 edits/model. Moving-output subsets differ across models; that row is descriptive.

| Phi4 comparison | Win credit | 95% interval | Same winner after reversal | First-position preference |
| --- | ---: | --- | ---: | ---: |
| LoRA1500 vs baseline | 52.34% | [46.88%, 57.81%] | 17/64 (26.56%) | 111/128 (86.72%) |
| Full250 vs baseline | 40.63% | [33.59%, 48.44%] | 28/64 (43.75%) | 98/128 (76.56%) |

Both comparisons use 64 story pairs from 32 briefs, each judged in both orders. Intervals are unadjusted 95% brief-cluster bootstraps. Combined first-position preference is 209/256 (81.64%).
