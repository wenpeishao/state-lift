# Per-domain hyperparameter check for low-SL rows (Reviewer MmQK)

Completed 2026-09-27 02:27 CDT. SLURM, Llama-3.1-8B-Instruct + LoRA r=8,
identical settings for both arms except lr/epochs; dialogue-level splits; metric = best eval accuracy over epochs
(so epochs=5 is >= epochs=3 by construction). Seeds 42, 43. Source files: results/multiseed/<dom>_hp_lr<lr>_ep<ep>_seed<s>.json.

Baselines (lr 2e-5, 3 ep, all reported seeds): ESConv +0.008 ± 0.009 (n=8); HH-RLHF +0.030 ± 0.012 (n=8); CaSiNo-dsplit +0.027 ± 0.022 (n=13).

| domain | lr | epochs | seed42 lift | seed43 lift | mean | baseline mean ± SD |
|---|---|---|---|---|---|---|
| ESConv | 5e-5 | 3 | +0.008 | −0.003 | +0.003 | +0.008 ± 0.009 |
| ESConv | 5e-5 | 5 | +0.008 | −0.003 | +0.003 | |
| ESConv | 2e-5 | 5 | +0.012 | +0.010 | +0.011 | |
| HH-RLHF | 5e-5 | 3 | +0.030 | +0.035 | +0.033 | +0.030 ± 0.012 |
| HH-RLHF | 5e-5 | 5 | +0.030 | +0.035 | +0.033 | |
| HH-RLHF | 2e-5 | 5 | +0.035 | +0.028 | +0.031 | |
| CaSiNo (dsplit) | 5e-5 | 3 | +0.009 | +0.035 | +0.022 | +0.027 ± 0.022 |
| CaSiNo (dsplit) | 5e-5 | 5 | +0.019 | +0.040 | +0.030 | |
| CaSiNo (dsplit) | 2e-5 | 5 | +0.020 | +0.035 | +0.028 | |

Raw accuracies (blind / conditioned):
ESConv s42: 5e-5/3 0.5764/0.5844; 5e-5/5 0.5764/0.5844; 2e-5/5 0.562/0.574. s43: 5e-5/3 0.5888/0.586; 5e-5/5 0.5888/0.586; 2e-5/5 0.574/0.5836.
HH s42: 5e-5/3 0.6273/0.6573; 5e-5/5 0.6273/0.6573; 2e-5/5 0.622/0.6567. s43: 5e-5/3 0.6493/0.6847; 5e-5/5 0.6493/0.6847; 2e-5/5 0.6307/0.6587.
CaSiNo s42: 5e-5/3 0.527/0.5364; 5e-5/5 0.527/0.5458; 2e-5/5 0.5162/0.5364. s43: 5e-5/3 0.531/0.566; 5e-5/5 0.531/0.5714; 2e-5/5 0.5296/0.5647.

Conclusion: no configuration moves any domain's mean lift by more than 0.005 from its baseline; every value lies within
one baseline SD. Identical ep3/ep5 values at lr 5e-5 mean the best epoch was <= 3 in both arms. The small lifts on the
low-SL rows are not an artifact of under-tuning the blind arm.
