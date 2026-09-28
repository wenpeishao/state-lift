# State-Lift: When Does Knowing the State Help?

Code and results for

> Wenpei Shao and Ross Jacobucci. **When Does Knowing the State Help? Diagnosing Process vs. Outcome Reward Design.** *Advances in Neural Information Processing Systems (NeurIPS)*, 2026.

```bibtex
@inproceedings{shao2026statelift,
  title     = {When Does Knowing the State Help? Diagnosing Process vs. Outcome Reward Design},
  author    = {Shao, Wenpei and Jacobucci, Ross},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## What is state-lift?

State-lift (SL) is a cheap, model-free diagnostic for deciding whether a sequential task needs a
process reward model (PRM, state-conditioned) or whether an outcome reward model (ORM, state-blind)
is enough. Given ~500 labelled steps, we embed each step's state and action with a sentence
encoder and fit two Ridge regressions on the step quality label: one from the action alone and one
from state plus action. The difference `SL = R^2(s+a) - R^2(a)` measures how much of the quality
signal is only readable once you know the state; the effective gap `sqrt((1-SL)/SL)` maps it to a
regret bound on state-blind reward. `SL > 0.10` supports investing in a PRM; a near-zero reading is
not read as "ORM sufficient" but escalated to a short truncated-training check (the early-training
signal, `B*`). The paper computes SL on ten evaluation domains, validates it against matched
Llama-3.1-8B LoRA reward-model training, characterises where the probe fails (fine-grained math
step grading), and audits the proxy pitfalls that make naive versions of the measurement
misleading.

## Main result

State-lift on the paper's evaluation domains (Table 2 of the paper; MiniLM-L6 + PCA-16 + Ridge,
trajectory-grouped folds where the label is trajectory-level). The context-correctness rows are
upper bounds on the mechanism; no genuine-label domain clears the 0.10 threshold.

| Domain | Label type | SL | Script |
|---|---|---|---|
| Math reasoning (GSM8K) | context-correctness | 0.584 | `scripts/statelift/t10_mechanism_alignment.py` |
| Tool-use agents (Glaive) | context-correctness | 0.398 | `scripts/statelift/t8b_agent_domains.py` |
| Code reasoning (CodeContests) | context-correctness | 0.003 | `scripts/statelift/t10_mechanism_alignment.py` |
| DealOrNoDeal | deal outcome | 0.057 | `scripts/statelift/r19b_sl_r17d_recipe_cv.py` |
| ESConv | strategy progress | 0.028 | `scripts/statelift/t10_mechanism_alignment.py` |
| ProsocialDialog | rule-of-thumb adherence | 0.024 | `scripts/statelift/r26_prosocial_sl_grouped.py` |
| USS-Satisfaction | turn satisfaction | 0.012 | `scripts/statelift/r24_uss_sl_grouped.py` |
| CraigslistBargain | deal outcome | 0.003 | `scripts/statelift/r29_craigslist_prereg_sl.py` |
| AirDialogue | booking outcome | 0.001 | `scripts/statelift/r25_airdialogue_sl_grouped.py` |
| CaSiNo | deal outcome | -0.003 | `scripts/statelift/r19b_sl_r17d_recipe_cv.py` |
| PersuasionForGood | donation outcome | -0.001 | `scripts/statelift/r22_p4g_sl_grouped.py` |
| HH-RLHF | preference | -0.003 | `scripts/statelift/t10_mechanism_alignment.py` |

Matched Llama-3.1-8B LoRA reward-model training (Table 3) and the truncated-training escalation
(Table 11) are produced by `scripts/training/e6_multiseed_sscc.py` and
`scripts/statelift/r48_ets_analysis.py`; the per-seed outputs are bundled under `results/multiseed/`.

## Quick start

Everything below runs on CPU from the repository root.

```bash
# 1. State-lift on GSM8K, ESConv, HH-RLHF and CodeContests (downloads the datasets from the HF hub,
#    embeds with MiniLM; ~10-20 min on a laptop). Writes results/t10_mechanism_alignment.json.
python scripts/statelift/t10_mechanism_alignment.py

# 2. Admissibility audit (Table 1) on any preprocessed pairwise .pt file.
python scripts/statelift/admissibility_audit.py --pt data/e4_hh_data.pt

# 3. Truncated-training read (Table 11) from the bundled training curves; instant.
python scripts/statelift/r48_ets_analysis.py

# 4. Regenerate the paper figures into figures/out/.
python figures/gen_fig1_cr.py && python figures/gen_fig_sl_vs_lift_cr2.py && python figures/gen_all_nature_style.py
```

## Layout

```
scripts/data/       preprocessing and downloads (build data/*.pt from public datasets)
scripts/statelift/  CPU probes: state-lift, admissibility, robustness, calibration
scripts/training/   GPU: Llama-3.1-8B LoRA reward models (blind vs. state-conditioned), SLURM wrappers
figures/            camera-ready figure generators (+ nature_style.py / paper_plot_style.py)
figures/source/     fig0_headline.png, the hand-drawn base image edited by gen_fig0_headline_cr.py
figures/out/        where regenerated figures land (git-ignored)
results/            small JSON/CSV/MD result files (registry/, capacity/, multiseed/ kept)
data/               datasets and preprocessed .pt files (git-ignored, see below)
logs/               SLURM output (git-ignored)
```

All scripts are meant to be run **from the repository root** with relative paths, e.g.
`python scripts/statelift/r48_ets_analysis.py`.

## Install

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch`, `transformers`, `peft` and a GPU are only needed for `scripts/training/` and for a few
probes that embed with `sentence-transformers` (CPU is fine for those). The figure scripts need only
`numpy`, `scipy`, `matplotlib`, `adjustText` and `pillow`.

## Data

### Downloaded automatically through HF `datasets`

The preprocessing and probe scripts call `datasets.load_dataset` for: `openai/gsm8k`, `casino`,
`thu-coai/esconv`, `Anthropic/hh-rlhf`, `osunlp/Mind2Web`, `glaiveai/glaive-function-calling-v2`,
`deepmind/code_contests`, `peiyi9979/Math-Shepherd`, `trl-lib/math_shepherd`,
`RLHFlow/Mistral-PRM-Data`, `DigitalLearningGmbH/PRM800K`, `Qwen/ProcessBench`, `eth-nlped/mathdial`,
`google/air_dialogue`, `allenai/prosocial-dialog`, `budzianowski/multiwoz_v21`, `facebook/dond` and a
few others named in the individual scripts. No token is required; all are public.

### Manual download

* **PRM800K**: put `phase2_train.jsonl` from
  <https://github.com/openai/prm800k> into `data/PRM800K/`
  (used by `scripts/data/r5_step_beam_preprocess.py`, `scripts/statelift/r2_prm800k_statelift.py`,
  `r9_prm800k_nonlinear.py`, `scripts/training/r11_llm_feature_sl.py`).
* **DealOrNoDeal** (`data/dealornodeal_train.txt`) and **PersuasionForGood**
  (`data/persuasion_full.csv`): run `bash scripts/data/download_data.sh`.
  `e9_preprocess_p4g.py` / `r22_p4g_sl_grouped.py` additionally expect `data/persuasion_info.csv`
  (the `Dialog Info` sheet of the same PersuasionForGood release).

### Preprocessed `.pt` files

`scripts/data/` rebuilds the following into `data/`:

| file | builder |
|---|---|
| `e4_math_data.pt` | `e4_preprocess_math.py` |
| `e4_tool_data.pt` | `e4_preprocess_tool.py` |
| `e4_code_data.pt` | `e4_preprocess_code.py` |
| `e4_hh_data.pt` | `e4_preprocess_hh.py` |
| `e4_casino_data.pt` | `e4_preprocess_casino.py` |
| `e5_dealornodeal_data.pt` | `e5_preprocess_dealornodeal.py` |
| `e9_p4g_data.pt` | `e9_preprocess_p4g.py` |
| `r5_step_data.pt` | `r5_step_beam_preprocess.py` |
| `e17_prm800k_pairs.pt` | `e17_prm800k_pairs.py` (needs `r5_step_data.pt`) |
| `r6_gsm8k_problems.pt` | `r6_preprocess_local.py` |

**Not included.** The following preprocessed files are referenced by scripts in this release but
their builder scripts are not part of it. They are available on request from the authors:
`e4_esconv_data.pt`, `e8_casino_dsplit_data.pt`, `e8_dealornodeal_dsplit_data.pt`,
the P4G variants `e9b_p4g_censored_data.pt` / `e9c_p4g_truncated_data.pt` /
`e9d_p4g_lastk_data.pt` / `e9e_p4g_pdisjoint_data.pt`, `e10_craigslist_data.pt`,
`e10b_craigslist_ldisjoint_data.pt`, `e11_multiwoz_data.pt`, `e11d_multiwoz_precode_data.pt`, and
the constructed constraints / arithmetic domains `e12_constraints_data.pt`,
`e13_constraints_v2_data.pt`, `e14_arithmetic_data.pt`, `e15_constraints_v3_data.pt`,
`e16_constraints_balanced.pt` (plus `e18_additive_*.pt`, `e19_salient_*.pt`).

## Paper artifact -> script -> results file

Numbering follows the camera-ready. CPU = runs on a laptop in minutes; GPU = needs one
Llama-3.1-8B-capable GPU (set `LLAMA_PATH` to a local checkpoint, default is the HF hub id
`meta-llama/Llama-3.1-8B-Instruct`).

| Artifact | Script(s) | Results file(s) | Compute |
|---|---|---|---|
| Table 1 (`tab:icc`, admissibility audit) | `scripts/statelift/admissibility_audit.py` | `results/admissibility_audit.json` | CPU |
| Table 2 (`tab:statelift`, state-lift registry) | `scripts/statelift/t8b_agent_domains.py`, `t6c_mind2web_and_negotiation.py`, `r19_sl_dialogue_cv.py`, `r19b_sl_r17d_recipe_cv.py`, `r21c_casino_ctxcorr_exact_t8b.py`, `r22_p4g_sl_grouped.py`, `r24_uss_sl_grouped.py`, `r25_airdialogue_sl_grouped.py`, `r26_prosocial_sl_grouped.py`, `r29_craigslist_prereg_sl.py`, `r2_prm800k_statelift.py`; corrected (at-capacity) entries: `scripts/training/r40_registry_at_capacity.py`, `r36_capacity_ladder.py` | `results/t8b_agent_domains.json`, `t6c_mind2web_negotiation.json`, `r19_sl_dialogue_cv.json`, `r19b_sl_r17d_recipe_cv.json`, `r21c_casino_ctxcorr_exact_t8b.json`, `r22_..`/`r24_..`/`r25_..`/`r26_..`/`r29_..json`, `r2_prm800k_statelift.json`, `results/registry/*.json`, `results/capacity/*.json` | CPU (r36/r40: GPU) |
| Table 3 (`tab:llm`, blind vs. conditioned training) | context-correctness rows: `scripts/training/e4_{math,tool,code,hh}_train.py` (+ `e4_*_slurm.sh`); genuine-label rows: `e6_multiseed_train.py` / `e6_multiseed_sscc.py` (+ `submit_multiseed.sh`) | `results/multiseed/<domain>_seed*.json`, `results/multiseed/AGGREGATE.json`; `results/e4_{math,tool,code,hh}_results.json` | GPU |
| Table 4 (`tab:signals`), Table 5 (`tab:datasets`) | descriptive; `n` comes from the `scripts/data/` builders | -- | -- |
| Table 6 (`tab:artifacts`, shuffled-state controls) | `scripts/statelift/t5b_artifact_check.py`, `t6b_fill_gaps_and_artifacts.py` | `results/t5b_artifact_check.json`, `results/t6b_fill_gaps_artifacts.json` | CPU |
| Table 7 (`tab:downstream`, GSM8K beam search) | `scripts/data/r6_preprocess_local.py`, `scripts/training/r6_matched_downstream.py` (+ `r6_matched_slurm.sh`); best-of-N on low-SL domains: `r51_downstream_bon.py` | `results/r6_matched_downstream.json`; `results/r51_bon_*_seed*.json` | GPU |
| Table 8 (`tab:boundary_datasets`, four math step-grading sets) | `scripts/statelift/r2_prm800k_statelift.py`, `r12c_proper_mathshepherd.py`, `r14d_more_math.py` | `results/r2_prm800k_statelift.json`, `results/r12c_combined.json`, `results/r12c_mathshepherd_proper.json` | CPU |
| Table 9 (`tab:encoders`, eight encoding architectures) | `scripts/statelift/r13b_math_sl_methods.py`, `r9_prm800k_nonlinear.py`; frozen-LLM features: `scripts/training/r11_llm_feature_sl.py` (+ `r11_llm_feature_slurm.sh`) | `results/r13b_results.json`, `results/r9_prm800k_nonlinear.json`, `results/r11_llm_feature_sl.json`, `results/r11_resolution_comparison.csv` | CPU (r11: GPU) |
| Table 10 (`tab:training_curves`, per-epoch metrics) | `scripts/training/e6_multiseed_sscc.py` | `results/multiseed/<domain>_seed*.json` (`blind_curve`, `cond_curve`) | GPU |
| Table 11 (`tab:bstar`, truncated-training read) | `scripts/statelift/r48_ets_analysis.py` | `results/r48_ets_analysis.json` (reads `results/multiseed/*_curve_seed*.json`, `*_shufcurve_*`) | CPU |
| Table 12 (`tab:hpcheck`, hyperparameter check) | `scripts/training/e6_multiseed_sscc.py --lr .. --epochs ..` | `results/multiseed/*_hp_*_smoke.json`, `results/multiseed/HP_CHECK_SUMMARY_2026-09-27.md` | GPU |
| Table 13 (`tab:stratified`, HH-RLHF by state distance) | `scripts/statelift/r10_hh_heterogeneity.py` | `results/r10_hh_heterogeneity.json` | CPU |
| Table 14 (`tab:litfull`, published PRM vs. ORM) | `scripts/statelift/t6f_literature_calibration.py` (+ `t6e_standardized_prediction.py`) | `results/t6f_literature_calibration.json`, `results/t6e_standardized.json` | CPU |
| Table 15 (`tab:calibration`), Table 16 (`tab:bootstrap`) | `scripts/statelift/r1_robustness.py` | `results/r1_robustness.json` | CPU |
| Table 17 (`tab:mixedeffects`, estimator comparison) | `scripts/statelift/r1_robustness.py` (variance-decomposition block), `t7_construct_validation.py`, `r49_calibration_fixed.py` | `results/r1_robustness.json`, `results/t7_construct_validation.json`, `results/r49_calibration_fixed.json` | CPU |
| Table 18 (`tab:distrobust`, regret-bound robustness) | `scripts/statelift/r4_correlated_heavy_tail.py`, `t9b_lower_bound.py` | `results/r4_correlated_heavy_tail.json`, `results/t9b_lower_bound.json` | CPU |
| Figure 1 (`fig:headline`, pipeline) | `figures/gen_fig0_headline_cr.py` (edits `figures/source/fig0_headline.png`) | `figures/out/fig0_headline_cr.{png,pdf}` | CPU |
| Figure 2 (`fig:regret`, regret bound) | `figures/gen_fig1_cr.py` | `figures/out/fig1_regret_bound.*` | CPU |
| Figure 3 (`fig:proxy`, proxy pitfall) | `figures/gen_fig3_proxy_pitfall.py` | `figures/out/fig3_proxy_pitfall.*` | CPU |
| Figure 4 (`fig:cost`, SL vs. trained lift) | `figures/gen_fig_sl_vs_lift_cr2.py` | `figures/out/fig_sl_vs_lift_cr.*` | CPU |
| Figure 5 (`fig:artifacts`), Figure 8 (`fig:samplesize`), Figure 9 (`fig:encoders`), Figure 10 (`fig:persistence`), Figure 11 (`fig:nonlinear`), Figure 6 (`fig:resolution`) | `figures/gen_all_nature_style.py` (reads `results/r1_robustness.json`) | `figures/out/fig6_artifact_checks.*`, `fig7_sample_size.*`, `fig8_encoder_sensitivity.*`, `fig9_persistence.*`, `fig10_nonlinear.*`, `fig_r11_resolution.*` | CPU |
| Figure 7 (`fig:effgap`, two regimes) | `figures/gen_fig5_cr.py` | `figures/out/fig5_effective_gap.*` | CPU |
| Robustness / audit appendices | `scripts/statelift/t10_mechanism_alignment.py`, `r20_segmentation_sensitivity.py`, `r49_additive_confound_audit.py` | `results/t10_mechanism_alignment.json`, `results/r20_segmentation_sensitivity.json`, `results/r49_additive_confound_audit.json` | CPU |

Note on output locations: the `t*`-series probes write their JSON to `results/` and keep embedding caches (`*.npz`) in `data/`. The `e4_*_train.py` trainers write their result JSON next to their input `.pt` files in `data/`; the copies under `results/` were moved there by hand for the paper.

## Running the GPU scripts

```bash
export LLAMA_PATH=/path/to/Llama-3.1-8B-Instruct      # optional; default: meta-llama/Llama-3.1-8B-Instruct (HF hub)
export CONDA_ENV=statelift SLURM_PARTITION=gpu          # used by scripts/training/*_slurm.sh
sbatch -p "$SLURM_PARTITION" scripts/training/e4_math_slurm.sh
python scripts/training/e6_multiseed_sscc.py --domain arithmetic --seed 42
```

SLURM does not expand shell variables inside `#SBATCH` directives, so pass the partition with
`sbatch -p ...` (the `${SLURM_PARTITION:-gpu}` in the header is a placeholder to edit). Set
`HF_HUB_OFFLINE=1` only if `LLAMA_PATH` points to a local directory.

The per-seed result JSONs and training curves of every run in the paper are bundled under
`results/multiseed/`; the LoRA adapter weights themselves are not (about 300 MB) and are available
on request.

## License

MIT, see `LICENSE`.

## Contact

Wenpei Shao, wshao33@wisc.edu
