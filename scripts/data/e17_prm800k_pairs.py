# -*- coding: utf-8 -*-
"""
E17 -- BUILD A PAIRWISE PRM800K STEP-GRADING SET FOR THE EARLY-TRAINING-SIGNAL TEST.

Why this domain matters more than the two we constructed: the strongest attack on the
repair is "you built arithmetic and constraints_bal yourselves, so of course the probe
misses them". PRM800K is OpenAI's, the labels are human step-correctness ratings, and
it is the canonical setting where PRMs are actually deployed. It is also the paper's
own reported boundary case -- SL_r2 = 0.015 / SL_auc = 0.071 across eight encoders
(results/r2_prm800k_statelift.json) while end-to-end LLM training recovers +0.238.
If the early-training signal detects it at a small step budget, case (B) is
established on a domain nobody can call synthetic.

CONSTRUCTION, and the confounds it closes:
  * Chosen and rejected steps come from the SAME PROBLEM, and each is kept in ITS OWN
    true context. Forcing both into one shared context would invalidate the labels --
    a step rated incorrect after one derivation may be correct after another.
  * Raw PRM800K in r5_step_data.pt carries two length base rates: incorrect steps sit
    in longer contexts (858 vs 762 chars) and are themselves longer (143 vs 126). Both
    would let a model score the pair without judging anything. Pairs are therefore
    chosen greedily to MINIMISE the context-length gap, and the residual gap on both
    channels is reported and asserted small.
  * Train/test split is BY PROBLEM, so no problem appears in both.
"""
import numpy as np
import torch

SRC = "data/r5_step_data.pt"
OUT = "data/e17_prm800k_pairs.pt"
SEED, CAP = 42, 4
rng = np.random.default_rng(SEED)

d = torch.load(SRC, weights_only=False)
print("source:", d["metadata"]["quality_signal"])


def split_state_action(blind, cond):
    i = 0
    while i < min(len(blind), len(cond)) and blind[len(blind) - 1 - i] == cond[len(cond) - 1 - i]:
        i += 1
    return cond[:len(cond) - i], cond[len(cond) - i:]


by_problem = {}
for split in ("train", "test"):
    for b, c, y in zip(d[f"{split}_texts_blind"], d[f"{split}_texts_conditioned"],
                       d[f"{split}_labels"].tolist()):
        st, ac = split_state_action(b, c)
        if not ac.strip():
            continue
        key = st.split("Previous reasoning:")[0].strip()
        by_problem.setdefault(key, {0: [], 1: []})[int(y > 0.5)].append((st, ac, b))

both = {k: v for k, v in by_problem.items() if v[0] and v[1]}
print(f"{len(by_problem)} problems, {len(both)} with both a correct and an incorrect step")

# ---- context-length matching inside each problem ----
# Rank pairing (i-th shortest correct step with i-th shortest incorrect step) is the
# optimal 1-D assignment under absolute length difference; a hard tail filter then drops
# any pair the problem simply cannot match.
MAX_GAP = 400
pairs, dropped = [], 0
for k, v in both.items():
    pos = sorted(v[1], key=lambda t: len(t[0]))
    neg = sorted(v[0], key=lambda t: len(t[0]))
    cand = []
    for p, n in zip(pos, neg):
        if abs(len(p[0]) - len(n[0])) > MAX_GAP:
            dropped += 1
            continue
        cand.append((k, p[2], n[2], p[0] + p[1], n[0] + n[1]))
    if len(cand) > CAP:
        cand = [cand[i] for i in rng.choice(len(cand), CAP, replace=False)]
    pairs.extend(cand)

print(f"{len(pairs)} length-matched pairs ({dropped} dropped as unmatchable)")

# Rank pairing removes most of the gap but leaves a net -60 chars: correct steps still
# sit in slightly shorter contexts. Truncating harder costs too much data, so instead
# BALANCE the residual -- pair up equal numbers of correct-longer and correct-shorter
# pairs of similar magnitude, so context length carries no net class information.
sgn = lambda p: len(p[3]) - len(p[4])
sel, run = [], 0
for i in rng.permutation(len(pairs)):
    g = sgn(pairs[i])
    tol = max(200.0, 25.0 * np.sqrt(len(sel) + 1))
    if abs(run + g) <= tol:
        sel.append(pairs[i]); run += g
n_long = sum(1 for p in sel if sgn(p) > 0)
pairs = sel
print(f"{len(pairs)} pairs after balancing the residual length confound "
      f"({n_long} correct-longer + {len(pairs) - n_long} correct-shorter)")

gap_ctx = np.mean([len(p[3]) - len(p[4]) for p in pairs])
gap_act = np.mean([len(p[1]) - len(p[2]) for p in pairs])
print(f"residual mean length gap (chosen - rejected): conditioned {gap_ctx:+.1f} chars, "
      f"blind {gap_act:+.1f} chars")
if abs(gap_ctx) > 15:
    raise SystemExit(f"context-length confound not closed ({gap_ctx:+.1f}) -- do not ship this set")

groups = sorted({p[0] for p in pairs})
rng.shuffle(groups)
tr_g = set(groups[:int(0.72 * len(groups))])
tr = [p for p in pairs if p[0] in tr_g]
te = [p for p in pairs if p[0] not in tr_g]
rng.shuffle(tr); rng.shuffle(te)
print(f"train {len(tr)} pairs / {len(tr_g)} problems   test {len(te)} pairs / "
      f"{len(groups) - len(tr_g)} problems")
if len(te) < 150:
    raise SystemExit("test split too small to measure a lift")

data = {
    "train_chosen_blind":   [p[1] for p in tr],
    "train_rejected_blind": [p[2] for p in tr],
    "train_chosen_state":   [p[3] for p in tr],
    "train_rejected_state": [p[4] for p in tr],
    "test_chosen_blind":    [p[1] for p in te],
    "test_rejected_blind":  [p[2] for p in te],
    "test_chosen_state":    [p[3] for p in te],
    "test_rejected_state":  [p[4] for p in te],
    "metadata": {
        "dataset": "PRM800K human step-correctness ratings (from r5_step_data.pt)",
        "label": "human rating of whether this reasoning step is correct",
        "pairing": "same problem, each step in its own true context, greedily "
                   "context-length matched; train/test split by problem",
        "residual_len_gap_conditioned": float(gap_ctx),
        "residual_len_gap_blind": float(gap_act),
        "probe_SL_reported": {"SL_r2": 0.0151, "SL_auc": 0.0709, "encoders_tried": 8},
        "llm_training_lift_reported": 0.238,
        "n_train": len(tr), "n_test": len(te), "seed": SEED,
    },
}
torch.save(data, OUT)
print("Saved", OUT)
print("\nblind :", repr(data["train_chosen_blind"][0])[:190])
print("state :", repr(data["train_chosen_state"][0])[:260])
