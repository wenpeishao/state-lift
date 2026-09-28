# -*- coding: utf-8 -*-
"""
ADMISSIBILITY AUDIT -- can this dataset answer the process-vs-outcome reward question at all?

Standalone, model-free, one pass over the data. Run it BEFORE any diagnostic.

Write step quality as  q(s, a) = m(s) + h(a) + g(s, a) + noise.
A state-blind policy picks argmax_a h(a), so its regret is a functional of the interaction
g ALONE -- m(s) shifts every action in a state equally and cancels, and h(a) is what the
blind policy already optimises (r53 verifies this numerically: pooled state-lift reaches
0.94 with regret exactly 0). A dataset is therefore only informative about the decision if
it can speak to g. Two independent things can go wrong, and both are checkable without
training anything:

  CRITERION 1 -- does quality vary WITHIN a state?
      The fraction of states carrying more than one distinct label. If a state holds only
      one label, the comparison the decision turns on does not exist there at all.
      NOT the ICC: tested on a pooled ranking collection built to pass (100% of queries
      carry both labels) the ICC form rejected it at 0.64, because ICC conflates "labels do
      not vary within state" with "labels vary but base rates differ across states". The
      second is harmless for the within-state estimand, where m(s) cancels. ICC is still
      reported, but as a measure of how badly the POOLED estimator will be confounded.

  CRITERION 2 -- is any action ever seen in more than one state?
      g(s,a) is an interaction. Identifying it needs at least some actions to appear under
      different states; if every action occurs exactly once, no amount of data separates
      g from h, because each (s,a) cell has a single observation.

A dataset passing neither cannot support the question. A dataset passing 1 but not 2 can
support a within-state comparison but cannot attribute it to interaction versus action
quality. Only a dataset passing both admits the full decomposition.

Usage:
  python admissibility_audit.py --pt e4_hh_data.pt
  python admissibility_audit.py --all --data_root data
"""
import argparse, glob, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pt", help="pairwise .pt file to audit")
ap.add_argument("--all", action="store_true", help="audit every e*.pt in --data_root")
ap.add_argument("--data_root", default="data")
ap.add_argument("--out", default="results/admissibility_audit.json")
a = ap.parse_args()


def split_state_action(blind, cond):
    i = 0
    while i < min(len(blind), len(cond)) and blind[len(blind) - 1 - i] == cond[len(cond) - 1 - i]:
        i += 1
    return cond[:len(cond) - i].strip(), cond[len(cond) - i:].strip()


def audit(path):
    d = torch.load(path, weights_only=False)
    if "train_chosen_state" not in d:
        return None
    S, A, Y = [], [], []
    for split in ("train", "test"):
        for pol, y in (("chosen", 1.0), ("rejected", 0.0)):
            for b, c in zip(d.get(f"{split}_{pol}_blind", []), d.get(f"{split}_{pol}_state", [])):
                st, ac = split_state_action(b, c)
                if ac:
                    S.append(st); A.append(ac); Y.append(y)
    if not Y:
        return None
    Y = np.array(Y)

    # criterion 1 -- ICC of the labels by state
    by_s = {}
    for s, y in zip(S, Y):
        by_s.setdefault(s, []).append(y)
    sizes = np.array([len(v) for v in by_s.values()])
    means = np.array([np.mean(v) for v in by_s.values()])
    tot_var = Y.var()
    between = float(np.sum(sizes * (means - Y.mean()) ** 2) / len(Y)) if tot_var > 0 else 0.0
    icc = between / tot_var if tot_var > 0 else float("nan")
    pct_contrastive = float(np.mean([len(set(v)) > 1 for v in by_s.values()]))

    # criterion 2 -- action reuse across distinct states
    by_a = {}
    for s, act in zip(S, A):
        by_a.setdefault(act, set()).add(s)
    reuse = np.array([len(v) for v in by_a.values()])
    pct_actions_reused = float(np.mean(reuse > 1))
    pct_items_reusable = float(sum(len(v) for k, v in by_a.items() if len(v) > 1) / len(Y))

    # Criterion 1 is the fraction of states carrying more than one label, NOT the ICC.
    # Tested on a case built to pass (a pooled ranking collection: 100% of queries carry both
    # labels, ICC 0.64) the ICC form rejected admissible data, because it conflates "labels do
    # not vary within state" with "labels vary but base rates differ across states" -- and the
    # second is harmless for the within-state estimand, where m(s) cancels. ICC is still
    # reported, as a measure of how badly the POOLED estimator will be confounded.
    c1 = pct_contrastive > 0.50
    c2 = pct_actions_reused > 0.05
    verdict = ("ADMISSIBLE -- full decomposition available" if c1 and c2 else
               "PARTIAL -- within-state comparison possible, interaction not attributable"
               if c1 else
               "INADMISSIBLE -- labels are a property of the state")
    return dict(file=os.path.basename(path), n_items=int(len(Y)), n_states=len(by_s),
                n_actions=len(by_a), ICC_label=float(icc),
                pct_states_contrastive=pct_contrastive,
                mean_states_per_action=float(reuse.mean()),
                pct_actions_reused=pct_actions_reused,
                pct_items_in_reused_actions=pct_items_reusable,
                criterion1_within_state_variation=bool(c1),
                criterion2_interaction_identifiable=bool(c2),
                verdict=verdict)


paths = (sorted(glob.glob(os.path.join(a.data_root, "e*.pt"))) if a.all
         else [os.path.join(a.data_root, a.pt)])
rows = [r for r in (audit(p) for p in paths) if r]
rows.sort(key=lambda r: r["ICC_label"])

print(f"{'dataset':34s} {'ICC':>6s} {'%contr':>7s} {'act/reuse':>9s} {'verdict'}")
for r in rows:
    print(f"{r['file']:34s} {r['ICC_label']:6.3f} {r['pct_states_contrastive']:7.1%} "
          f"{r['pct_actions_reused']:9.1%} {r['verdict'].split(' -- ')[0]}")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rows, open(a.out, "w"), indent=2)
print(f"\nSaved {a.out}")
print("\nCriterion 1 (>50% of states carry both labels): the within-state comparison exists.")
print("Criterion 2 (>5% of actions seen in 2+ states): the interaction is identifiable at all.")
print("Failing 1 means the question cannot be asked here; failing only 2 means it can be asked")
print("but not attributed to interaction versus action quality. ICC is reported separately as")
print("how badly a POOLED estimator will be confounded -- it is not an admissibility test.")
