# -*- coding: utf-8 -*-
"""
R49 -- THE ADDITIVE-CONFOUND AUDIT: separating the predictive estimand from the
decision-relevant one.

An adversarial audit of the repair (2026-07-28) found the defect that matters most, and
it is not the probe. State-lift as defined and estimated is a PREDICTIVE quantity:

    E1:  E[q | s, a]  !=  E[q | a]        "state helps predict quality"

but the decision the paper claims to inform -- build a process reward model, or a
state-blind one -- and Theorem 1's regret bound both need a DIFFERENT quantity:

    E2:  argmax_a E[q | s, a]  depends on s     "state changes which action is best"

These come apart exactly when quality is additive, E[q|s,a] = m(s) + h(a). Then state-lift
can be arbitrarily large while the best action is the same in every state, so a state-blind
policy has regret EXACTLY ZERO and a PRM buys nothing. E1 is necessary for E2 but nowhere
near sufficient, and every number in the paper estimates E1.

This script measures how much of each domain's label variance is a pure level effect m(s).
The estimator is a variance decomposition of the pairwise labels by state:

    ICC_label = between-state variance / total variance

  ICC ~ 1  -> the label is a property of the STATE, not of the action taken in it. Every
             point of measured lift can be a level shift. The PRM claim is unsupported here
             no matter how large the lift, and within-state AUC is often not even defined.
  ICC ~ 0  -> all contrast lives inside states. Lift here is decision-relevant.

This is cheap, needs no model, and belongs in Stage 0 of the protocol -- before the probe,
because it decides whether the probe's answer could mean anything.
"""
import glob, json, os
import numpy as np
import torch

DATA = "data/"
OUT = "results/r49_additive_confound_audit.json"


def split_state_action(blind, cond):
    i = 0
    while i < min(len(blind), len(cond)) and blind[len(blind) - 1 - i] == cond[len(cond) - 1 - i]:
        i += 1
    return cond[:len(cond) - i], cond[len(cond) - i:]


rows = []
for f in sorted(glob.glob(DATA + "e*.pt")):
    try:
        d = torch.load(f, weights_only=False)
    except Exception:
        continue
    if "train_chosen_state" not in d:
        continue

    # pool every observed (state, action, label); chosen = 1, rejected = 0
    by_state = {}
    for split in ("train", "test"):
        for pol, y in (("chosen", 1.0), ("rejected", 0.0)):
            bl = d.get(f"{split}_{pol}_blind", [])
            st = d.get(f"{split}_{pol}_state", [])
            for b, c in zip(bl, st):
                p, a = split_state_action(b, c)
                if not a.strip():
                    continue
                by_state.setdefault(p, []).append(y)

    if not by_state:
        continue
    ys = np.array([y for v in by_state.values() for y in v])
    n_tot = len(ys)
    grand = ys.mean()
    tot_var = ys.var()

    sizes = np.array([len(v) for v in by_state.values()])
    means = np.array([np.mean(v) for v in by_state.values()])
    between = float(np.sum(sizes * (means - grand) ** 2) / n_tot)
    icc = between / tot_var if tot_var > 0 else float("nan")

    multi = sizes > 1
    both = np.array([len(set(v)) > 1 for v in by_state.values()])

    rows.append(dict(
        domain=os.path.basename(f), n_items=int(n_tot), n_states=int(len(by_state)),
        items_per_state=float(sizes.mean()),
        pct_states_multi_item=float(multi.mean()),
        pct_states_both_labels=float(both.mean()),
        pct_items_in_contrastive_state=float(sizes[both].sum() / n_tot) if both.any() else 0.0,
        ICC_label=float(icc),
        within_state_var_fraction=float(1 - icc) if tot_var > 0 else float("nan"),
    ))

rows.sort(key=lambda r: r["ICC_label"])
print(f"{'domain':34s} {'states':>7s} {'/state':>6s} {'%both':>6s} {'%items':>7s} {'ICC':>6s} {'verdict'}")
for r in rows:
    v = ("DECISION-RELEVANT" if r["ICC_label"] < 0.35 else
         "MIXED" if r["ICC_label"] < 0.9 else "PURE LEVEL EFFECT")
    print(f"{r['domain']:34s} {r['n_states']:7d} {r['items_per_state']:6.2f} "
          f"{r['pct_states_both_labels']:6.1%} {r['pct_items_in_contrastive_state']:7.1%} "
          f"{r['ICC_label']:6.3f} {v}")

print("\nICC ~ 1 means the label is a property of the state, so the whole measured lift can be")
print("a level shift m(s) with zero decision value. ICC ~ 0 means all contrast is within-state.")

json.dump(rows, open(OUT, "w"), indent=2)
print("\nSaved", OUT)
