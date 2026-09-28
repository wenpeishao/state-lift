# -*- coding: utf-8 -*-
"""
R40 -- REBUILD THE GENUINE-LABEL REGISTRY WITH AN ADEQUATE PROBE.

The lethal objection assumed Table 2's genuine-label values (all <= 0.057) are the
domain's property. The DoND capacity ladder suggests otherwise: minilm-l6 0.066 ->
roberta-large 0.119, i.e. above the 0.10 threshold once the probe can actually read
the state. If that generalizes, "no genuine-label domain clears 0.10" is a statement
about our encoder, not about the domains.

This re-measures every staged genuine-label domain with a chosen probe, reconstructing
(state, action, label) directly from the pairwise .pt files used for reward-model
training -- so the SL measurement and the training lift come from *identical* data.

state/action recovery: for each item the blind prompt and the conditioned prompt share
the action as a common suffix; the conditioned prompt's remaining prefix is the state.
Grouping: by state string (a dialogue prefix / constraint), so folds never split one
state's items.

Usage:
  python r40_registry_at_capacity.py --pt e4_esconv_data.pt --tag esconv --model <path>
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"
import numpy as np, json, argparse, hashlib
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--pt", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--permute", action="store_true")
ap.add_argument("--within", action="store_true",
                help="DECISION-RELEVANT estimand: score only WITHIN-state comparisons, "
                     "size-weighted. Inside a fixed state the state input is constant, so "
                     "only a state-by-action interaction can help -- which is exactly the "
                     "quantity the PRM-vs-ORM decision depends on. The pooled AUC this "
                     "script reports by default also rewards a pure level effect m(s), "
                     "which has zero decision value (see r49_additive_confound_audit).")
ap.add_argument("--n_max", type=int, default=6000)
ap.add_argument("--data_root", default="data")
ap.add_argument("--out_dir", default="results/registry")
a = ap.parse_args()
np.random.seed(a.seed); torch.manual_seed(a.seed)
os.makedirs(a.out_dir, exist_ok=True)
mtag = os.path.basename(a.model.rstrip("/"))
OUT = os.path.join(a.out_dir, f"r40_{a.tag}_{mtag}{'_within' if a.within else ''}{'_perm' if a.permute else ''}_s{a.seed}.json")

d = torch.load(os.path.join(a.data_root, a.pt), weights_only=False)

def split_state_action(blind, cond):
    """action = longest common suffix of the two prompts; state = cond's prefix."""
    i = 0
    while i < min(len(blind), len(cond)) and blind[len(blind)-1-i] == cond[len(cond)-1-i]:
        i += 1
    action = blind[len(blind)-i:]
    state = cond[:len(cond)-i]
    return state.strip(), action.strip()

S, A, Y, G = [], [], [], []
for split in ("train", "test"):
    for pol, lab in (("chosen", 1.0), ("rejected", 0.0)):
        bl = d.get(f"{split}_{pol}_blind", []); st = d.get(f"{split}_{pol}_state", [])
        for b, c in zip(bl, st):
            s, act = split_state_action(b, c)
            if not act: continue
            S.append(s[:400]); A.append(act[:200]); Y.append(lab)
            G.append(int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16))
Y = np.array(Y); G = np.array(G)
print(f"{a.tag}: {len(Y)} items, {len(set(G.tolist()))} distinct states, pos {Y.mean():.3f}", flush=True)
if len(Y) < 200:
    raise SystemExit("too few items")
if len(Y) > a.n_max:
    k = np.random.default_rng(a.seed).choice(len(Y), a.n_max, replace=False)
    S = [S[i] for i in k]; A = [A[i] for i in k]; Y = Y[k]; G = G[k]

if a.permute:
    # CORRECT null for "does the state add information": shuffle the STATE-action
    # pairing, keeping (action, label) intact. Group-level label permutation is
    # invalid here -- with few distinct states it makes state->label perfectly
    # learnable and inflates the null (observed on the constraints domain: 0.226).
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(S))
    S = [S[i] for i in order]
    G = np.array([int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) for s in S])
    print("PERMUTED (state-action pairing shuffled; action/label preserved)", flush=True)

def fit_predict(pairs_tr, y_tr, pairs_te):
    m = CrossEncoder(a.model, num_labels=1, max_length=384, device="cuda")
    dl = DataLoader([InputExample(texts=list(p), label=float(l)) for p, l in zip(pairs_tr, y_tr)],
                    shuffle=True, batch_size=a.batch)
    m.fit(train_dataloader=dl, epochs=a.epochs, warmup_steps=max(10, int(0.1*len(dl)*a.epochs)),
          optimizer_params={"lr": 2e-5}, show_progress_bar=False)
    sc = m.predict(pairs_te, batch_size=32, show_progress_bar=False)
    del m; torch.cuda.empty_cache()
    return sc

def within_auc(y, score, grp):
    """size-weighted mean AUC computed INSIDE each state; states without both labels
    contribute nothing, because the decision they pose is empty."""
    tot_w, acc = 0.0, 0.0
    for g in set(grp.tolist()):
        m = grp == g
        if len(set(y[m].tolist())) < 2:
            continue
        w = int(m.sum())
        acc += w * roc_auc_score(y[m], score[m]); tot_w += w
    return (acc / tot_w) if tot_w else float("nan"), tot_w


cv = GroupKFold(n_splits=a.folds); per = []
for k, (tr, te) in enumerate(cv.split(S, Y, G)):
    if len(set(Y[te].tolist())) < 2: continue
    try:
        p_sa = fit_predict([(S[i], A[i]) for i in tr], Y[tr], [(S[i], A[i]) for i in te])
        p_ao = fit_predict([("", A[i]) for i in tr], Y[tr], [("", A[i]) for i in te])
        if a.within:
            sa, w = within_auc(Y[te], np.asarray(p_sa), G[te])
            ao, _ = within_auc(Y[te], np.asarray(p_ao), G[te])
            if not np.isfinite(sa) or not np.isfinite(ao):
                print(f"fold {k}: no state carries both labels -- within-state SL UNDEFINED", flush=True)
                continue
        else:
            sa = roc_auc_score(Y[te], p_sa); ao = roc_auc_score(Y[te], p_ao)
    except ValueError:
        continue
    per.append(float(sa - ao))
    tagw = " [within-state]" if a.within else ""
    print(f"fold {k}{tagw}: AUC(s,a)={sa:.4f} AUC(a)={ao:.4f} SL={sa-ao:+.4f}", flush=True)

sl = float(np.mean(per)) if per else float("nan")
sd = float(np.std(per, ddof=1)) if len(per) > 1 else None
json.dump({"domain": a.tag, "pt": a.pt, "model": a.model, "permuted": bool(a.permute),
           "within_state": bool(a.within),
           "SL": sl, "SL_sd": sd, "per_fold": per, "n": int(len(Y)), "seed": a.seed,
           "threshold": 0.10}, open(OUT, "w"), indent=2)
print(f"\n{a.tag} @ {mtag}{' [PERM]' if a.permute else ''}: SL = {sl:+.4f} (sd {sd}) | threshold 0.10", flush=True)
print("Saved", OUT, flush=True)
