# -*- coding: utf-8 -*-
"""
R36 -- PROBE-CAPACITY LADDER (the decisive salvage experiment).

Every "probe can't see it" result in this project used a ~22M-param encoder, while the
signal was recovered by an 8B reward model. So "probe-invisible" may really be
"under-capacity measurement". This runs the SAME cross-attention probe protocol at
four capacities and looks for a monotone SL-vs-capacity curve.

  MiniLM-L6   (~22M)   -> reference, r34b gave +0.019 on P4G
  MiniLM-L12  (~33M)
  roberta-base (~125M)
  roberta-large (~355M)
                          vs full Llama-8B RM training lift = +0.088

Reads:
  monotone rise toward 0.088  => state-lift is capacity-limited; measure at 2-3
      capacities and extrapolate. This is a WORKING diagnostic and answers 52zF Q5
      (representation-adequacy check) with a positive result.
  flat near 0.02              => genuinely training-only; clean architecture-level
      negative, and the phenomenon paper is the whole story.

Domain: PersuasionForGood (bi-encoder SL ~= 0, full training +0.088) -- the case where
the gap is largest. Grouped by dialogue; genuine donation-outcome labels.
Usage: python r36_capacity_ladder.py --model roberta-base
"""
import os
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"
import numpy as np, json, csv, argparse
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--domain", default="p4g", choices=["p4g","dond"])
ap.add_argument("--tag", default=None)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--folds", type=int, default=3)
ap.add_argument("--permute", action="store_true", help="shuffle dialogue labels (null control)")
ap.add_argument("--seed", type=int, default=42, help="controls the permutation draw / subsample")
ap.add_argument("--data_root", default="data")
ap.add_argument("--out_dir", default="results/capacity")
args = ap.parse_args()

SEED, N_MAX = args.seed, 6000
np.random.seed(SEED); torch.manual_seed(SEED)
tag = args.tag or args.model.split("/")[-1]
os.makedirs(args.out_dir, exist_ok=True)
OUT = os.path.join(args.out_dir, f"r36_{args.domain}_{tag}{'_perm' if args.permute else ''}_s{args.seed}.json")

# ---------- genuine-outcome labels, dialogue-grouped ----------
S, A, Y, G = [], [], [], []
if args.domain == "p4g":
    donation = {}
    with open(os.path.join(args.data_root, "persuasion_info.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: amt = float(r.get("B6", "") or "nan")
            except ValueError: continue
            if amt != amt: continue
            if r.get("B4", "") == "1" or r.get("B2", "") not in donation:
                donation[r.get("B2", "")] = amt
    with open(os.path.join(args.data_root, "persuasion_full.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dlg = defaultdict(list)
    for r in rows: dlg[r["B2"]].append(r)
    med = np.median([donation[d] for d in dlg if d in donation])
    for gi, (did, turns) in enumerate(sorted(dlg.items())):
        if did not in donation: continue
        label = 1 if donation[did] > med else 0
        texts = [t["Unit"] for t in turns if t.get("Unit", "").strip()]
        roles = [t.get("B4", "") for t in turns if t.get("Unit", "").strip()]
        if len(texts) < 4: continue
        for i in range(2, len(texts)):
            if roles[i] != "0": continue
            S.append(" ".join(texts[:i])[:400]); A.append(texts[i][:200])
            Y.append(label); G.append(gi)
else:   # dond -- highest genuine-label SL domain (bi-encoder 0.057)
    import re as _re
    recs = []
    with open(os.path.join(args.data_root, "dealornodeal_train.txt")) as f:
        for di, line in enumerate(f):
            line = line.strip()
            if not line: continue
            inp = _re.search(r"<input>\s*(.*?)\s*</input>", line)
            dm  = _re.search(r"<dialogue>\s*(.*?)\s*</dialogue>", line)
            om  = _re.search(r"<output>\s*(.*?)\s*</output>", line)
            if not inp or not dm or not om: continue
            tk = inp.group(1).split()
            if len(tk) < 6: continue
            try:
                counts = [int(tk[i]) for i in range(0, 6, 2)]; values = [int(tk[i]) for i in range(1, 6, 2)]
            except: continue
            ch = [int(t.split("=")[1]) for t in om.group(1).split() if "=" in t and t.split("=")[1].isdigit()]
            if len(ch) >= 3:
                q = sum(c*v for c, v in zip(ch[:3], values)) / max(sum(c*v for c, v in zip(counts, values)), 1)
            elif "<no_agreement>" in om.group(1): q = 0.0
            else: continue
            tr = [t.strip() for t in dm.group(1).split("<eos>") if t.strip() and "<selection>" not in t]
            tr = [_re.sub(r"^(YOU|THEM):", "", t).strip() for t in tr]
            if len(tr) < 3: continue
            recs.append((di, q, tr))
    med = np.median([r[1] for r in recs])
    for di, q, tr in recs:
        label = 1 if q > med else 0
        for i in range(1, len(tr)):
            S.append(" ".join(tr[:i])[:400]); A.append(tr[i][:200])
            Y.append(label); G.append(di)
Y = np.array(Y); G = np.array(G)
print(f"{args.domain}: {len(Y)} turns, {len(set(G.tolist()))} dialogues, pos {Y.mean():.3f}", flush=True)
if len(Y) > N_MAX:
    idx = np.random.default_rng(SEED).choice(len(Y), N_MAX, replace=False)
    S = [S[i] for i in idx]; A = [A[i] for i in idx]; Y = Y[idx]; G = G[idx]

if args.permute:
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(G.tolist())))
    lab = {g: Y[G == g][0] for g in uniq}
    pm = dict(zip(uniq, rng.permutation(list(lab.values()))))
    Y = np.array([pm[g] for g in G])
    print("PERMUTED labels (null control)", flush=True)

def fit_predict(pairs_tr, y_tr, pairs_te):
    m = CrossEncoder(args.model, num_labels=1, max_length=384, device="cuda")
    dl = DataLoader([InputExample(texts=list(p), label=float(l)) for p, l in zip(pairs_tr, y_tr)],
                    shuffle=True, batch_size=args.batch)
    m.fit(train_dataloader=dl, epochs=args.epochs,
          warmup_steps=max(10, int(0.1 * len(dl) * args.epochs)),
          optimizer_params={"lr": 2e-5}, show_progress_bar=False)
    sc = m.predict(pairs_te, batch_size=32, show_progress_bar=False)
    del m; torch.cuda.empty_cache()
    return sc

cv = GroupKFold(n_splits=args.folds)
sa_l, ao_l = [], []
for k, (tr, te) in enumerate(cv.split(S, Y, G)):
    try:
        a_sa = roc_auc_score(Y[te], fit_predict([(S[i], A[i]) for i in tr], Y[tr], [(S[i], A[i]) for i in te]))
        a_ao = roc_auc_score(Y[te], fit_predict([("", A[i]) for i in tr], Y[tr], [("", A[i]) for i in te]))
    except ValueError:
        continue
    sa_l.append(float(a_sa)); ao_l.append(float(a_ao))
    print(f"fold {k}: AUC(s,a)={a_sa:.4f} AUC(a)={a_ao:.4f} SL={a_sa - a_ao:+.4f}", flush=True)

per_fold = [x - y for x, y in zip(sa_l, ao_l)]
sl = float(np.mean(per_fold)) if per_fold else float("nan")
sd = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else None
json.dump({"model": args.model, "tag": tag, "permuted": bool(args.permute),
           "SL": sl, "SL_sd": sd, "per_fold": per_fold,
           "auc_state_action": sa_l, "auc_action": ao_l,
           "epochs": args.epochs, "folds": args.folds, "n": int(len(Y)),
           "reference": {"bi_encoder_SL": -0.001, "minilm_L6_crossenc_SL": 0.0188,
                         "full_RM_training_lift": 0.088}},
          open(OUT, "w"), indent=2)
print(f"\n{tag}{' [PERM]' if args.permute else ''}  SL = {sl:+.4f} (sd {sd})  "
      f"| bi-enc ~= 0 | Llama-8B training = +0.088", flush=True)
print("Saved", OUT, flush=True)
