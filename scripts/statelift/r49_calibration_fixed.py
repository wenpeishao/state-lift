# -*- coding: utf-8 -*-
"""R49 -- corrected r47 recompute: t-intervals, Llama-only p4g, matched .pt variants,
smoke excluded, DoND under both instruments. Reports negatives count + CP bounds."""
import os, json, glob, collections
import numpy as np
from scipy import stats

RES = "results"

# ---- ground truth lifts, keyed on FILE STEM (not the 'domain' field) ----
lifts = collections.defaultdict(list)
for f in glob.glob(os.path.join(RES, "multiseed", "*.json")):
    b = os.path.basename(f)
    if any(t in b for t in ("_curve", "_shufcurve", "_smoke", "AGGREGATE", "bon_")):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get("lift") is None:
        continue
    stem = b.rsplit("_seed", 1)[0]          # p4g, p4g_mistral, p4g_qwen3, ...
    lifts[stem].append(float(d["lift"]))

# ---- SL-hat, matched to the .pt the lift was trained on ----
sl_by_pt = collections.defaultdict(list)
for f in glob.glob(os.path.join(RES, "registry", "r40_*.json")):
    d = json.load(open(f))
    # This table calibrates the POOLED estimator. The --within runs share the same `pt`
    # field, so keying on `pt` silently merged them in: HH read SLn=12 mixing pooled with
    # within-state values, and DealOrNoDeal's within-state NaN would have propagated. The
    # fold-count variants (_f3/_f10) are likewise a different protocol, not more seeds.
    if d.get("permuted") or d.get("within_state"):
        continue
    if "_f3_" in os.path.basename(f) or "_f10_" in os.path.basename(f):
        continue
    v = float(d["SL"])
    if np.isfinite(v):
        sl_by_pt[d["pt"]].append(v)

PT = {"casino": "e4_casino_data.pt", "esconv": "e4_esconv_data.pt",
      "hhrlhf": "e4_hh_data.pt", "constraints": "e12_constraints_data.pt",
      "craigslist_ldisjoint": "e10b_craigslist_ldisjoint_data.pt",
      "p4g_pdisjoint": "e9e_p4g_pdisjoint_data.pt",
      "multiwoz_precode": "e11d_multiwoz_precode_data.pt"}

rows = []
for dom, pt in PT.items():
    v = sl_by_pt.get(pt, [])
    y = np.array(lifts[dom])
    rows.append(dict(domain=dom, sl=float(np.mean(v)), sl_n=len(v),
                     sl_sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                     n=len(y), lift=float(y.mean()),
                     se=float(y.std(ddof=1)/np.sqrt(len(y))), src="r40 xenc"))
# hard-coded rows, kept but LABELLED
for dom, sl, src in [("arithmetic", 0.0097, "MiniLM dR2->dAUC r42"),
                     ("constraints_bal", -0.0048, "MiniLM r46"),
                     ("dealornodeal", 0.057, "MiniLM dR2 r19b")]:
    y = np.array(lifts[dom])
    rows.append(dict(domain=dom, sl=sl, sl_n=1, sl_sd=None, n=len(y),
                     lift=float(y.mean()), se=float(y.std(ddof=1)/np.sqrt(len(y))), src=src))

for r in rows:
    r["ci_z"] = r["lift"] - 1.96*r["se"]
    r["ci_t"] = r["lift"] - stats.t.ppf(0.975, r["n"]-1)*r["se"]

print(f"{'domain':22s} {'SLn':>3s} {'SL':>8s} {'n':>2s} {'lift':>8s} {'ciZ':>8s} {'ciT':>8s}  src")
for r in sorted(rows, key=lambda x: -x["sl"]):
    print(f"{r['domain']:22s} {r['sl_n']:3d} {r['sl']:+8.4f} {r['n']:2d} {r['lift']:+8.4f} "
          f"{r['ci_z']:+8.4f} {r['ci_t']:+8.4f}  {r['src']}")

def sweep(rows, key, dond_sl=None):
    rr = [dict(r) for r in rows]
    if dond_sl is not None:
        for r in rr:
            if r["domain"] == "dealornodeal":
                r["sl"] = dond_sl
    out = []
    for delta in (0.02, 0.05):
        for tau in (0.0, 0.05, 0.10, 0.15, 0.20):
            pred = np.array([r["sl"] >= tau for r in rr])
            true = np.array([r[key] > delta for r in rr])
            tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
            fn = int((~pred & true).sum()); tn = int((~pred & ~true).sum())
            out.append((tau, delta, tp, fp, fn, tn,
                        [rr[i]["domain"] for i in range(len(rr)) if pred[i] and not true[i]]))
    return out

for label, key, dsl in [("t-interval, DoND=0.057 (MiniLM)", "ci_t", None),
                        ("t-interval, DoND=0.1083 (r36 roberta-large)", "ci_t", 0.10835),
                        ("z-interval, DoND=0.057 (as published)", "ci_z", None)]:
    print("\n=== " + label + " ===")
    print(f"{'tau':>5s} {'delta':>6s} {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s}  prec   false_PRM")
    for tau, delta, tp, fp, fn, tn, fpn in sweep(rows, key, dsl):
        p = tp/(tp+fp) if tp+fp else float('nan')
        print(f"{tau:5.2f} {delta:6.2f} {tp:3d} {fp:3d} {fn:3d} {tn:3d}  {p:5.2f}  {fpn}")

sl = np.array([r["sl"] for r in rows]); lf = np.array([r["lift"] for r in rows])
sp = stats.spearmanr(sl, lf); print(f"\nSpearman {sp.statistic:+.3f} p={sp.pvalue:.3f} n={len(rows)}")
for i in range(len(rows)):
    m = [j for j in range(len(rows)) if j != i]
    s2 = stats.spearmanr(sl[m], lf[m])
    print(f"  drop {rows[i]['domain']:22s} -> rho {s2.statistic:+.3f} p={s2.pvalue:.3f}")
neg = [r["domain"] for r in rows if r["lift"] <= 0]
print(f"\ndomains with non-positive mean lift (true negatives at delta=0): {neg}  -> n={len(neg)}")

json.dump(dict(rows=rows,
               sweeps={lab: [dict(zip(("tau","delta","tp","fp","fn","tn","false_PRM"), c))
                             for c in sweep(rows, key, dsl)]
                       for lab, key, dsl in [("t_dond057","ci_t",None),
                                             ("t_dond1083","ci_t",0.10835),
                                             ("z_dond057_as_published","ci_z",None)]},
               spearman=[float(sp.statistic), float(sp.pvalue)],
               n_true_negatives_at_delta0=len(neg)),
          open("results/r49_calibration_fixed.json","w"), indent=2)
print("Saved results/r49_calibration_fixed.json")
