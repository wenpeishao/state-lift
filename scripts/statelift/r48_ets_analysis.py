# -*- coding: utf-8 -*-
"""
R48 -- EARLY-TRAINING-SIGNAL (ETS) ANALYSIS.

Stage 2 of the repaired protocol. The probe (Stage 1) cannot read state-dependence that
requires computation, so a low probe reading is not identified. ETS replaces the proxy
with a TRUNCATION of the deployment target: train the real blind and conditioned reward
models for a small budget B of optimizer steps and read

    Delta_B = eval_acc_conditioned(B) - eval_acc_blind(B).

Three questions, in order of how much they matter:
  (a) On the case-B certificates -- arithmetic, constraints_bal, prm800k -- where the
      probe reads ~0, does Delta_B become large and correctly signed at small B?
  (b) On the --shuffle_state nulls, where format, length and action text are preserved
      but the state->quality relation is destroyed, is Delta_B ~ 0? Without this,
      Delta_B could be nothing but a capacity/format advantage of the conditioned arm.
  (c) What is the smallest budget B* at which sign(Delta_B) matches sign(Delta_final)
      and STAYS matched, as a fraction of full training? That fraction is the protocol's
      cost claim, and it must be honest about seed variance -- a single seed can carry
      the wrong sign early (arithmetic seed 43 is negative for its first ~35 steps).
"""
import os, json, glob, collections
import numpy as np

D = "results/multiseed/"
OUT = "results/r48_ets_analysis.json"
CERTIFICATES = {"arithmetic", "constraints_bal", "prm800k"}


def load(pattern):
    runs = collections.defaultdict(list)
    for f in sorted(glob.glob(D + pattern)):
        d = json.load(open(f))
        if not d.get("blind_curve") or not d.get("cond_curve"):
            continue
        # flatten (epoch, opt_step) -> a single monotone step index
        def flat(c):
            per_epoch = max((x["opt_step"] for x in c), default=0)
            return {x["epoch"] * per_epoch + x["opt_step"]: x["eval_acc"] for x in c}
        b, cc = flat(d["blind_curve"]), flat(d["cond_curve"])
        ks = sorted(set(b) & set(cc))
        runs[d["domain"]].append(dict(seed=d.get("seed"), steps=ks,
                                      delta=[cc[k] - b[k] for k in ks],
                                      final=float(d["lift"]),
                                      total=max(ks) if ks else 0))
    return runs


def summarize(runs, label):
    out = {}
    for dom, rs in sorted(runs.items()):
        total = max(r["total"] for r in rs)
        grid = sorted({k for r in rs for k in r["steps"]})
        # mean Delta_B across seeds at each budget
        curve = []
        for k in grid:
            vals = [r["delta"][r["steps"].index(k)] for r in rs if k in r["steps"]]
            if len(vals) < len(rs):
                continue
            curve.append((k, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0))
        finals = [r["final"] for r in rs]
        fmean = float(np.mean(finals))
        sgn = np.sign(fmean)
        # B* = smallest budget from which the signal is both correctly signed AND larger
        # than the spread across seeds, and never loses that property again. Requiring the
        # margin matters: on arithmetic the two seeds are +0.112 and -0.102 at step 10, so
        # a bare sign-of-the-mean test would declare B*=10 on a coin flip.
        bstar = None
        for i, (k, m, s) in enumerate(curve):
            if all(np.sign(x[1]) == sgn and abs(x[1]) > x[2] for x in curve[i:]):
                bstar = k
                break
        # per-seed B*, to expose how much a single seed can mislead
        bstar_seed = []
        for r in rs:
            bs = None
            for i, k in enumerate(r["steps"]):
                if all(np.sign(x) == np.sign(r["final"]) for x in r["delta"][i:]):
                    bs = k
                    break
            bstar_seed.append(bs)
        out[dom] = dict(n_seeds=len(rs), total_steps=int(total), final_lift=fmean,
                        final_sd=float(np.std(finals, ddof=1)) if len(finals) > 1 else None,
                        B_star=bstar, B_star_frac=(bstar / total) if bstar and total else None,
                        B_star_per_seed=bstar_seed,
                        delta_at_10pct=next((m for k, m, _ in curve if k >= 0.10 * total), None),
                        delta_at_25pct=next((m for k, m, _ in curve if k >= 0.25 * total), None),
                        curve=[(int(k), round(m, 4), round(s, 4)) for k, m, s in curve])
    print(f"\n===== {label} =====")
    print(f"{'domain':18s} {'seeds':>5s} {'final':>8s} {'d@10%':>8s} {'d@25%':>8s} {'B*':>5s} {'B*/tot':>7s}")
    for dom, v in out.items():
        f10 = f"{v['delta_at_10pct']:+.4f}" if v["delta_at_10pct"] is not None else "   --  "
        f25 = f"{v['delta_at_25pct']:+.4f}" if v["delta_at_25pct"] is not None else "   --  "
        bs = str(v["B_star"]) if v["B_star"] is not None else "--"
        bf = f"{v['B_star_frac']:.2f}" if v["B_star_frac"] is not None else "--"
        print(f"{dom:18s} {v['n_seeds']:5d} {v['final_lift']:+8.4f} {f10:>8s} {f25:>8s} {bs:>5s} {bf:>7s}")
    return out


real = summarize(load("*_curve_seed*.json"), "ETS on real state")
null = summarize(load("*_shufcurve_seed*.json"), "ETS null (state prefixes permuted)")

print("\n--- (a) case-B certificates: probe ~0, does ETS fire early? ---")
for dom in sorted(CERTIFICATES & set(real)):
    v = real[dom]
    print(f"  {dom:16s} final {v['final_lift']:+.4f}   Delta at 10% of budget "
          f"{v['delta_at_10pct'] if v['delta_at_10pct'] is None else round(v['delta_at_10pct'], 4)}")

print("\n--- (b) specificity: null must sit at ~0 ---")
for dom, v in sorted(null.items()):
    tail = [m for _, m, _ in v["curve"][-5:]]
    print(f"  {dom:16s} final {v['final_lift']:+.4f}   mean Delta over last 5 evals "
          f"{np.mean(tail):+.4f}" if tail else f"  {dom}: no data")

fr = [v["B_star_frac"] for v in real.values() if v["B_star_frac"] is not None]
if fr:
    print(f"\n--- (c) B* across domains: max {max(fr):.2f} of full training "
          f"(median {np.median(fr):.2f}) ---")

json.dump({"real": real, "null": null}, open(OUT, "w"), indent=2)
print("\nSaved", OUT)
