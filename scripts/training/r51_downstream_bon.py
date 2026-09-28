# -*- coding: utf-8 -*-
"""
R51 -- DOWNSTREAM BEST-OF-N ON A LOW-STATE-LIFT DOMAIN.

Reviewer MmQK (our only positive review) asked for exactly one thing we never delivered:

  "The downstream validation (beam search on GSM8K, Appendix F) confirms state-lift predicts
   task performance for math. Can you provide similar downstream evaluations for at least one
   low-SL domain (e.g. HH-RLHF or code) to confirm that low state-lift correctly predicts no
   downstream PRM benefit?"

ESConv is the low-state-lift domain we have trained adapters for (SL between -0.044 and
+0.028 depending on probe; training lift +0.0084 +/- 0.0086 over 8 seeds). DealOrNoDeal is
the contrast, the highest genuine-label SL in the corrected registry.

The metric is best-of-N selection accuracy, computed WITHIN a state: at a held-out dialogue
state, score the ground-truth response against N-1 distractors drawn from the SAME state,
and ask whether the reward model ranks the true one first. This is judge-free, it is the
decision a reward model is actually deployed to make, and -- because the state is held fixed
across the candidates -- it cannot be won by a level effect m(s). Chance is 1/N.

Usage:
  python r51_downstream_bon.py --domain esconv --seed 42
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"
import argparse, json, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_list", default="2,4,8,16")
ap.add_argument("--n_trials", type=int, default=600)
ap.add_argument("--model_path", default=os.environ.get("LLAMA_PATH", "meta-llama/Llama-3.1-8B-Instruct"))
ap.add_argument("--root", default=".", help="repo root (contains data/ and results/)")
a = ap.parse_args()

FILES = {"esconv": "e4_esconv_data.pt", "dealornodeal": "e5_dealornodeal_data.pt",
         "arithmetic": "e14_arithmetic_data.pt",
         "constraints_bal": "e16_constraints_balanced.pt",
         "hhrlhf": "e4_hh_data.pt"}
N_LIST = [int(x) for x in a.n_list.split(",")]
OUT = f"{a.root}/results/r51_bon_{a.domain}_seed{a.seed}.json"
d = torch.load(f"{a.root}/data/{FILES[a.domain]}", weights_only=False)


def split_state_action(blind, cond):
    i = 0
    while i < min(len(blind), len(cond)) and blind[len(blind) - 1 - i] == cond[len(cond) - 1 - i]:
        i += 1
    return cond[:len(cond) - i], cond[len(cond) - i:]


# pool held-out items by state; keep the blind and conditioned rendering of each
by_state = {}
for pol, y in (("chosen", 1), ("rejected", 0)):
    for b, c in zip(d[f"test_{pol}_blind"], d[f"test_{pol}_state"]):
        st, act = split_state_action(b, c)
        if not act.strip():
            continue
        by_state.setdefault(st, []).append(dict(y=y, blind=b, cond=c))

usable = {k: v for k, v in by_state.items()
          if any(x["y"] == 1 for x in v) and sum(x["y"] == 0 for x in v) >= max(N_LIST) - 1}
print(f"{a.domain}: {len(by_state)} held-out states, {len(usable)} support best-of-{max(N_LIST)}",
      flush=True)
if not usable:
    raise SystemExit("no state has enough within-state distractors -- best-of-N is UNDEFINED here, "
                     "which is itself the finding (see r49_additive_confound_audit)")

rng = np.random.default_rng(a.seed)
keys = sorted(usable)
trials = []
for _ in range(a.n_trials):
    k = keys[rng.integers(0, len(keys))]
    pool = usable[k]
    pos = [x for x in pool if x["y"] == 1]
    neg = [x for x in pool if x["y"] == 0]
    t = pos[rng.integers(0, len(pos))]
    dis = [neg[i] for i in rng.choice(len(neg), max(N_LIST) - 1, replace=False)]
    trials.append((t, dis))

tok = AutoTokenizer.from_pretrained(a.model_path)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def score_all(adapter, field):
    m = AutoModelForSequenceClassification.from_pretrained(
        a.model_path, num_labels=1, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    m.config.pad_token_id = tok.pad_token_id
    m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    out = []
    with torch.no_grad():
        for t, dis in trials:
            texts = [t[field]] + [x[field] for x in dis]
            enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            s = m(input_ids=enc["input_ids"].cuda(),
                  attention_mask=enc["attention_mask"].cuda()).logits.squeeze(-1)
            out.append(s.float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return out


ad = f"{a.root}/results/multiseed/adapters"
res = {}
for arm, adapter, field in (("blind", f"{ad}/{a.domain}_blind_seed{a.seed}", "blind"),
                            ("conditioned", f"{ad}/{a.domain}_state_conditioned_seed{a.seed}", "cond")):
    if not os.path.isdir(adapter):
        raise SystemExit(f"missing adapter {adapter}")
    print(f"scoring {arm} ...", flush=True)
    sc = score_all(adapter, field)
    res[arm] = {}
    for N in N_LIST:
        # candidate 0 is always the true response; truncate to the first N candidates
        hits = [float(np.argmax(s[:N]) == 0) for s in sc]
        res[arm][N] = dict(top1=float(np.mean(hits)),
                           se=float(np.std(hits, ddof=1) / np.sqrt(len(hits))),
                           chance=1.0 / N)
        print(f"  best-of-{N}: top1={np.mean(hits):.4f} (chance {1/N:.3f})", flush=True)

print(f"\n{'N':>4s} {'blind':>18s} {'conditioned':>18s} {'lift':>10s}")
lifts = {}
for N in N_LIST:
    b, c = res["blind"][N], res["conditioned"][N]
    lifts[N] = c["top1"] - b["top1"]
    print(f"{N:4d} {b['top1']:.4f}+/-{b['se']:.4f} {c['top1']:.4f}+/-{c['se']:.4f} {lifts[N]:+10.4f}")

json.dump(dict(domain=a.domain, seed=a.seed, n_trials=len(trials),
               n_states=len(usable), results=res, lifts=lifts,
               metric="within-state best-of-N top-1 selection accuracy, judge-free",
               note=("state is held fixed across candidates, so a pure level effect m(s) "
                     "cannot win; this is the decision a reward model is deployed to make")),
          open(OUT, "w"), indent=2)
print("Saved", OUT, flush=True)
