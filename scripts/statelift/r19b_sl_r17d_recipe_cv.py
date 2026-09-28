"""
R19b: Replicate the EXACT r17d SL recipe (continuous trajectory-level quality,
dialogue-prefix states, MiniLM+PCA16+Ridge 5-fold) for DealOrNoDeal, and the
analogous continuous-points recipe for CaSiNo, under:
  (a) random KFold (original r17d protocol -- suspected leakage)
  (b) GroupKFold by dialogue (leakage-proof)
Decides whether Table 2's DoND SL=0.18 survives clean CV.
"""

import numpy as np
import json, os
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sentence_transformers import SentenceTransformer

SEED = 42
D_PCA = 16
OUT = "results/r19b_sl_r17d_recipe_cv.json"
np.random.seed(SEED)

enc = SentenceTransformer('all-MiniLM-L6-v2')

def compute_sl_cv(states, actions, qualities, groups, name):
    es = enc.encode(states, batch_size=128, show_progress_bar=False)
    ea = enc.encode(actions, batch_size=128, show_progress_bar=False)
    y = np.array(qualities, dtype=float)
    groups = np.array(groups)
    pca_s = PCA(n_components=D_PCA, random_state=SEED).fit_transform(es)
    pca_a = PCA(n_components=D_PCA, random_state=SEED).fit_transform(ea)
    Xa = pca_a
    Xsa = np.hstack([pca_s, pca_a])
    out = {}
    for mode in ['random', 'grouped']:
        if mode == 'random':
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            r2a = cross_val_score(Ridge(alpha=1.0), Xa, y, cv=cv, scoring='r2').mean()
            r2sa = cross_val_score(Ridge(alpha=1.0), Xsa, y, cv=cv, scoring='r2').mean()
        else:
            cv = GroupKFold(n_splits=5)
            r2a = cross_val_score(Ridge(alpha=1.0), Xa, y, cv=cv, groups=groups, scoring='r2').mean()
            r2sa = cross_val_score(Ridge(alpha=1.0), Xsa, y, cv=cv, groups=groups, scoring='r2').mean()
        sl = r2sa - r2a
        out[mode] = {'r2_action': float(r2a), 'r2_state_action': float(r2sa), 'sl': float(sl)}
        print(f"  {name} {mode:8s}: R2(a)={r2a:+.4f} R2(s+a)={r2sa:+.4f} SL={sl:+.4f}")
    out['n'] = len(y)
    return out

results = {}

# ============================================================
# 1. DealOrNoDeal -- exact r17d parsing (continuous quality)
# ============================================================
print("[1] DealOrNoDeal (r17d recipe)...")
import re
states, actions, qualities, groups = [], [], [], []
with open("data/dealornodeal_train.txt") as f:
    for di, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        inp = re.search(r"<input>\s*(.*?)\s*</input>", line)
        dlg = re.search(r"<dialogue>\s*(.*?)\s*</dialogue>", line)
        out = re.search(r"<output>\s*(.*?)\s*</output>", line)
        if not inp or not dlg or not out:
            continue
        toks = inp.group(1).strip().split()
        if len(toks) < 6:
            continue
        try:
            counts = [int(toks[i]) for i in range(0, 6, 2)]
            values = [int(toks[i]) for i in range(1, 6, 2)]
        except:
            continue
        output = out.group(1)
        chosen = []
        for t in output.strip().split():
            if '=' in t:
                try: chosen.append(int(t.split('=')[1]))
                except: pass
        if len(chosen) >= 3:
            av = sum(c*v for c, v in zip(chosen[:3], values[:3]))
            mv = sum(c*v for c, v in zip(counts[:3], values[:3]))
            quality = av / max(mv, 1)
        elif '<no_agreement>' in output or 'disagree' in output.lower():
            quality = 0.0
        else:
            continue
        turns = [t.strip() for t in dlg.group(1).split("<eos>") if t.strip() and "<selection>" not in t]
        if len(turns) < 2:
            continue
        for i, turn in enumerate(turns):
            tc = turn.replace("YOU:", "").replace("THEM:", "").strip()
            if not tc:
                continue
            prev = " ".join(t.replace("YOU:", "").replace("THEM:", "").strip()[:80]
                            for t in turns[:i])[:300] if i > 0 else "[START]"
            states.append("Negotiation: " + prev)
            actions.append(tc[:200])
            qualities.append(quality)
            groups.append(di)

print(f"  {len(states)} tuples")
if len(states) > 15000:
    idx = np.random.choice(len(states), 15000, replace=False)
    states = [states[i] for i in idx]
    actions = [actions[i] for i in idx]
    qualities = [qualities[i] for i in idx]
    groups = [groups[i] for i in idx]
results['dealornodeal_r17d'] = compute_sl_cv(states, actions, qualities, groups, "DoND")

# ============================================================
# 2. CaSiNo -- same recipe with continuous points quality
# ============================================================
print("[2] CaSiNo (continuous points, r17d-style)...")
from datasets import load_dataset
ds = load_dataset('casino', split='train')
states, actions, qualities, groups = [], [], [], []
for di, ex in enumerate(ds):
    pts_map = {}
    for aid in ['mturk_agent_1', 'mturk_agent_2']:
        pts = ex['participant_info'].get(aid, {}).get('outcomes', {}).get('points_scored', None)
        if pts is not None:
            pts_map[aid] = int(pts)
    if len(pts_map) < 2:
        continue
    texts, agents = [], []
    for t in ex['chat_logs']:
        if t['text'] in ('Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away') or not t['text'].strip():
            continue
        texts.append(t['text']); agents.append(t['id'])
    if len(texts) < 4:
        continue
    for i in range(1, len(texts)):
        pts = pts_map.get(agents[i], None)
        if pts is None:
            continue
        prev = " ".join(t[:80] for t in texts[:i])[:300]
        states.append("Negotiation: " + prev)
        actions.append(texts[i][:200])
        qualities.append(float(pts))
        groups.append(di)

print(f"  {len(states)} tuples")
results['casino_points_r17d_style'] = compute_sl_cv(states, actions, qualities, groups, "CaSiNo")

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved {OUT}")
