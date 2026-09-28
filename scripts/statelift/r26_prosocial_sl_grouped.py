"""
R26: ProsocialDialog SL under random vs GroupKFold-by-dialogue.
Last unverified low-SL Table 2 row (paper: 0.023). Safety label is per-turn
(5-level ordinal), grouped by dialogue_id where available.
"""

import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

SEED = 42
D_PCA = 16
N_MAX = 12000
OUT = "results/r26_prosocial_sl_grouped.json"
np.random.seed(SEED)
enc = SentenceTransformer('all-MiniLM-L6-v2')

ds = load_dataset("allenai/prosocial-dialog", split="train")
print(f"{len(ds)} rows, keys={list(ds[0].keys())}")

label_map = {
    '__casual__': 4, '__possibly_needs_caution__': 3, '__probably_needs_caution__': 2,
    '__needs_caution__': 1, '__needs_intervention__': 0,
}

states, actions, y, groups = [], [], [], []
for ri, ex in enumerate(ds):
    context, response, safety = ex.get('context', ''), ex.get('response', ''), ex.get('safety_label', '')
    if not context or not response or not safety:
        continue
    label = label_map.get(safety)
    if label is None:
        continue
    gid = ex.get('dialogue_id', ri)
    states.append(str(context)[:1024])
    actions.append(str(response)[:512])
    y.append(label)
    groups.append(gid)

y = np.array(y, float); groups = np.array(groups)
print(f"{len(y)} transitions from {len(set(groups.tolist()))} dialogues")
if len(y) > N_MAX:
    idx = np.random.default_rng(SEED).choice(len(y), N_MAX, replace=False)
    states = [states[i] for i in idx]; actions = [actions[i] for i in idx]
    y = y[idx]; groups = groups[idx]

es = enc.encode(states, batch_size=128, show_progress_bar=False)
ea = enc.encode(actions, batch_size=128, show_progress_bar=False)
ps = PCA(n_components=D_PCA, random_state=SEED).fit_transform(es)
pa = PCA(n_components=D_PCA, random_state=SEED).fit_transform(ea)
Xsa = np.hstack([ps, pa])

out = {}
for mode in ['random', 'grouped']:
    if mode == 'random':
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        r2a = cross_val_score(Ridge(alpha=1.0), pa, y, cv=cv, scoring='r2').mean()
        r2sa = cross_val_score(Ridge(alpha=1.0), Xsa, y, cv=cv, scoring='r2').mean()
    else:
        cv = GroupKFold(n_splits=5)
        r2a = cross_val_score(Ridge(alpha=1.0), pa, y, cv=cv, groups=groups, scoring='r2').mean()
        r2sa = cross_val_score(Ridge(alpha=1.0), Xsa, y, cv=cv, groups=groups, scoring='r2').mean()
    out[mode] = {'r2_action': float(r2a), 'r2_state_action': float(r2sa), 'sl': float(r2sa - r2a)}
    print(f"  {mode:8s}: R2(a)={r2a:+.4f} R2(s+a)={r2sa:+.4f} SL={r2sa-r2a:+.4f}")
out['n'] = int(len(y))
json.dump(out, open(OUT, 'w'), indent=2)
print("Saved", OUT)
