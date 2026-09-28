"""
R24: USS-Satisfaction genuine-label SL under random vs GroupKFold-by-session.
Extends the corrected Table 2 genuine-label set (6th domain). Turn-level
mean ratings (labels vary within session, unlike trajectory-level P4G/DoND).
Recipe = r14b load_uss + the standard MiniLM+PCA16+Ridge pipeline.
"""

import numpy as np
import json
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import pandas as pd

SEED = 42
D_PCA = 16
OUT = "results/r24_uss_sl_grouped.json"
np.random.seed(SEED)
enc = SentenceTransformer('all-MiniLM-L6-v2')

# HF datasets builder fails on schema cast; read the parquet directly.
p = hf_hub_download('akomma/uss-ratings-dataset', 'uss-ratings-dataset.parquet', repo_type='dataset')
df = pd.read_parquet(p)
print(f"{len(df)} rows")
sessions = defaultdict(list)
for ex in df.to_dict('records'):
    # session_idx repeats across source splits (CCPE, JDDC, ...) -- key on both
    sid = f"{ex.get('split','')}_{ex.get('session_idx',0)}_{ex.get('tree_idx',0)}"
    sessions[sid].append(ex)

states, actions, ratings, groups = [], [], [], []
for gi, (sid, turns) in enumerate(sorted(sessions.items(), key=lambda kv: str(kv[0]))):
    turns.sort(key=lambda x: x.get('turn_idx', 0))
    for i, turn in enumerate(turns):
        system = turn.get('system', '')
        user = turn.get('user', '')
        rating = turn.get('mean_turn_rating', turn.get('mode_turn_rating', None))
        if not system or rating is None:
            continue
        if i == 0:
            state = user if user else "Start of conversation"
        else:
            prev = []
            for j in range(i):
                u, s = turns[j].get('user', ''), turns[j].get('system', '')
                if u: prev.append(f"User: {u}")
                if s: prev.append(f"System: {s}")
            if user: prev.append(f"User: {user}")
            state = '\n'.join(prev)
        states.append(state[:1024]); actions.append(system[:512])
        ratings.append(float(rating)); groups.append(gi)

y = np.array(ratings, float); g = np.array(groups)
print(f"{len(y)} transitions from {len(set(groups))} sessions")

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
        r2a = cross_val_score(Ridge(alpha=1.0), pa, y, cv=cv, groups=g, scoring='r2').mean()
        r2sa = cross_val_score(Ridge(alpha=1.0), Xsa, y, cv=cv, groups=g, scoring='r2').mean()
    out[mode] = {'r2_action': float(r2a), 'r2_state_action': float(r2sa), 'sl': float(r2sa - r2a)}
    print(f"  {mode:8s}: R2(a)={r2a:+.4f} R2(s+a)={r2sa:+.4f} SL={r2sa-r2a:+.4f}")
out['n'] = len(y)
json.dump(out, open(OUT, 'w'), indent=2)
print("Saved", OUT)
