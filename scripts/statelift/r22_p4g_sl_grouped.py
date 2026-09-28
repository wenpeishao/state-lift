"""
R22: PersuasionForGood genuine-label SL (donation outcome), random vs GroupKFold.
Fifth genuine-label domain for the SL-vs-lift analysis (currently n=4 domains).
Recipe = r17d (continuous quality, prefix states) with dialogue grouping added.
"""

import numpy as np
import csv, json
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sentence_transformers import SentenceTransformer

SEED = 42
D_PCA = 16
OUT = "results/r22_p4g_sl_grouped.json"
np.random.seed(SEED)

enc = SentenceTransformer('all-MiniLM-L6-v2')

with open("data/persuasion_full.csv", 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print(f"{len(rows)} rows; cols: {list(rows[0].keys())[:10]}")

did_col, text_col = 'B2', 'Unit'

# Donation outcomes from companion info file: one row per (dialogue B2, role B4);
# B4==1 is the persuadee; B6 = amount donated. Zero donations INCLUDED (genuine outcome).
donation_map = {}
with open("data/persuasion_info.csv", 'r', encoding='utf-8') as f:
    for irow in csv.DictReader(f):
        try:
            amt = float(irow.get('B6', '') or 'nan')
        except ValueError:
            continue
        if amt != amt:
            continue
        did = irow.get('B2', '')
        role = irow.get('B4', '')
        # prefer persuadee row (role==1); otherwise keep max seen
        if role == '1' or did not in donation_map:
            donation_map[did] = amt
print(f"donation outcomes for {len(donation_map)} dialogues; "
      f"{sum(1 for v in donation_map.values() if v > 0)} donated > 0")

dialogues = defaultdict(list)
for row in rows:
    dialogues[row.get(did_col, '')].append(row)

states, actions, qualities, groups = [], [], [], []
for gi, (did, turns) in enumerate(dialogues.items()):
    if did not in donation_map:
        continue
    donation = donation_map[did]
    quality = min(donation / 2.0, 1.0)
    for i, turn in enumerate(turns):
        text = turn.get(text_col, '')
        if not text or not text.strip():
            continue
        prev = " ".join(t.get(text_col, '')[:80] for t in turns[:i])[:300] if i > 0 else "[START]"
        states.append("Persuasion: " + prev)
        actions.append(text[:200])
        qualities.append(quality)
        groups.append(gi)

print(f"{len(states)} tuples from {len(set(groups))} dialogues")
if len(states) > 15000:
    idx = np.random.choice(len(states), 15000, replace=False)
    states = [states[i] for i in idx]
    actions = [actions[i] for i in idx]
    qualities = [qualities[i] for i in idx]
    groups = [groups[i] for i in idx]

es = enc.encode(states, batch_size=128, show_progress_bar=False)
ea = enc.encode(actions, batch_size=128, show_progress_bar=False)
y = np.array(qualities, dtype=float)
g = np.array(groups)
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
with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved {OUT}")
