"""
R25: AirDialogue genuine-label SL under random vs GroupKFold-by-dialogue.
7th genuine-label domain; trajectory-level label (correct_sample) -> leakage-prone
class, so grouped CV is the honest number. Paper Table 2 value: 0.004.
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
N_DIALOGUES = 2500
N_MAX = 12000
OUT = "results/r25_airdialogue_sl_grouped.json"
np.random.seed(SEED)
enc = SentenceTransformer('all-MiniLM-L6-v2')

ds = load_dataset("google/air_dialogue", split="train", streaming=True)
states, actions, y, groups = [], [], [], []
n_parsed = 0
for ex in ds:
    if n_parsed >= N_DIALOGUES:
        break
    dialogue = ex.get('dialogue', '')
    correct = ex.get('correct_sample', None)
    if not dialogue or correct is None:
        continue
    turns = [t.strip() for t in dialogue.split('\n') if t.strip()] if isinstance(dialogue, str) \
        else [str(t) for t in dialogue]
    if len(turns) < 2:
        continue
    outcome = 1.0 if correct else 0.0
    for i in range(1, len(turns)):
        states.append('\n'.join(turns[:i])[:1024])
        actions.append(turns[i][:512])
        y.append(outcome)
        groups.append(n_parsed)
    n_parsed += 1

y = np.array(y, float); groups = np.array(groups)
print(f"{n_parsed} dialogues, {len(y)} transitions, positive rate {y.mean():.3f}")
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
