"""
R20: Segmentation-sensitivity of state-lift (AC ask 2 / reviewer 7ccj).
Question: is measured SL an intrinsic property of the task, or an artifact of
how state/action are segmented and formatted?

Domain: DealOrNoDeal (the surviving genuine-label domain), continuous
deal-value quality, GroupKFold by dialogue (leakage-proof), MiniLM+PCA16+Ridge.

Segmentation grid:
  state ∈ { full-prefix (r17d default), last-turn-only, last-2-turns, empty }
  action ∈ { turn-text (default), turn-text + speaker tag }
The 'empty' state row gives the floor (SL must be ~0 by construction).
"""

import numpy as np
import re, json
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sentence_transformers import SentenceTransformer

SEED = 42
D_PCA = 16
N_MAX = 15000
OUT = "results/r20_segmentation_sensitivity.json"
np.random.seed(SEED)

enc = SentenceTransformer('all-MiniLM-L6-v2')

# ---- parse DoND (tagged format) ----
dialogues = []
with open("data/dealornodeal_train.txt") as f:
    for line in f:
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
        raw = [t.strip() for t in dlg.group(1).split("<eos>") if t.strip() and "<selection>" not in t]
        if len(raw) < 2:
            continue
        turns, speakers = [], []
        for t in raw:
            if t.startswith("YOU:"): turns.append(t[4:].strip()); speakers.append("YOU")
            elif t.startswith("THEM:"): turns.append(t[5:].strip()); speakers.append("THEM")
            else: turns.append(t); speakers.append("UNK")
        dialogues.append((turns, speakers, quality))

print(f"{len(dialogues)} dialogues")

def build(state_mode, action_mode):
    states, actions, y, groups = [], [], [], []
    for di, (turns, speakers, quality) in enumerate(dialogues):
        for i in range(1, len(turns)):
            if state_mode == 'full_prefix':
                st = " ".join(t[:80] for t in turns[:i])[:300]
            elif state_mode == 'last_turn':
                st = turns[i-1][:300]
            elif state_mode == 'last_2':
                st = " ".join(t[:150] for t in turns[max(0, i-2):i])[:300]
            elif state_mode == 'empty':
                st = "[NONE]"
            states.append("Negotiation: " + st)
            if action_mode == 'plain':
                actions.append(turns[i][:200])
            else:
                actions.append(f"{speakers[i]}: {turns[i]}"[:200])
            y.append(quality)
            groups.append(di)
    return states, actions, np.array(y, dtype=float), np.array(groups)

def sl_grouped(states, actions, y, groups):
    if len(y) > N_MAX:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(y), N_MAX, replace=False)
        states = [states[i] for i in idx]; actions = [actions[i] for i in idx]
        y = y[idx]; groups = groups[idx]
    es = enc.encode(states, batch_size=128, show_progress_bar=False)
    ea = enc.encode(actions, batch_size=128, show_progress_bar=False)
    ps = PCA(n_components=D_PCA, random_state=SEED).fit_transform(es)
    pa = PCA(n_components=D_PCA, random_state=SEED).fit_transform(ea)
    cv = GroupKFold(n_splits=5)
    r2a = cross_val_score(Ridge(alpha=1.0), pa, y, cv=cv, groups=groups, scoring='r2').mean()
    r2sa = cross_val_score(Ridge(alpha=1.0), np.hstack([ps, pa]), y, cv=cv, groups=groups, scoring='r2').mean()
    return float(r2a), float(r2sa), float(r2sa - r2a)

results = {}
for sm in ['full_prefix', 'last_turn', 'last_2', 'empty']:
    for am in ['plain', 'speaker']:
        key = f"{sm}|{am}"
        s, a, y, g = build(sm, am)
        r2a, r2sa, sl = sl_grouped(s, a, y, g)
        results[key] = {'r2_action': r2a, 'r2_state_action': r2sa, 'sl': sl}
        print(f"  {key:22s}: R2(a)={r2a:+.4f} R2(s+a)={r2sa:+.4f} SL={sl:+.4f}")

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved {OUT}")
