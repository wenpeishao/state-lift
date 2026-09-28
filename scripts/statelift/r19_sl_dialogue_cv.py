"""
R19: Does the state-lift DIAGNOSTIC survive dialogue-level cross-validation?
Complements the e8/e6 training-side leakage check (which showed lifts survive).

For CaSiNo and DealOrNoDeal: compute SL = R2(state+action) - R2(action)
with (a) random KFold over turns (original protocol) and
     (b) GroupKFold by dialogue (leakage-proof).
MiniLM embeddings + PCA-16 + Ridge, matching the paper's pipeline.
"""

import numpy as np
import re, json, os, sys
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold
from sentence_transformers import SentenceTransformer

SEED = 42
D_PCA = 16
OUT = "results/r19_sl_dialogue_cv.json"
np.random.seed(SEED)

def extract_casino():
    from datasets import load_dataset
    ds = load_dataset('casino', split='train')
    all_points = []
    for ex in ds:
        for aid in ['mturk_agent_1', 'mturk_agent_2']:
            pts = ex['participant_info'].get(aid, {}).get('outcomes', {}).get('points_scored', None)
            if pts is not None:
                all_points.append(int(pts))
    med = np.median(all_points)
    rows = []
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
        for i in range(2, len(texts)):
            pts = pts_map.get(agents[i], None)
            if pts is None:
                continue
            rows.append(('\n'.join(texts[:i])[:1024], texts[i][:512], 1 if pts > med else 0, di))
    return rows

def extract_dond():
    path = None
    for cand in ["data/dealornodeal_train.txt",
                 "data/dealornodeal_train.txt"]:
        if os.path.exists(cand):
            path = cand; break
    if path is None:
        # download from github (local machine has internet)
        import urllib.request
        url = "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/train.txt"
        path = "data/dealornodeal_train.txt"
        urllib.request.urlretrieve(url, path)
    rows = []
    with open(path) as f:
        for di, line in enumerate(f):
            line = line.strip()
            if not line: continue
            inp = re.search(r"<input>\s*(.*?)\s*</input>", line)
            dlg = re.search(r"<dialogue>\s*(.*?)\s*</dialogue>", line)
            out = re.search(r"<output>\s*(.*?)\s*</output>", line)
            if not inp or not dlg or not out: continue
            toks = inp.group(1).strip().split()
            if len(toks) < 6: continue
            try:
                counts = [int(toks[0]), int(toks[2]), int(toks[4])]
                values = [int(toks[1]), int(toks[3]), int(toks[5])]
            except: continue
            items = re.findall(r"item\d+=(\d+)", out.group(1))
            if len(items) >= 3:
                chosen = [int(x) for x in items[:3]]
                av = sum(c*v for c, v in zip(chosen, values))
                mv = sum(c*v for c, v in zip(counts, values))
            else:
                av, mv = 0, 10
            quality = av / max(mv, 1)
            turns_raw = [t.strip() for t in dlg.group(1).split("<eos>") if t.strip() and "<selection>" not in t]
            if len(turns_raw) < 3: continue
            clean, spk = [], []
            for t in turns_raw:
                if t.startswith("YOU:"): clean.append(t[4:].strip()); spk.append("YOU")
                elif t.startswith("THEM:"): clean.append(t[5:].strip()); spk.append("THEM")
                else: clean.append(t); spk.append("UNK")
            rows.append((clean, spk, quality, di))
    quals = [r[2] for r in rows]
    med = np.median(quals)
    out_rows = []
    for clean, spk, quality, di in rows:
        label = 1 if quality > med else 0
        for i in range(2, len(clean)):
            if spk[i] != 'YOU': continue
            out_rows.append(('\n'.join(clean[:i])[:1024], clean[i][:512], label, di))
    return out_rows

def compute_sl(rows, embedder, mode, n_max=6000):
    rng = np.random.default_rng(SEED)
    if len(rows) > n_max:
        idx = rng.choice(len(rows), n_max, replace=False)
        rows = [rows[i] for i in idx]
    states = [r[0] for r in rows]; actions = [r[1] for r in rows]
    y = np.array([r[2] for r in rows], dtype=float)
    groups = np.array([r[3] for r in rows])
    es = embedder.encode(states, batch_size=128, show_progress_bar=False)
    ea = embedder.encode(actions, batch_size=128, show_progress_bar=False)
    ps = PCA(n_components=D_PCA, random_state=SEED).fit_transform(es)
    pa = PCA(n_components=D_PCA, random_state=SEED).fit_transform(ea)
    Xa = pa
    Xsa = np.hstack([ps, pa])
    if mode == 'random':
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = cv.split(Xa)
    else:
        cv = GroupKFold(n_splits=5)
        splits = cv.split(Xa, y, groups)
    r2a_list, r2sa_list = [], []
    for tr, te in splits:
        ra = Ridge(alpha=1.0).fit(Xa[tr], y[tr])
        rsa = Ridge(alpha=1.0).fit(Xsa[tr], y[tr])
        ssr_a = np.sum((y[te] - ra.predict(Xa[te]))**2)
        ssr_sa = np.sum((y[te] - rsa.predict(Xsa[te]))**2)
        sst = np.sum((y[te] - y[te].mean())**2)
        r2a_list.append(1 - ssr_a/sst)
        r2sa_list.append(1 - ssr_sa/sst)
    r2a, r2sa = np.mean(r2a_list), np.mean(r2sa_list)
    return {'r2_action': float(r2a), 'r2_state_action': float(r2sa), 'sl': float(r2sa - r2a), 'n': len(rows)}

def main():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    results = {}
    for name, fn in [('casino', extract_casino), ('dealornodeal', extract_dond)]:
        print(f"[{name}] extracting...")
        rows = fn()
        print(f"  {len(rows)} turns from {len(set(r[3] for r in rows))} dialogues")
        for mode in ['random', 'grouped']:
            res = compute_sl(rows, embedder, mode)
            results[f"{name}_{mode}"] = res
            print(f"  {mode:8s}: SL={res['sl']:+.4f} (R2a={res['r2_action']:.4f}, R2sa={res['r2_state_action']:.4f}, n={res['n']})")
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT}")

if __name__ == '__main__':
    main()
