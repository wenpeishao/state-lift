"""
E9: PersuasionForGood pair data for LLM reward training — PROSPECTIVE test.
SL was measured FIRST (r22: grouped SL = -0.001) -> framework predicts a small
blind-vs-conditioned lift. Training runs after the prediction is on record.

Construction follows e5/e8 pattern: persuader turns, quality = persuadee donation
> median (binary), position-bucketed cross-dialogue pos/neg pairing, own-state
conditioning, dialogue-disjoint 80/20 split from the start.
"""

import numpy as np
import csv, os
import torch
from collections import defaultdict

SEED = 42
MAX_TRAIN = 5000
MAX_TEST = 1500
OUT = "data/e9_p4g_data.pt"
np.random.seed(SEED)

# donation map (persuadee row B4==1, amount B6)
donation_map = {}
with open("data/persuasion_info.csv", encoding='utf-8') as f:
    for r in csv.DictReader(f):
        try:
            amt = float(r.get('B6', '') or 'nan')
        except ValueError:
            continue
        if amt != amt:
            continue
        if r.get('B4', '') == '1' or r.get('B2', '') not in donation_map:
            donation_map[r.get('B2', '')] = amt

with open("data/persuasion_full.csv", encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

dialogues = defaultdict(list)
for r in rows:
    dialogues[r['B2']].append(r)

donations = [donation_map[d] for d in dialogues if d in donation_map]
med = np.median(donations)
print(f"{len(dialogues)} dialogues, median donation = {med}")

turns_data = []
for di, (did, turns) in enumerate(sorted(dialogues.items())):
    if did not in donation_map:
        continue
    label = 1 if donation_map[did] > med else 0
    texts = [t['Unit'] for t in turns if t.get('Unit', '').strip()]
    roles = [t.get('B4', '') for t in turns if t.get('Unit', '').strip()]
    if len(texts) < 4:
        continue
    for i in range(2, len(texts)):
        if roles[i] != '0':   # persuader turns only (the acting agent)
            continue
        turns_data.append({
            'state': '\n'.join(texts[:i])[:1024],
            'action': texts[i][:512],
            'label': label,
            'dialogue_id': di,
            'turn_idx': i,
        })

print(f"{len(turns_data)} persuader turns, "
      f"{sum(t['label'] for t in turns_data)} pos / {sum(1-t['label'] for t in turns_data)} neg")

# dialogue-disjoint split FIRST
dids = sorted(set(t['dialogue_id'] for t in turns_data))
np.random.shuffle(dids)
cut = int(0.8 * len(dids))
train_d, test_d = set(dids[:cut]), set(dids[cut:])

def build_pairs(turns, seed):
    by_pos = defaultdict(lambda: {'pos': [], 'neg': []})
    for t in turns:
        by_pos[min(t['turn_idx'] // 3, 3)]['pos' if t['label'] == 1 else 'neg'].append(t)
    rng = np.random.default_rng(seed)
    pairs = []
    for b, g in by_pos.items():
        n = min(len(g['pos']), len(g['neg']))
        if n == 0:
            continue
        rng.shuffle(g['pos']); rng.shuffle(g['neg'])
        for i in range(n):
            pairs.append((g['pos'][i], g['neg'][i]))
    rng.shuffle(pairs)
    return pairs

train_pairs = build_pairs([t for t in turns_data if t['dialogue_id'] in train_d], SEED)[:MAX_TRAIN]
test_pairs = build_pairs([t for t in turns_data if t['dialogue_id'] in test_d], SEED + 1)[:MAX_TEST]
print(f"pairs: {len(train_pairs)} train / {len(test_pairs)} test (dialogue-disjoint)")

bc = lambda s, a: f"Given this persuasion conversation:\n{s}\n\nRate this persuader response:\n{a}"
bb = lambda a: f"Rate this persuader response:\n{a}"

data = {
    'train_chosen_blind':    [bb(p['action']) for p, _ in train_pairs],
    'train_rejected_blind':  [bb(n['action']) for _, n in train_pairs],
    'train_chosen_state':    [bc(p['state'], p['action']) for p, _ in train_pairs],
    'train_rejected_state':  [bc(n['state'], n['action']) for _, n in train_pairs],
    'test_chosen_blind':     [bb(p['action']) for p, _ in test_pairs],
    'test_rejected_blind':   [bb(n['action']) for _, n in test_pairs],
    'test_chosen_state':     [bc(p['state'], p['action']) for p, _ in test_pairs],
    'test_rejected_state':   [bc(n['state'], n['action']) for _, n in test_pairs],
    'metadata': {
        'dataset': 'PersuasionForGood-dsplit',
        'quality_signal': 'persuadee donation > median',
        'split': 'dialogue-disjoint',
        'prospective_note': 'SL measured BEFORE training: r22 grouped SL = -0.001 -> predicted lift small (<~0.02)',
        'n_train': len(train_pairs), 'n_test': len(test_pairs),
    },
}
torch.save(data, OUT)
print(f"Saved {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
