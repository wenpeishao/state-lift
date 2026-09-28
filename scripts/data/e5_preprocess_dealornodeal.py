"""
E5: Pre-process DealOrNoDeal for SLURM training.
Moderate SDI-SL domain (SL=0.18) with genuine deal-outcome labels.

Uses deal outcome (agent value / max value) as quality signal:
- Each turn labeled by whether the speaker's final deal value > median.
- Builds pairwise data: high-outcome turns (chosen) vs low-outcome turns (rejected).
"""

import numpy as np
import os, re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from collections import defaultdict

SEED = 42
D_PCA = 16
MAX_TRAIN = 5000
MAX_TEST = 1500
OUT_DIR = 'data'
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

print("=" * 70)
print("E5: PREPROCESSING DEALORNODEAL WITH DEAL-OUTCOME LABELS")
print("=" * 70)

# ===================================================================
# 1. PARSE RAW DATA
# ===================================================================
print("\n[1] Parsing DealOrNoDeal...")

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

        # Parse values
        toks = inp.group(1).strip().split()
        if len(toks) < 6:
            continue
        try:
            counts = [int(toks[0]), int(toks[2]), int(toks[4])]
            values = [int(toks[1]), int(toks[3]), int(toks[5])]
        except:
            continue

        # Parse output (agent's chosen items)
        items = re.findall(r"item\d+=(\d+)", out.group(1))
        if len(items) >= 3:
            chosen = [int(x) for x in items[:3]]
            agent_value = sum(c * v for c, v in zip(chosen, values))
            max_value = sum(c * v for c, v in zip(counts, values))
        else:
            agent_value = 0
            max_value = 10

        # Parse turns
        turns = [t.strip() for t in dlg.group(1).split("<eos>")
                 if t.strip() and "<selection>" not in t]
        if len(turns) < 3:
            continue

        clean_turns = []
        turn_speakers = []
        for t in turns:
            if t.startswith("YOU:"):
                clean_turns.append(t.replace("YOU:", "").strip())
                turn_speakers.append("YOU")
            elif t.startswith("THEM:"):
                clean_turns.append(t.replace("THEM:", "").strip())
                turn_speakers.append("THEM")
            else:
                clean_turns.append(t)
                turn_speakers.append("UNK")

        dialogues.append({
            'turns': clean_turns,
            'speakers': turn_speakers,
            'agent_value': agent_value,
            'max_value': max_value,
            'quality': agent_value / max(max_value, 1),
        })

print(f"  {len(dialogues)} dialogues parsed")

# ===================================================================
# 2. EXTRACT TURNS WITH OUTCOME LABELS
# ===================================================================
print("\n[2] Extracting turns with outcome labels...")

all_qualities = [d['quality'] for d in dialogues]
median_quality = np.median(all_qualities)
print(f"  Quality distribution: mean={np.mean(all_qualities):.3f}, median={median_quality:.3f}")

turns_data = []
for di, d in enumerate(dialogues):
    label = 1 if d['quality'] > median_quality else 0

    for i in range(2, len(d['turns'])):
        # Only use "YOU" turns (the agent's perspective)
        if d['speakers'][i] != 'YOU':
            continue

        state = '\n'.join(d['turns'][:i])[:1024]
        action = d['turns'][i][:512]

        turns_data.append({
            'state': state,
            'action': action,
            'label': label,
            'quality': d['quality'],
            'dialogue_id': di,
            'turn_idx': i,
        })

print(f"  Total turns: {len(turns_data)}")
print(f"  Positive: {sum(t['label'] for t in turns_data)}, "
      f"Negative: {sum(1-t['label'] for t in turns_data)}")

# ===================================================================
# 3. BUILD PAIRWISE DATA
# ===================================================================
print("\n[3] Building pairwise data...")

by_position = defaultdict(lambda: {'pos': [], 'neg': []})
for t in turns_data:
    bucket = min(t['turn_idx'] // 2, 4)
    if t['label'] == 1:
        by_position[bucket]['pos'].append(t)
    else:
        by_position[bucket]['neg'].append(t)

pairs = []
np.random.seed(SEED)
for bucket, groups in by_position.items():
    pos = groups['pos']
    neg = groups['neg']
    n_pairs = min(len(pos), len(neg))
    if n_pairs == 0:
        continue
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    for i in range(n_pairs):
        pairs.append({
            'state_chosen': pos[i]['state'],
            'action_chosen': pos[i]['action'],
            'state_rejected': neg[i]['state'],
            'action_rejected': neg[i]['action'],
        })

np.random.shuffle(pairs)
print(f"  Total pairs: {len(pairs)}")

# ===================================================================
# 4. TRAIN/TEST SPLIT
# ===================================================================
print("\n[4] Splitting 80/20...")
n = len(pairs)
idx = np.arange(n)
np.random.shuffle(idx)
split = int(0.8 * n)
train_idx = idx[:split]
test_idx = idx[split:]

if len(train_idx) > MAX_TRAIN:
    train_idx = train_idx[:MAX_TRAIN]
if len(test_idx) > MAX_TEST:
    test_idx = test_idx[:MAX_TEST]

print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

# ===================================================================
# 5. BUILD TEXT PROMPTS
# ===================================================================
print("\n[5] Building prompts...")

def build_conditioned(state, action):
    return f"Given this negotiation:\n{state}\n\nRate this response:\n{action}"

def build_blind(action):
    return f"Rate this negotiation response:\n{action}"

train_chosen_blind = [build_blind(pairs[i]['action_chosen']) for i in train_idx]
train_rejected_blind = [build_blind(pairs[i]['action_rejected']) for i in train_idx]
train_chosen_state = [build_conditioned(pairs[i]['state_chosen'], pairs[i]['action_chosen']) for i in train_idx]
train_rejected_state = [build_conditioned(pairs[i]['state_rejected'], pairs[i]['action_rejected']) for i in train_idx]

test_chosen_blind = [build_blind(pairs[i]['action_chosen']) for i in test_idx]
test_rejected_blind = [build_blind(pairs[i]['action_rejected']) for i in test_idx]
test_chosen_state = [build_conditioned(pairs[i]['state_chosen'], pairs[i]['action_chosen']) for i in test_idx]
test_rejected_state = [build_conditioned(pairs[i]['state_rejected'], pairs[i]['action_rejected']) for i in test_idx]

# ===================================================================
# 6. SAVE
# ===================================================================
data = {
    'train_chosen_blind': train_chosen_blind,
    'train_rejected_blind': train_rejected_blind,
    'train_chosen_state': train_chosen_state,
    'train_rejected_state': train_rejected_state,
    'test_chosen_blind': test_chosen_blind,
    'test_rejected_blind': test_rejected_blind,
    'test_chosen_state': test_chosen_state,
    'test_rejected_state': test_rejected_state,
    'metadata': {
        'dataset': 'DealOrNoDeal',
        'quality_signal': 'deal_outcome_value',
        'median_quality': float(median_quality),
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'n_dialogues': len(dialogues),
        'n_total_turns': len(turns_data),
        'sdi_sl': 0.180,
    },
}

out_path = os.path.join(OUT_DIR, 'e5_dealornodeal_data.pt')
torch.save(data, out_path)
print(f"\nSaved to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

print(f"\nSummary:")
print(f"  Train: {len(train_chosen_blind)} pairs")
print(f"  Test: {len(test_chosen_blind)} pairs")
print(f"  Avg conditioned text: {np.mean([len(t) for t in train_chosen_state]):.0f} chars")
print(f"  Avg blind text: {np.mean([len(t) for t in train_chosen_blind]):.0f} chars")
