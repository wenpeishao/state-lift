"""
E4: Pre-process Anthropic HH-RLHF (helpful-base) for SSCC training.
NEGATIVE CONTROL: chat preference quality should show zero PRM advantage.

Parses chosen/rejected conversation pairs, finds divergence point,
extracts state (prefix) and action (first divergent assistant response).
Saves as .pt file with blind + conditioned text prompts + PCA-16 state vectors.
"""

import numpy as np
import os
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

SEED = 42
D_PCA = 16
MAX_TRAIN = 5000
MAX_TEST = 1500
OUT_DIR = 'data'
np.random.seed(SEED)


def parse_conversation(text):
    """Parse HH-RLHF conversation text into list of (role, content) turns."""
    turns = []
    lines = text.strip().split('\n\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Human: '):
            turns.append(('Human', line[len('Human: '):].strip()))
        elif line.startswith('Assistant: '):
            turns.append(('Assistant', line[len('Assistant: '):].strip()))
    return turns


def find_divergence(chosen_turns, rejected_turns):
    """Find the index where chosen and rejected conversations diverge."""
    min_len = min(len(chosen_turns), len(rejected_turns))
    for i in range(min_len):
        if chosen_turns[i] != rejected_turns[i]:
            return i
    return min_len


def turns_to_text(turns):
    """Convert list of (role, content) turns into formatted text."""
    parts = []
    for role, content in turns:
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


# ===================================================================
# 1. LOAD HH-RLHF
# ===================================================================
print("Loading HH-RLHF (helpful-base)...")
ds = load_dataset('Anthropic/hh-rlhf', data_dir='helpful-base', split='train')
print(f"  Total examples: {len(ds)}")

# ===================================================================
# 2. PARSE AND FIND DIVERGENCE POINTS
# ===================================================================
print("Parsing conversations and finding divergence points...")
pairs = []
skipped_short = 0
skipped_no_assistant = 0

for ex in ds:
    chosen_turns = parse_conversation(ex['chosen'])
    rejected_turns = parse_conversation(ex['rejected'])

    div_idx = find_divergence(chosen_turns, rejected_turns)

    # Need at least 2 turns of prefix (1 human + 1 assistant minimum)
    if div_idx < 2:
        skipped_short += 1
        continue

    # The divergence should be at an Assistant turn
    # (both branches share prefix, then assistant responds differently)
    if div_idx >= len(chosen_turns) or div_idx >= len(rejected_turns):
        skipped_no_assistant += 1
        continue

    # Get the first divergent assistant response
    chosen_action = chosen_turns[div_idx]
    rejected_action = rejected_turns[div_idx]

    # Both should be assistant turns at divergence
    if chosen_action[0] != 'Assistant' or rejected_action[0] != 'Assistant':
        skipped_no_assistant += 1
        continue

    prefix = chosen_turns[:div_idx]  # shared prefix

    pairs.append({
        'state': turns_to_text(prefix),
        'action_chosen': chosen_action[1],
        'action_rejected': rejected_action[1],
        'n_prefix_turns': div_idx,
    })

print(f"  Valid pairs: {len(pairs)}")
print(f"  Skipped (prefix < 2 turns): {skipped_short}")
print(f"  Skipped (no assistant divergence): {skipped_no_assistant}")

# ===================================================================
# 3. EMBED STATES FOR PCA
# ===================================================================
print("Embedding states with all-MiniLM-L6-v2...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

state_texts = [p['state'] for p in pairs]
state_embeddings = embedder.encode(state_texts, show_progress_bar=True, batch_size=64)

print(f"  Embeddings shape: {state_embeddings.shape}")

# PCA-16
pca = PCA(n_components=D_PCA)
state_pca = pca.fit_transform(state_embeddings)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# ===================================================================
# 4. TRAIN/TEST SPLIT BY CONVERSATION
# ===================================================================
print("Splitting 80/20 by conversation...")
n_pairs = len(pairs)
indices = np.arange(n_pairs)
np.random.shuffle(indices)

split_point = int(0.8 * n_pairs)
train_indices = indices[:split_point]
test_indices = indices[split_point:]

# Cap sizes
if len(train_indices) > MAX_TRAIN:
    train_indices = train_indices[:MAX_TRAIN]
if len(test_indices) > MAX_TEST:
    test_indices = test_indices[:MAX_TEST]

print(f"  Train pairs: {len(train_indices)}")
print(f"  Test pairs: {len(test_indices)}")

# ===================================================================
# 5. BUILD TEXT PROMPTS
# ===================================================================
print("Building text prompts...")


def build_conditioned(state, action):
    return f"Given this conversation:\n{state}\n\nRate this response:\n{action}"


def build_blind(action):
    return f"Rate this response:\n{action}"


# Train
train_chosen_blind = [build_blind(pairs[i]['action_chosen']) for i in train_indices]
train_rejected_blind = [build_blind(pairs[i]['action_rejected']) for i in train_indices]
train_chosen_state = [build_conditioned(pairs[i]['state'], pairs[i]['action_chosen']) for i in train_indices]
train_rejected_state = [build_conditioned(pairs[i]['state'], pairs[i]['action_rejected']) for i in train_indices]

# Test
test_chosen_blind = [build_blind(pairs[i]['action_chosen']) for i in test_indices]
test_rejected_blind = [build_blind(pairs[i]['action_rejected']) for i in test_indices]
test_chosen_state = [build_conditioned(pairs[i]['state'], pairs[i]['action_chosen']) for i in test_indices]
test_rejected_state = [build_conditioned(pairs[i]['state'], pairs[i]['action_rejected']) for i in test_indices]

# State vectors for stratified eval
test_state_pca = state_pca[test_indices]
test_z_mean = state_pca[train_indices].mean(axis=0)
test_z_dist = np.linalg.norm(test_state_pca - test_z_mean, axis=1)

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
    'test_z_dist': torch.FloatTensor(test_z_dist),
}

out_path = os.path.join(OUT_DIR, 'e4_hh_data.pt')
torch.save(data, out_path)
print(f"\nSaved to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

# Summary stats
print(f"\nSummary:")
print(f"  Train: {len(train_chosen_blind)} pairs")
print(f"  Test: {len(test_chosen_blind)} pairs")
print(f"  Avg conditioned text len (train): {np.mean([len(t) for t in train_chosen_state]):.0f} chars")
print(f"  Avg blind text len (train): {np.mean([len(t) for t in train_chosen_blind]):.0f} chars")
