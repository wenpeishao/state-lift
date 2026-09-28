"""
E4: Preprocess CodeContests for PRM vs ORM training.
Quality signal: steps from CORRECT solutions (label=1) vs INCORRECT solutions (label=0).
"""

import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datasets import load_dataset

SEED = 42
np.random.seed(SEED)
OUT_PATH = 'data/e4_code_data.pt'

print("[1] Loading CodeContests...")
ds = load_dataset('deepmind/code_contests', split='train')
print(f"  {len(ds)} problems")

# Build chains from correct and incorrect solutions
chains = []
for ex in list(ds)[:1500]:
    solutions = ex.get('solutions', {})
    incorrect = ex.get('incorrect_solutions', {})
    correct_sols = solutions.get('solution', []) if isinstance(solutions, dict) else []
    incorrect_sols = incorrect.get('solution', []) if isinstance(incorrect, dict) else []
    desc = str(ex.get('description', ''))[:512]

    for sol, label in [(s, 1) for s in correct_sols[:2]] + [(s, 0) for s in incorrect_sols[:2]]:
        if not isinstance(sol, str) or len(sol) < 50:
            continue
        lines = sol.split('\n')
        blocks, current = [], []
        for line in lines:
            current.append(line)
            if len('\n'.join(current)) > 100 and (
                line.strip() == '' or line.strip().startswith('def ') or
                line.strip().startswith('class ')):
                blocks.append('\n'.join(current))
                current = []
        if current:
            blocks.append('\n'.join(current))
        if len(blocks) >= 3:
            chains.append({'description': desc, 'blocks': [b[:512] for b in blocks], 'label': label})

n_correct = sum(1 for c in chains if c['label'] == 1)
n_incorrect = sum(1 for c in chains if c['label'] == 0)
print(f"  {len(chains)} chains ({n_correct} correct, {n_incorrect} incorrect)")

# Build transitions: state = description + previous blocks, action = current block
print("[2] Building transitions...")
pairs = []
for chain in chains:
    blocks = chain['blocks']
    for i in range(len(blocks)):
        if i == 0:
            state = chain['description']
        else:
            state = chain['description'] + "\n\n" + "\n".join(blocks[:i])
        state = state[:1024]
        action = blocks[i]
        pairs.append({
            'state': state,
            'action': action,
            'label': chain['label'],
        })

print(f"  {len(pairs)} step-level pairs")

# Build text prompts
print("[3] Building prompts...")
conditioned_texts = []
blind_texts = []
labels = []

for p in pairs:
    conditioned_texts.append(
        f"Given the following code context:\n{p['state']}\n\n"
        f"Rate the quality of this next code block:\n{p['action']}"
    )
    blind_texts.append(
        f"Rate the quality of this code block:\n{p['action']}"
    )
    labels.append(p['label'])

# Embed states for PCA
print("[4] Embedding states...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
state_texts = [p['state'] for p in pairs]
# Deduplicate for speed
unique_states = list(set(state_texts))
print(f"  {len(unique_states)} unique states")
state_embs_dict = {}
batch_size = 256
for i in range(0, len(unique_states), batch_size):
    batch = unique_states[i:i+batch_size]
    embs = encoder.encode(batch, show_progress_bar=False)
    for text, emb in zip(batch, embs):
        state_embs_dict[text] = emb

state_embs = np.array([state_embs_dict[s] for s in state_texts])

pca = PCA(n_components=16)
state_vectors = pca.fit_transform(state_embs)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# Split by chain (80/20)
print("[5] Splitting...")
np.random.seed(SEED)
n_chains = len(chains)
perm = np.random.permutation(n_chains)
n_train_chains = int(0.8 * n_chains)
train_chain_set = set(perm[:n_train_chains].tolist())

# Map each pair back to its chain
pair_chain_idx = []
pair_idx = 0
for ci, chain in enumerate(chains):
    for _ in chain['blocks']:
        pair_chain_idx.append(ci)
        pair_idx += 1

train_mask = [pair_chain_idx[i] in train_chain_set for i in range(len(pairs))]
test_mask = [not m for m in train_mask]

train_idx = [i for i, m in enumerate(train_mask) if m]
test_idx = [i for i, m in enumerate(test_mask) if m]

# Cap
MAX_TRAIN, MAX_TEST = 5000, 1500
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)
train_idx = train_idx[:MAX_TRAIN]
test_idx = test_idx[:MAX_TEST]

print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
print(f"  Train labels: {sum(labels[i] for i in train_idx)} pos / {sum(1-labels[i] for i in train_idx)} neg")
print(f"  Test labels: {sum(labels[i] for i in test_idx)} pos / {sum(1-labels[i] for i in test_idx)} neg")

# Save
data = {
    'train_texts_conditioned': [conditioned_texts[i] for i in train_idx],
    'train_texts_blind': [blind_texts[i] for i in train_idx],
    'train_labels': torch.tensor([labels[i] for i in train_idx], dtype=torch.long),
    'train_state_vectors': torch.tensor(state_vectors[train_idx], dtype=torch.float32),
    'test_texts_conditioned': [conditioned_texts[i] for i in test_idx],
    'test_texts_blind': [blind_texts[i] for i in test_idx],
    'test_labels': torch.tensor([labels[i] for i in test_idx], dtype=torch.long),
    'test_state_vectors': torch.tensor(state_vectors[test_idx], dtype=torch.float32),
    'metadata': {
        'n_train': len(train_idx), 'n_test': len(test_idx),
        'dataset': 'deepmind/code_contests',
        'quality_signal': 'correct_vs_incorrect_solution',
        'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
    },
}
torch.save(data, OUT_PATH)
print(f"\nSaved to {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")

import os
print(f"\nSaved to {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
