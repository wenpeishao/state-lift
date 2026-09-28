"""
E4: Preprocess GSM8K for PRM vs ORM math reward model experiment.
Runs LOCALLY (Windows, has internet, has GPU for embeddings).
Saves .pt file to Z: drive for SSCC access via NFS.

Creates positive pairs (step in correct chain context) and negative pairs
(same step paired with state from a DIFFERENT chain).
"""

import numpy as np
import os
import re
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

SEED = 42
D_PCA = 16
MAX_TRAIN = 5000
MAX_TEST = 1500
MIN_STEPS = 3
MIN_STEP_LEN = 10
OUT_DIR = 'data'

np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================================================================
# 1. LOAD GSM8K
# ===================================================================
print("[1/6] Loading GSM8K...")
ds = load_dataset('openai/gsm8k', 'main', split='train')
print(f"  Raw examples: {len(ds)}")

# ===================================================================
# 2. PARSE INTO CHAINS OF STEPS
# ===================================================================
print("[2/6] Parsing reasoning chains...")

chains = []
for ex in ds:
    question = ex['question']
    answer = ex['answer']

    # Split answer into steps
    raw_steps = answer.split('\n')
    steps = [s.strip() for s in raw_steps if len(s.strip()) > MIN_STEP_LEN]

    if len(steps) >= MIN_STEPS:
        chains.append({
            'question': question,
            'steps': steps,
        })

print(f"  Chains with >= {MIN_STEPS} steps: {len(chains)}")

# ===================================================================
# 3. BUILD TRANSITIONS (state, action pairs)
# ===================================================================
print("[3/6] Building transitions...")

transitions = []
for ci, chain in enumerate(chains):
    q = chain['question']
    steps = chain['steps']

    for si in range(len(steps)):
        # State = question + all previous steps
        if si == 0:
            state_text = f"Question: {q}"
        else:
            prev = "\n".join(steps[:si])
            state_text = f"Question: {q}\n\nPrevious steps:\n{prev}"

        action_text = steps[si]

        transitions.append({
            'chain_idx': ci,
            'step_idx': si,
            'state_text': state_text,
            'action_text': action_text,
        })

print(f"  Total transitions: {len(transitions)}")

# ===================================================================
# 4. BUILD POSITIVE AND NEGATIVE PAIRS
# ===================================================================
print("[4/6] Building positive/negative pairs...")

# Group transitions by chain
chain_to_trans = {}
for ti, t in enumerate(transitions):
    ci = t['chain_idx']
    if ci not in chain_to_trans:
        chain_to_trans[ci] = []
    chain_to_trans[ci].append(ti)

# For each transition: positive = (own state, own action, label=1)
# negative = (random OTHER chain's state, same action, label=0)
all_chain_ids = list(chain_to_trans.keys())

pairs = []
for ti, t in enumerate(transitions):
    ci = t['chain_idx']

    # Positive pair
    pairs.append({
        'chain_idx': ci,
        'state_text': t['state_text'],
        'action_text': t['action_text'],
        'label': 1,
    })

    # Negative pair: same action, state from different chain
    other_ci = ci
    while other_ci == ci:
        other_ci = np.random.choice(all_chain_ids)

    other_trans = chain_to_trans[other_ci]
    other_ti = np.random.choice(other_trans)
    other_state = transitions[other_ti]['state_text']

    pairs.append({
        'chain_idx': ci,
        'state_text': other_state,
        'action_text': t['action_text'],
        'label': 0,
    })

print(f"  Total pairs: {len(pairs)}")

# ===================================================================
# 5. EMBED STATES AND PCA
# ===================================================================
print("[5/6] Embedding states with all-MiniLM-L6-v2...")

# Collect unique state texts
unique_states = list(set(p['state_text'] for p in pairs))
state_to_idx = {s: i for i, s in enumerate(unique_states)}
print(f"  Unique states: {len(unique_states)}")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(unique_states, batch_size=256, show_progress_bar=True,
                          convert_to_numpy=True)
del model

print(f"  Embedding shape: {embeddings.shape}")

# PCA to 16 dimensions
pca = PCA(n_components=D_PCA)
embeddings_pca = pca.fit_transform(embeddings)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# Map each pair to its PCA state vector
pair_state_vectors = np.array([
    embeddings_pca[state_to_idx[p['state_text']]] for p in pairs
])

# ===================================================================
# 6. SPLIT BY CHAIN, BUILD TEXTS, SAVE
# ===================================================================
print("[6/6] Splitting and saving...")

# Split by chain (80/20)
unique_chains = np.array(list(set(p['chain_idx'] for p in pairs)))
np.random.shuffle(unique_chains)
split_point = int(0.8 * len(unique_chains))
train_chains = set(unique_chains[:split_point])
test_chains = set(unique_chains[split_point:])

train_indices = [i for i, p in enumerate(pairs) if p['chain_idx'] in train_chains]
test_indices = [i for i, p in enumerate(pairs) if p['chain_idx'] in test_chains]

# Cap sizes
if len(train_indices) > MAX_TRAIN:
    train_indices = list(np.random.choice(train_indices, MAX_TRAIN, replace=False))
if len(test_indices) > MAX_TEST:
    test_indices = list(np.random.choice(test_indices, MAX_TEST, replace=False))

print(f"  Train pairs: {len(train_indices)}, Test pairs: {len(test_indices)}")
print(f"  Train chains: {len(train_chains)}, Test chains: {len(test_chains)}")

# Build text prompts
def build_conditioned_text(p):
    return (
        f"Given the following reasoning state:\n"
        f"{p['state_text']}\n\n"
        f"Rate the quality of this next step:\n"
        f"{p['action_text']}"
    )

def build_blind_text(p):
    return (
        f"Rate the quality of this reasoning step:\n"
        f"{p['action_text']}"
    )

train_texts_blind = [build_blind_text(pairs[i]) for i in train_indices]
train_texts_conditioned = [build_conditioned_text(pairs[i]) for i in train_indices]
train_labels = torch.FloatTensor([pairs[i]['label'] for i in train_indices])
train_state_vectors = torch.FloatTensor(pair_state_vectors[train_indices])

test_texts_blind = [build_blind_text(pairs[i]) for i in test_indices]
test_texts_conditioned = [build_conditioned_text(pairs[i]) for i in test_indices]
test_labels = torch.FloatTensor([pairs[i]['label'] for i in test_indices])
test_state_vectors = torch.FloatTensor(pair_state_vectors[test_indices])

# Label balance
train_pos = train_labels.sum().item()
test_pos = test_labels.sum().item()
print(f"  Train label balance: {train_pos:.0f} pos / {len(train_labels) - train_pos:.0f} neg")
print(f"  Test label balance: {test_pos:.0f} pos / {len(test_labels) - test_pos:.0f} neg")

data = {
    'train_texts_blind': train_texts_blind,
    'train_texts_conditioned': train_texts_conditioned,
    'train_labels': train_labels,
    'train_state_vectors': train_state_vectors,
    'test_texts_blind': test_texts_blind,
    'test_texts_conditioned': test_texts_conditioned,
    'test_labels': test_labels,
    'test_state_vectors': test_state_vectors,
    'metadata': {
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'n_chains_train': len(train_chains),
        'n_chains_test': len(test_chains),
        'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
        'dataset': 'openai/gsm8k',
        'split': 'train',
        'min_steps': MIN_STEPS,
        'seed': SEED,
    },
}

out_path = os.path.join(OUT_DIR, 'e4_math_data.pt')
torch.save(data, out_path)
print(f"\nSaved to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
print("Done.")
