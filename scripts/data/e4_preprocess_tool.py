"""
E4: Preprocess Glaive function-calling for PRM vs ORM training.
Quality signal: action in CORRECT context (label=1) vs WRONG context (label=0).
"""

import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datasets import load_dataset

SEED = 42
np.random.seed(SEED)
OUT_PATH = 'data/e4_tool_data.pt'

print("[1] Loading Glaive function-calling v2...")
ds = load_dataset('glaiveai/glaive-function-calling-v2', split='train')
print(f"  {len(ds)} examples")

# Parse conversations
convs = []
for ex in list(ds)[:5000]:
    chat = ex.get('chat', '')
    if not isinstance(chat, str) or len(chat) < 100:
        continue
    conv = []
    for line in chat.split('\n'):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        for prefix in ['USER:', 'ASSISTANT:', 'FUNCTION RESPONSE:', 'SYSTEM:']:
            if line.upper().startswith(prefix):
                text = line[len(prefix):].strip()
                if text:
                    conv.append((prefix[:-1].lower(), text[:512]))
                break
    if len(conv) >= 4:
        convs.append(conv)

print(f"  {len(convs)} conversations with >= 4 turns")

# Build pairs: correct context (label=1) vs wrong context (label=0)
print("[2] Building context-correctness pairs...")
pairs = []
np.random.seed(SEED)

for ci, conv in enumerate(convs):
    for i in range(1, len(conv) - 1):
        # State = previous turns
        state_parts = [f"{role}: {text}" for role, text in conv[:i]]
        state = "\n".join(state_parts)[-1024:]  # last 1024 chars
        action = conv[i][1]

        # Correct context
        pairs.append({'state': state, 'action': action, 'label': 1, 'conv_idx': ci})

        # Wrong context: same action, random state from different conversation
        other_ci = np.random.randint(0, len(convs))
        while other_ci == ci:
            other_ci = np.random.randint(0, len(convs))
        other_conv = convs[other_ci]
        other_i = min(i, len(other_conv) - 1)
        wrong_state_parts = [f"{role}: {text}" for role, text in other_conv[:max(1, other_i)]]
        wrong_state = "\n".join(wrong_state_parts)[-1024:]

        pairs.append({'state': wrong_state, 'action': action, 'label': 0, 'conv_idx': ci})

    if len(pairs) >= 20000:
        break

print(f"  {len(pairs)} pairs ({sum(p['label'] for p in pairs)} correct, {sum(1-p['label'] for p in pairs)} wrong)")

# Build prompts
print("[3] Building prompts...")
conditioned_texts = []
blind_texts = []
labels = []
conv_indices = []

for p in pairs:
    conditioned_texts.append(
        f"Given this conversation context:\n{p['state']}\n\n"
        f"Rate the quality of this next response:\n{p['action']}"
    )
    blind_texts.append(
        f"Rate the quality of this response:\n{p['action']}"
    )
    labels.append(p['label'])
    conv_indices.append(p['conv_idx'])

# Embed states
print("[4] Embedding states...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
state_texts = [p['state'] for p in pairs]

# Sample for embedding (full set may be too large)
emb_sample_idx = np.random.choice(len(state_texts), min(len(state_texts), 15000), replace=False)
sample_texts = [state_texts[i] for i in emb_sample_idx]
sample_embs = encoder.encode(sample_texts, batch_size=256, show_progress_bar=True)

# For non-sampled, use nearest sampled
all_state_embs = encoder.encode(state_texts[:len(pairs)], batch_size=256, show_progress_bar=True)

pca = PCA(n_components=16)
state_vectors = pca.fit_transform(all_state_embs)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# Split by conversation (80/20)
print("[5] Splitting...")
unique_convs = list(set(conv_indices))
np.random.shuffle(unique_convs)
n_train_convs = int(0.8 * len(unique_convs))
train_conv_set = set(unique_convs[:n_train_convs])

train_idx = [i for i, ci in enumerate(conv_indices) if ci in train_conv_set]
test_idx = [i for i, ci in enumerate(conv_indices) if ci not in train_conv_set]

MAX_TRAIN, MAX_TEST = 5000, 1500
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)
train_idx = train_idx[:MAX_TRAIN]
test_idx = test_idx[:MAX_TEST]

print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
print(f"  Train labels: {sum(labels[i] for i in train_idx)} pos / {sum(1-labels[i] for i in train_idx)} neg")
print(f"  Test labels: {sum(labels[i] for i in test_idx)} pos / {sum(1-labels[i] for i in test_idx)} neg")

# Save
import os
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
        'dataset': 'glaiveai/glaive-function-calling-v2',
        'quality_signal': 'action_in_correct_context_vs_wrong',
        'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
    },
}
torch.save(data, OUT_PATH)
print(f"\nSaved to {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
