"""
R5: Preprocess PRM800K for STEP-LEVEL reward model training.

Key difference from R3b: step-level labels (not solution-level).
Uses all PRM800K rated steps with balanced sampling.
"""

import torch
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datasets import load_dataset

SEED = 42
np.random.seed(SEED)
OUT_PATH = 'data/r5_step_data.pt'
PRM_PATH = 'data/PRM800K'

print("=" * 70)
print("R5: STEP-LEVEL PRM800K PREPROCESSING")
print("=" * 70)

# Parse ALL PRM800K -- step-level labels
print("\n[1] Parsing PRM800K step-level labels...")

steps_all = []  # (problem, state_text, step_text, rating)

with open(os.path.join(PRM_PATH, 'phase2_train.jsonl')) as f:
    for line_i, line in enumerate(f):
        if line_i % 10000 == 0 and line_i > 0:
            print(f"  Parsed {line_i} lines...")
        ex = json.loads(line)
        problem = ex['question']['problem']
        steps_info = ex['label'].get('steps', [])

        prev_texts = []
        for step in steps_info:
            chosen_idx = step.get('chosen_completion', 0)
            completions = step.get('completions', [])
            if not completions:
                continue
            comp = completions[min(chosen_idx, len(completions)-1)] if isinstance(chosen_idx, int) else completions[0]
            text = comp.get('text', '')
            rating = comp.get('rating', None)

            if text and rating is not None and int(rating) != 0:
                # Build state = problem + previous steps
                if prev_texts:
                    state = problem + "\n" + "\n".join(prev_texts)
                else:
                    state = problem

                steps_all.append({
                    'problem': problem[:512],
                    'state': state[:1024],
                    'step': text[:512],
                    'label': 1 if int(rating) > 0 else 0,
                })

            prev_texts.append(text)

n_pos = sum(1 for s in steps_all if s['label'] == 1)
n_neg = sum(1 for s in steps_all if s['label'] == 0)
print(f"  Total rated steps: {len(steps_all)} ({n_pos} positive, {n_neg} negative)")

# Balance
min_class = min(n_pos, n_neg)
pos = [s for s in steps_all if s['label'] == 1]
neg = [s for s in steps_all if s['label'] == 0]
np.random.shuffle(pos)
np.random.shuffle(neg)

# Use all negatives + equal positives, cap total
MAX_TOTAL = 12000
n_per_class = min(min_class, MAX_TOTAL // 2)
balanced = pos[:n_per_class] + neg[:n_per_class]
np.random.shuffle(balanced)

n_train = int(0.85 * len(balanced))
train_data = balanced[:n_train]
test_data = balanced[n_train:]

print(f"  Balanced: {len(balanced)} ({n_per_class} per class)")
print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

# Build prompts -- SAME format for training and step-level scoring
train_conditioned = [
    f"Problem: {s['problem']}\n\nPrevious reasoning:\n{s['state']}\n\nNext step: {s['step']}\n\nIs this step correct?"
    for s in train_data
]
train_blind = [
    f"Reasoning step: {s['step']}\n\nIs this step correct?"
    for s in train_data
]
train_labels = [s['label'] for s in train_data]

test_conditioned = [
    f"Problem: {s['problem']}\n\nPrevious reasoning:\n{s['state']}\n\nNext step: {s['step']}\n\nIs this step correct?"
    for s in test_data
]
test_blind = [
    f"Reasoning step: {s['step']}\n\nIs this step correct?"
    for s in test_data
]
test_labels = [s['label'] for s in test_data]

# GSM8K test problems
print("\n[2] Preparing GSM8K test problems...")
ds_test = load_dataset('openai/gsm8k', 'main', split='test')
gsm8k_problems = []
for ex in list(ds_test)[:200]:
    answer_text = ex.get('answer', '')
    final_answer = None
    for line in reversed(answer_text.split('\n')):
        if '####' in line:
            final_answer = line.split('####')[-1].strip().replace(',', '')
            break
    if final_answer:
        gsm8k_problems.append({
            'question': ex['question'],
            'answer': answer_text,
            'final_answer': final_answer,
            'steps': [s.strip() for s in answer_text.split('\n') if s.strip()],
        })

print(f"  {len(gsm8k_problems)} problems")

# Embed for PCA
print("\n[3] Embedding...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
state_texts = [s['state'] for s in train_data + test_data]
embs = encoder.encode(state_texts[:5000], batch_size=256, show_progress_bar=True)
pca = PCA(n_components=16)
pca.fit(embs)
state_vectors = pca.transform(encoder.encode([s['state'] for s in train_data + test_data][:len(train_data)+len(test_data)],
                                              batch_size=256, show_progress_bar=False))
train_vectors = state_vectors[:len(train_data)]
test_vectors = state_vectors[len(train_data):len(train_data)+len(test_data)]

# Save
print("\n[4] Saving...")
data = {
    'train_texts_conditioned': train_conditioned,
    'train_texts_blind': train_blind,
    'train_labels': torch.tensor(train_labels, dtype=torch.long),
    'train_state_vectors': torch.tensor(train_vectors, dtype=torch.float32),
    'test_texts_conditioned': test_conditioned,
    'test_texts_blind': test_blind,
    'test_labels': torch.tensor(test_labels, dtype=torch.long),
    'test_state_vectors': torch.tensor(test_vectors, dtype=torch.float32),
    'gsm8k_problems': gsm8k_problems,
    'metadata': {
        'n_train': len(train_data),
        'n_test': len(test_data),
        'n_gsm8k': len(gsm8k_problems),
        'n_pos_total': n_pos,
        'n_neg_total': n_neg,
        'n_per_class': n_per_class,
        'dataset': 'PRM800K step-level + GSM8K',
        'quality_signal': 'PRM800K human step-correctness ratings',
        'label_format': 'step-level (positive=correct, negative=incorrect)',
    },
}

torch.save(data, OUT_PATH)
print(f"Saved to {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
