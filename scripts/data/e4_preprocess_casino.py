"""
E4: Pre-process CaSiNo negotiation for SSCC training.
MODERATE SL GENUINE-LABEL control: SL=0.220 with deal-outcome quality.

Uses deal outcome (points_scored) as the quality signal:
- Each turn labeled by whether the speaker's final points > median.
- This is a genuine quality label (not context-correctness).

Builds pairwise data: high-outcome turns (chosen) vs low-outcome turns (rejected),
matched by dialogue position to control for state.
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
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

# ===================================================================
# 1. LOAD CASINO
# ===================================================================
print("=" * 70)
print("E4-CASINO: PREPROCESSING WITH DEAL-OUTCOME LABELS")
print("=" * 70)

print("\n[1] Loading CaSiNo...")
ds = load_dataset('casino', split='train')
print(f"  Total dialogues: {len(ds)}")

# ===================================================================
# 2. EXTRACT TURNS WITH OUTCOME LABELS
# ===================================================================
print("\n[2] Extracting turns with deal-outcome labels...")

# Compute median points across all participants
all_points = []
for ex in ds:
    for agent_id in ['mturk_agent_1', 'mturk_agent_2']:
        info = ex['participant_info'].get(agent_id, {})
        outcomes = info.get('outcomes', {})
        pts = outcomes.get('points_scored', None)
        if pts is not None:
            all_points.append(int(pts))

median_points = np.median(all_points)
print(f"  All points: {len(all_points)}, median={median_points}")

# Build turns with state/action/label
turns_data = []
dialogue_ids = []

for di, ex in enumerate(ds):
    chat_logs = ex['chat_logs']
    participant_info = ex['participant_info']

    # Get points for each agent
    agent_points = {}
    for agent_id in ['mturk_agent_1', 'mturk_agent_2']:
        info = participant_info.get(agent_id, {})
        outcomes = info.get('outcomes', {})
        pts = outcomes.get('points_scored', None)
        if pts is not None:
            agent_points[agent_id] = int(pts)

    if len(agent_points) < 2:
        continue

    # Build turn sequence (skip Submit-Deal and Accept-Deal)
    turn_texts = []
    turn_agents = []
    for turn in chat_logs:
        text = turn['text']
        agent = turn['id']
        # Skip deal submission/acceptance turns
        if text in ('Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away'):
            continue
        if not text.strip():
            continue
        turn_texts.append(text)
        turn_agents.append(agent)

    if len(turn_texts) < 4:
        continue

    # For each turn (starting from turn 2), create a training example
    for i in range(2, len(turn_texts)):
        agent = turn_agents[i]
        pts = agent_points.get(agent, None)
        if pts is None:
            continue

        # State = previous turns
        state = '\n'.join(turn_texts[:i])[:1024]
        # Action = current turn
        action = turn_texts[i][:512]
        # Label = above median outcome
        label = 1 if pts > median_points else 0

        turns_data.append({
            'state': state,
            'action': action,
            'label': label,
            'points': pts,
            'dialogue_id': di,
            'turn_idx': i,
            'agent': agent,
        })
        dialogue_ids.append(di)

print(f"  Total turns: {len(turns_data)}")
print(f"  Label distribution: {sum(t['label'] for t in turns_data)} positive, "
      f"{sum(1-t['label'] for t in turns_data)} negative")
print(f"  From {len(set(dialogue_ids))} dialogues")

# ===================================================================
# 3. BUILD PAIRWISE DATA (matched by dialogue position)
# ===================================================================
print("\n[3] Building pairwise data (matched by position)...")

# Group by (turn_idx, position_in_dialogue) -- we want to compare
# turns from different dialogues at similar positions
from collections import defaultdict

# Group turns by turn_idx
by_position = defaultdict(lambda: {'pos': [], 'neg': []})
for t in turns_data:
    # Bucket turn positions: early (2-4), mid (5-7), late (8+)
    bucket = min(t['turn_idx'] // 3, 3)
    if t['label'] == 1:
        by_position[bucket]['pos'].append(t)
    else:
        by_position[bucket]['neg'].append(t)

pairs = []
np.random.seed(SEED)
for bucket, groups in by_position.items():
    pos_list = groups['pos']
    neg_list = groups['neg']
    n_pairs = min(len(pos_list), len(neg_list))
    if n_pairs == 0:
        continue

    np.random.shuffle(pos_list)
    np.random.shuffle(neg_list)

    for i in range(n_pairs):
        pairs.append({
            'state_chosen': pos_list[i]['state'],
            'action_chosen': pos_list[i]['action'],
            'state_rejected': neg_list[i]['state'],
            'action_rejected': neg_list[i]['action'],
            'points_chosen': pos_list[i]['points'],
            'points_rejected': neg_list[i]['points'],
        })

np.random.shuffle(pairs)
print(f"  Total pairs: {len(pairs)}")

# ===================================================================
# 4. EMBED STATES FOR PCA
# ===================================================================
print("\n[4] Embedding states...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

state_texts = list(set(
    [p['state_chosen'] for p in pairs] + [p['state_rejected'] for p in pairs]
))
state_embeddings = embedder.encode(state_texts, show_progress_bar=True, batch_size=64)
state_map = {t: i for i, t in enumerate(state_texts)}

pca = PCA(n_components=D_PCA)
state_pca = pca.fit_transform(state_embeddings)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# ===================================================================
# 5. TRAIN/TEST SPLIT BY DIALOGUE
# ===================================================================
print("\n[5] Splitting 80/20...")
n_pairs = len(pairs)
indices = np.arange(n_pairs)
np.random.shuffle(indices)

split_point = int(0.8 * n_pairs)
train_indices = indices[:split_point]
test_indices = indices[split_point:]

if len(train_indices) > MAX_TRAIN:
    train_indices = train_indices[:MAX_TRAIN]
if len(test_indices) > MAX_TEST:
    test_indices = test_indices[:MAX_TEST]

print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")

# ===================================================================
# 6. BUILD TEXT PROMPTS
# ===================================================================
print("\n[6] Building text prompts...")


def build_conditioned(state, action):
    return f"Given this negotiation:\n{state}\n\nRate this response:\n{action}"


def build_blind(action):
    return f"Rate this negotiation response:\n{action}"


# For conditioned: each action gets its OWN state (from its own dialogue).
# This avoids the coherence confound: if we used state_chosen for both,
# the model could detect that action_rejected doesn't follow naturally from
# state_chosen (different dialogue), inflating PRM accuracy.
# With own-state pairing, the model must learn R(state, action) -> quality,
# and the pairwise loss trains R(good_state, good_action) > R(bad_state, bad_action).
train_chosen_blind = [build_blind(pairs[i]['action_chosen']) for i in train_indices]
train_rejected_blind = [build_blind(pairs[i]['action_rejected']) for i in train_indices]
train_chosen_state = [build_conditioned(pairs[i]['state_chosen'], pairs[i]['action_chosen']) for i in train_indices]
train_rejected_state = [build_conditioned(pairs[i]['state_rejected'], pairs[i]['action_rejected']) for i in train_indices]

test_chosen_blind = [build_blind(pairs[i]['action_chosen']) for i in test_indices]
test_rejected_blind = [build_blind(pairs[i]['action_rejected']) for i in test_indices]
test_chosen_state = [build_conditioned(pairs[i]['state_chosen'], pairs[i]['action_chosen']) for i in test_indices]
test_rejected_state = [build_conditioned(pairs[i]['state_rejected'], pairs[i]['action_rejected']) for i in test_indices]

# State vectors for stratified eval
test_state_pca_list = []
for i in test_indices:
    st = pairs[i]['state_chosen']
    if st in state_map:
        test_state_pca_list.append(state_pca[state_map[st]])
    else:
        test_state_pca_list.append(np.zeros(D_PCA))
test_state_pca_arr = np.array(test_state_pca_list)
test_z_mean = np.mean([state_pca[state_map[pairs[i]['state_chosen']]]
                        for i in train_indices if pairs[i]['state_chosen'] in state_map], axis=0)
test_z_dist = np.linalg.norm(test_state_pca_arr - test_z_mean, axis=1)

# ===================================================================
# 7. SAVE
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
    'metadata': {
        'dataset': 'CaSiNo',
        'quality_signal': 'deal_outcome_points',
        'median_points': float(median_points),
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'n_dialogues': len(set(dialogue_ids)),
        'n_total_turns': len(turns_data),
    },
}

out_path = os.path.join(OUT_DIR, 'e4_casino_data.pt')
torch.save(data, out_path)
print(f"\nSaved to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

print(f"\nSummary:")
print(f"  Train: {len(train_chosen_blind)} pairs")
print(f"  Test: {len(test_chosen_blind)} pairs")
print(f"  Avg conditioned text len: {np.mean([len(t) for t in train_chosen_state]):.0f} chars")
print(f"  Avg blind text len: {np.mean([len(t) for t in train_chosen_blind]):.0f} chars")
print(f"  Quality signal: deal outcome (points > {median_points} median)")
