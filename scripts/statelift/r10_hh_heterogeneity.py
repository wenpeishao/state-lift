"""
R10: Per-quartile state-lift for HH-RLHF.

Addresses Claude review Q3: Table 7 shows Q4 has +7.5% PRM advantage
at global SL≈0. If per-quartile SL is high in Q4 and zero elsewhere,
the diagnostic is correct *locally* -- turns a weakness into a strength.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from scipy import stats

SEED = 42
D_PCA = 16
np.random.seed(SEED)
OUT_DIR = 'results'

print("=" * 70)
print("R10: PER-QUARTILE STATE-LIFT FOR HH-RLHF")
print("=" * 70)

# ===================================================================
# 1. LOAD HH-RLHF
# ===================================================================
print("\n[1] Loading HH-RLHF...")
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ds = load_dataset("Anthropic/hh-rlhf", split="train")
print(f"  Total: {len(ds)}")

def parse_conversation(text):
    turns = []
    parts = text.split('\n\nHuman: ')
    for i, part in enumerate(parts):
        if i == 0 and not part.strip():
            continue
        sub = part.split('\n\nAssistant: ')
        if len(sub) >= 1 and sub[0].strip():
            human_text = sub[0].replace('Human: ', '').strip()
            if human_text:
                turns.append(('human', human_text))
        if len(sub) >= 2 and sub[1].strip():
            turns.append(('assistant', sub[1].strip()))
    return turns

# Extract examples with shared prefix
examples = []
for i, row in enumerate(ds):
    chosen = parse_conversation(row['chosen'])
    rejected = parse_conversation(row['rejected'])
    if len(chosen) < 3 or len(rejected) < 3:
        continue
    prefix_len = 0
    for j in range(min(len(chosen), len(rejected))):
        if chosen[j] == rejected[j]:
            prefix_len = j + 1
        else:
            break
    if prefix_len < 2:
        continue
    examples.append({
        'chosen': chosen,
        'rejected': rejected,
        'prefix': chosen[:prefix_len],
        'prefix_len': prefix_len,
    })

print(f"  Multi-turn examples: {len(examples)}")

# ===================================================================
# 2. EMBED
# ===================================================================
print("\n[2] Embedding...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Collect unique texts
all_texts_set = set()
for ex in examples:
    for role, text in ex['chosen']:
        all_texts_set.add(text)
    for role, text in ex['rejected']:
        all_texts_set.add(text)
all_texts = list(all_texts_set)
print(f"  Unique texts: {len(all_texts)}")

# Check for cache
cache_path = 'data/hh_rlhf_embeddings.npz'
if os.path.exists(cache_path):
    print("  Loading cached embeddings...")
    cached = np.load(cache_path, allow_pickle=True)
    all_embeddings = cached['embeddings']
    embed_map = cached['embed_map'].item()
    print(f"  Cached: {len(all_embeddings)} embeddings")
else:
    print("  Encoding from scratch...")
    all_embeddings = encoder.encode(all_texts, batch_size=512, show_progress_bar=True)
    embed_map = {text: idx for idx, text in enumerate(all_texts)}

# ===================================================================
# 3. BUILD STATE + ACTION + LABEL
# ===================================================================
print("\n[3] Building state/action/label triples...")

pca = PCA(n_components=D_PCA)

# Collect prefix embeddings for PCA
prefix_embs = []
for ex in examples:
    for role, text in ex['prefix']:
        if text in embed_map:
            prefix_embs.append(all_embeddings[embed_map[text]])
prefix_embs = np.array(prefix_embs)
pca.fit(prefix_embs)
print(f"  PCA variance: {pca.explained_variance_ratio_.sum():.3f}")

# For each example: state = mean of prefix embeddings, action = response text
# Label = 1 for chosen, 0 for rejected
states_z, actions_a, labels = [], [], []
state_norms = []  # for quartile splitting

for ex in examples:
    # Compute state
    turn_embs = []
    for role, text in ex['prefix']:
        if text in embed_map:
            turn_embs.append(all_embeddings[embed_map[text]])
    if not turn_embs:
        continue
    state_raw = np.mean(turn_embs, axis=0)
    state_pca = pca.transform(state_raw.reshape(1, -1))[0]

    # Chosen response
    chosen_texts = [t for _, t in ex['chosen'][ex['prefix_len']:]]
    if chosen_texts:
        chosen_text = ' '.join(chosen_texts)[:512]
        if chosen_text in embed_map:
            action_raw = all_embeddings[embed_map[chosen_text]]
        else:
            action_raw = encoder.encode([chosen_text])[0]
        action_pca = pca.transform(action_raw.reshape(1, -1))[0]

        states_z.append(state_pca)
        actions_a.append(action_pca)
        labels.append(1)
        state_norms.append(np.linalg.norm(state_pca))

    # Rejected response
    rejected_texts = [t for _, t in ex['rejected'][ex['prefix_len']:]]
    if rejected_texts:
        rejected_text = ' '.join(rejected_texts)[:512]
        if rejected_text in embed_map:
            action_raw = all_embeddings[embed_map[rejected_text]]
        else:
            action_raw = encoder.encode([rejected_text])[0]
        action_pca = pca.transform(action_raw.reshape(1, -1))[0]

        states_z.append(state_pca)
        actions_a.append(action_pca)
        labels.append(0)
        state_norms.append(np.linalg.norm(state_pca))

z = np.array(states_z)
a = np.array(actions_a)
y = np.array(labels, dtype=float)
state_norms = np.array(state_norms)

print(f"  Total samples: {len(z)} ({y.sum():.0f} chosen, {(1-y).sum():.0f} rejected)")

# ===================================================================
# 4. GLOBAL STATE-LIFT (baseline)
# ===================================================================
print("\n[4] Global state-lift...")
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

r2_act_global = cross_val_score(Ridge(alpha=1.0), a, y, cv=cv, scoring='r2').mean()
X_sa = np.hstack([a, z, a * z])
r2_sa_global = cross_val_score(Ridge(alpha=1.0), X_sa, y, cv=cv, scoring='r2').mean()
sl_global = r2_sa_global - max(r2_act_global, 0)

print(f"  R2 action:    {r2_act_global:.4f}")
print(f"  R2 s+a+i:     {r2_sa_global:.4f}")
print(f"  SL global:    {sl_global:.4f}")

# Also AUC-based
cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
auc_act = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                           a, y, cv=cv_s, scoring='roc_auc').mean()
auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                          X_sa, y, cv=cv_s, scoring='roc_auc').mean()
sl_auc_global = auc_sa - auc_act

print(f"  AUC action:   {auc_act:.4f}")
print(f"  AUC s+a+i:    {auc_sa:.4f}")
print(f"  SL_auc global:{sl_auc_global:.4f}")

# ===================================================================
# 5. PER-QUARTILE STATE-LIFT
# ===================================================================
print("\n[5] Per-quartile state-lift...")

# Split by state PC1 (the dominant state axis)
state_pc1 = z[:, 0]
quartile_edges = np.percentile(state_pc1, [25, 50, 75])
quartile_labels = np.digitize(state_pc1, quartile_edges)  # 0, 1, 2, 3

quartile_results = {}
for q in range(4):
    mask = quartile_labels == q
    n_q = mask.sum()
    if n_q < 100:
        print(f"  Q{q+1}: too few samples ({n_q}), skipping")
        continue

    z_q, a_q, y_q = z[mask], a[mask], y[mask]
    X_sa_q = np.hstack([a_q, z_q, a_q * z_q])

    cv_q = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2_act_q = cross_val_score(Ridge(alpha=1.0), a_q, y_q, cv=cv_q, scoring='r2').mean()
    r2_sa_q = cross_val_score(Ridge(alpha=1.0), X_sa_q, y_q, cv=cv_q, scoring='r2').mean()
    sl_q = r2_sa_q - max(r2_act_q, 0)

    # AUC within quartile
    try:
        cv_sq = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        auc_act_q = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                     a_q, y_q, cv=cv_sq, scoring='roc_auc').mean()
        auc_sa_q = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                    X_sa_q, y_q, cv=cv_sq, scoring='roc_auc').mean()
        sl_auc_q = auc_sa_q - auc_act_q
    except:
        auc_act_q = auc_sa_q = sl_auc_q = float('nan')

    quartile_results[f'Q{q+1}'] = {
        'n': int(n_q),
        'label_mean': float(y_q.mean()),
        'state_pc1_range': [float(z_q[:, 0].min()), float(z_q[:, 0].max())],
        'r2_action': float(r2_act_q),
        'r2_state_action': float(r2_sa_q),
        'sl_r2': float(sl_q),
        'auc_action': float(auc_act_q),
        'auc_state_action': float(auc_sa_q),
        'sl_auc': float(sl_auc_q),
    }

    print(f"  Q{q+1} (n={n_q}): SL_r2={sl_q:.4f}, SL_auc={sl_auc_q:.4f}, "
          f"label_rate={y_q.mean():.3f}")

# ===================================================================
# 6. ALSO SPLIT BY STATE NORM (alternative to PC1)
# ===================================================================
print("\n[6] Per-quartile by state norm...")
norm_edges = np.percentile(state_norms, [25, 50, 75])
norm_quartiles = np.digitize(state_norms, norm_edges)

norm_results = {}
for q in range(4):
    mask = norm_quartiles == q
    n_q = mask.sum()
    if n_q < 100:
        continue

    z_q, a_q, y_q = z[mask], a[mask], y[mask]
    X_sa_q = np.hstack([a_q, z_q, a_q * z_q])

    cv_q = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2_act_q = cross_val_score(Ridge(alpha=1.0), a_q, y_q, cv=cv_q, scoring='r2').mean()
    r2_sa_q = cross_val_score(Ridge(alpha=1.0), X_sa_q, y_q, cv=cv_q, scoring='r2').mean()
    sl_q = r2_sa_q - max(r2_act_q, 0)

    norm_results[f'Q{q+1}'] = {
        'n': int(n_q),
        'norm_range': [float(state_norms[mask].min()), float(state_norms[mask].max())],
        'sl_r2': float(sl_q),
    }
    print(f"  Q{q+1} (n={n_q}): SL={sl_q:.4f}, norm_range=[{state_norms[mask].min():.2f}, {state_norms[mask].max():.2f}]")

# ===================================================================
# 7. SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("SUMMARY: HH-RLHF STATE-LIFT HETEROGENEITY")
print("=" * 70)

print(f"\n  Global SL (R2):  {sl_global:.4f}")
print(f"  Global SL (AUC): {sl_auc_global:.4f}")
print(f"\n  Per-quartile (by state PC1):")
for qname, qres in quartile_results.items():
    print(f"    {qname}: SL_r2={qres['sl_r2']:.4f}, SL_auc={qres['sl_auc']:.4f}, n={qres['n']}")

max_q = max(quartile_results.items(), key=lambda x: x[1]['sl_r2'])
min_q = min(quartile_results.items(), key=lambda x: x[1]['sl_r2'])

print(f"\n  Highest SL quartile: {max_q[0]} (SL={max_q[1]['sl_r2']:.4f})")
print(f"  Lowest SL quartile:  {min_q[0]} (SL={min_q[1]['sl_r2']:.4f})")
print(f"  Range: {max_q[1]['sl_r2'] - min_q[1]['sl_r2']:.4f}")

heterogeneous = max_q[1]['sl_r2'] > 3 * abs(sl_global) if sl_global != 0 else max_q[1]['sl_r2'] > 0.01
print(f"\n  Heterogeneous? {'YES -- SL varies across state space' if heterogeneous else 'NO -- uniformly low'}")
if heterogeneous:
    print(f"  INTERPRETATION: Global SL≈0 masks localized state-dependence.")
    print(f"  The diagnostic is correct per-region; aggregation washes out the signal.")

# Save
out = {
    'global': {
        'n': len(z),
        'sl_r2': float(sl_global),
        'sl_auc': float(sl_auc_global),
        'r2_action': float(r2_act_global),
        'r2_state_action': float(r2_sa_global),
        'auc_action': float(auc_act),
        'auc_state_action': float(auc_sa),
    },
    'per_quartile_pc1': quartile_results,
    'per_quartile_norm': norm_results,
    'heterogeneity': {
        'max_quartile': max_q[0],
        'max_sl': float(max_q[1]['sl_r2']),
        'min_quartile': min_q[0],
        'min_sl': float(min_q[1]['sl_r2']),
        'range': float(max_q[1]['sl_r2'] - min_q[1]['sl_r2']),
        'is_heterogeneous': bool(heterogeneous),
    },
}

out_path = os.path.join(OUT_DIR, 'r10_hh_heterogeneity.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
