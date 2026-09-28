"""
R2: State-lift with REAL PRM800K step-correctness labels.

This is the gold standard -- human-annotated step ratings (+1/-1)
from OpenAI's PRM800K dataset. No context-correctness proxy.

Quality label = human rating of step correctness.
State = problem + previous steps.
Action = current step.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from scipy import stats
from sentence_transformers import SentenceTransformer

SEED = 42
D_PCA = 16
np.random.seed(SEED)

encoder = SentenceTransformer('all-MiniLM-L6-v2')
DATA_DIR = 'data/PRM800K'
OUT_DIR = 'results'

print("=" * 70)
print("R2: STATE-LIFT WITH PRM800K REAL STEP-CORRECTNESS LABELS")
print("=" * 70)

# Parse PRM800K
print("\n[1] Parsing PRM800K phase2_train...")
step_data = []
with open(os.path.join(DATA_DIR, 'phase2_train.jsonl')) as f:
    for line_i, line in enumerate(f):
        if line_i >= 10000:  # cap for speed
            break
        ex = json.loads(line)
        problem = ex['question']['problem']
        label_info = ex['label']
        steps = label_info.get('steps', [])

        chain_steps = []
        for step in steps:
            # Get the chosen completion
            chosen_idx = step.get('chosen_completion', 0)
            completions = step.get('completions', [])
            if not completions:
                continue

            if isinstance(chosen_idx, int) and chosen_idx < len(completions):
                comp = completions[chosen_idx]
            else:
                comp = completions[0]

            text = comp.get('text', '')
            rating = comp.get('rating', None)

            if text and rating is not None:
                chain_steps.append({'text': text, 'rating': int(rating)})

        if len(chain_steps) >= 2:
            step_data.append({'problem': problem, 'steps': chain_steps})

print(f"  {len(step_data)} chains with >= 2 rated steps")

# Count ratings
all_ratings = [s['rating'] for chain in step_data for s in chain['steps']]
print(f"  Total rated steps: {len(all_ratings)}")
print(f"  Rating distribution: +1={all_ratings.count(1)}, 0={all_ratings.count(0)}, -1={all_ratings.count(-1)}")

# Build transitions: state = problem + previous steps, action = current step
print("\n[2] Building transitions...")
states, actions, ratings = [], [], []
chain_ids = []

for ci, chain in enumerate(step_data):
    problem = chain['problem']
    for i, step in enumerate(chain['steps']):
        # State = problem + previous steps
        if i == 0:
            state = problem
        else:
            prev_texts = [s['text'] for s in chain['steps'][:i]]
            state = problem + "\n" + "\n".join(prev_texts)

        state = state[:1024]
        action = step['text'][:512]
        rating = step['rating']

        states.append(state)
        actions.append(action)
        ratings.append(rating)
        chain_ids.append(ci)

print(f"  {len(states)} step-level transitions")

# Embed
print("\n[3] Embedding...")
state_embs = encoder.encode(states, batch_size=256, show_progress_bar=True)
action_embs = encoder.encode(actions, batch_size=256, show_progress_bar=True)

# PCA
pca = PCA(n_components=D_PCA)
pca.fit(np.vstack([state_embs, action_embs]))
z = pca.transform(state_embs)
a = pca.transform(action_embs)
y = np.array(ratings, dtype=float)

print(f"  z shape: {z.shape}, a shape: {a.shape}")
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# ===================================================================
# Compute state-lift with REAL correctness labels
# ===================================================================
print("\n[4] Computing state-lift...")

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Action only
r2_action = cross_val_score(Ridge(alpha=1.0), a, y, cv=cv, scoring='r2').mean()

# State + action + interaction
X_sa = np.hstack([a, z, a * z])
r2_sa = cross_val_score(Ridge(alpha=1.0), X_sa, y, cv=cv, scoring='r2').mean()

# State only
r2_state = cross_val_score(Ridge(alpha=1.0), z, y, cv=cv, scoring='r2').mean()

# Outcome model (with extras)
a_c = a.mean(0)
X_out = np.hstack([a, np.linalg.norm(a, axis=1, keepdims=True),
                    np.linalg.norm(a - a_c, axis=1, keepdims=True)])
r2_outcome = cross_val_score(Ridge(alpha=1.0), X_out, y, cv=cv, scoring='r2').mean()

# Process model
X_proc = np.hstack([z, a, z * a])
r2_process = cross_val_score(Ridge(alpha=1.0), X_proc, y, cv=cv, scoring='r2').mean()

state_lift = r2_sa - max(r2_action, 0)
sdi = r2_process / max(r2_outcome, 0.001)

print(f"\n  RESULTS WITH PRM800K REAL LABELS:")
print(f"    R2 action only:    {r2_action:.4f}")
print(f"    R2 state only:     {r2_state:.4f}")
print(f"    R2 state+action:   {r2_sa:.4f}")
print(f"    R2 outcome model:  {r2_outcome:.4f}")
print(f"    R2 process model:  {r2_process:.4f}")
print(f"    State-lift:        {state_lift:.4f}")
print(f"    SDI:               {sdi:.2f}")

# ===================================================================
# Compare with context-correctness (our T8 approach)
# ===================================================================
print("\n[5] Comparison with context-correctness approach...")

# Build context-correctness pairs from the same data
tz_cc, ta_cc, labels_cc = [], [], []
np.random.seed(SEED)
for ci, chain in enumerate(step_data[:2000]):
    for i in range(len(chain['steps'])):
        if i == 0:
            state = chain['problem']
        else:
            prev = [s['text'] for s in chain['steps'][:i]]
            state = chain['problem'] + "\n" + "\n".join(prev)

        idx = ci * 100 + i  # approximate index into embeddings
        if idx >= len(state_embs):
            break

        # Correct context
        tz_cc.append(state_embs[idx])
        ta_cc.append(action_embs[idx])
        labels_cc.append(1)

        # Wrong context
        other = np.random.randint(0, len(state_embs))
        tz_cc.append(state_embs[other])
        ta_cc.append(action_embs[idx])
        labels_cc.append(0)

z_cc = np.array(tz_cc[:10000])
a_cc = np.array(ta_cc[:10000])
y_cc = np.array(labels_cc[:10000], dtype=float)

pca_cc = PCA(n_components=D_PCA)
pca_cc.fit(np.vstack([z_cc, a_cc]))
zp_cc = pca_cc.transform(z_cc)
ap_cc = pca_cc.transform(a_cc)

r2_act_cc = cross_val_score(Ridge(alpha=1.0), ap_cc, y_cc, cv=cv, scoring='r2').mean()
X_sa_cc = np.hstack([ap_cc, zp_cc, ap_cc * zp_cc])
r2_sa_cc = cross_val_score(Ridge(alpha=1.0), X_sa_cc, y_cc, cv=cv, scoring='r2').mean()
sl_cc = r2_sa_cc - max(r2_act_cc, 0)

print(f"    Context-correctness state-lift: {sl_cc:.4f}")
print(f"    Real-label state-lift:          {state_lift:.4f}")
print(f"    Ratio (real/context):           {state_lift / max(sl_cc, 0.001):.2f}x")

# ===================================================================
# Bootstrap CI for PRM800K state-lift
# ===================================================================
print("\n[6] Bootstrap CI for PRM800K state-lift...")
boot_sls = []
for boot_i in range(200):
    np.random.seed(SEED + boot_i)
    idx = np.random.choice(len(z), min(2000, len(z)), replace=True)
    sl_b, _, _ = (
        cross_val_score(Ridge(alpha=1.0), np.hstack([a[idx], z[idx], a[idx]*z[idx]]),
                        y[idx], cv=cv, scoring='r2').mean() -
        max(cross_val_score(Ridge(alpha=1.0), a[idx], y[idx], cv=cv, scoring='r2').mean(), 0),
        None, None
    )
    boot_sls.append(sl_b)

# Simpler bootstrap
boot_sls = []
for boot_i in range(200):
    np.random.seed(SEED + boot_i)
    idx = np.random.choice(len(z), min(2000, len(z)), replace=True)
    cv_b = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2_a = cross_val_score(Ridge(alpha=1.0), a[idx], y[idx], cv=cv_b, scoring='r2').mean()
    X_b = np.hstack([a[idx], z[idx], a[idx]*z[idx]])
    r2_sa = cross_val_score(Ridge(alpha=1.0), X_b, y[idx], cv=cv_b, scoring='r2').mean()
    boot_sls.append(r2_sa - max(r2_a, 0))

boot_sls = np.array(boot_sls)
ci_lo, ci_hi = np.percentile(boot_sls, [2.5, 97.5])
print(f"    PRM800K state-lift: {boot_sls.mean():.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

# ===================================================================
# Also: binary split (positive vs negative ratings only)
# ===================================================================
print("\n[7] Binary analysis (positive=1 vs negative=-1, excluding neutral)...")
mask_binary = np.array(ratings) != 0
z_bin = z[mask_binary]
a_bin = a[mask_binary]
y_bin = (np.array(ratings)[mask_binary] > 0).astype(float)

print(f"  Binary samples: {len(z_bin)} ({y_bin.sum():.0f} positive, {(1-y_bin).sum():.0f} negative)")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

auc_action = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                              a_bin, y_bin, cv=cv_s, scoring='roc_auc').mean()
X_sa_bin = np.hstack([a_bin, z_bin, a_bin * z_bin])
auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                          X_sa_bin, y_bin, cv=cv_s, scoring='roc_auc').mean()

sl_binary = auc_sa - auc_action

print(f"  AUC action only:  {auc_action:.4f}")
print(f"  AUC state+action: {auc_sa:.4f}")
print(f"  State-lift (AUC): {sl_binary:.4f}")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("PRM800K STATE-LIFT SUMMARY")
print("=" * 70)

print(f"""
  PRM800K with REAL step-correctness labels:
    State-lift (R2 diff):  {state_lift:.4f}
    State-lift (AUC diff): {sl_binary:.4f}
    Bootstrap 95% CI:      [{ci_lo:.4f}, {ci_hi:.4f}]

  Context-correctness (our T8 approach):
    State-lift:            {sl_cc:.4f}

  COMPARISON:
    Real labels give {'HIGHER' if state_lift > sl_cc else 'LOWER'} state-lift than context-correctness.
    {'This confirms context-correctness is an UPPER BOUND.' if sl_cc > state_lift else 'Real step quality IS genuinely state-dependent.'}

  INTERPRETATION:
    Math reasoning with real step-correctness labels has state-lift = {state_lift:.4f}.
    {'This is HIGH -- confirming PRM should help for math.' if state_lift > 0.05 else 'This is modest -- PRM benefit may be smaller than context-correctness suggests.'}
""")

# Save
out = {
    'prm800k_real_labels': {
        'n_chains': len(step_data),
        'n_steps': len(states),
        'rating_distribution': {'+1': all_ratings.count(1), '0': all_ratings.count(0), '-1': all_ratings.count(-1)},
        'r2_action': float(r2_action),
        'r2_state': float(r2_state),
        'r2_state_action': float(r2_sa),
        'r2_outcome': float(r2_outcome),
        'r2_process': float(r2_process),
        'state_lift': float(state_lift),
        'sdi': float(sdi),
        'bootstrap_ci': [float(ci_lo), float(ci_hi)],
    },
    'binary_analysis': {
        'n': len(z_bin),
        'auc_action': float(auc_action),
        'auc_state_action': float(auc_sa),
        'state_lift_auc': float(sl_binary),
    },
    'context_correctness_comparison': {
        'sl_context_correctness': float(sl_cc),
        'sl_real_labels': float(state_lift),
        'ratio': float(state_lift / max(sl_cc, 0.001)),
    },
}

out_path = os.path.join(OUT_DIR, 'r2_prm800k_statelift.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved to {out_path}")
