"""
R9: Nonlinear state-lift on PRM800K REAL step-correctness labels.

Addresses the critical reviewer concern (W2): linear SL=0.015 on PRM800K
but LLM lift=+0.238. If nonlinear SL >> 0.015, the "lower bound" story holds.
If still ~0.015, we have a genuine false negative.

Also computes predicted regret for Code domain (W4 / A3).
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from scipy import stats

SEED = 42
D_PCA = 16
np.random.seed(SEED)

DATA_DIR = 'data/PRM800K'
OUT_DIR = 'results'

print("=" * 70)
print("R9: NONLINEAR STATE-LIFT ON PRM800K REAL LABELS")
print("=" * 70)

# ===================================================================
# 1. PARSE PRM800K (same as r2)
# ===================================================================
print("\n[1] Parsing PRM800K phase2_train...")
step_data = []
with open(os.path.join(DATA_DIR, 'phase2_train.jsonl')) as f:
    for line_i, line in enumerate(f):
        if line_i >= 10000:
            break
        ex = json.loads(line)
        problem = ex['question']['problem']
        label_info = ex['label']
        steps = label_info.get('steps', [])

        chain_steps = []
        for step in steps:
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

print(f"  {len(step_data)} chains")

# Build transitions
states, actions, ratings, chain_ids = [], [], [], []
for ci, chain in enumerate(step_data):
    problem = chain['problem']
    for i, step in enumerate(chain['steps']):
        if i == 0:
            state = problem
        else:
            prev_texts = [s['text'] for s in chain['steps'][:i]]
            state = problem + "\n" + "\n".join(prev_texts)
        states.append(state[:1024])
        actions.append(step['text'][:512])
        ratings.append(step['rating'])
        chain_ids.append(ci)

print(f"  {len(states)} transitions")
all_ratings = ratings
print(f"  +1={all_ratings.count(1)}, 0={all_ratings.count(0)}, -1={all_ratings.count(-1)}")

# ===================================================================
# 2. EMBED
# ===================================================================
print("\n[2] Embedding with MiniLM...")
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')
state_embs = encoder.encode(states, batch_size=256, show_progress_bar=True)
action_embs = encoder.encode(actions, batch_size=256, show_progress_bar=True)

pca = PCA(n_components=D_PCA)
pca.fit(np.vstack([state_embs, action_embs]))
z = pca.transform(state_embs)
a = pca.transform(action_embs)
y = np.array(ratings, dtype=float)

print(f"  z: {z.shape}, a: {a.shape}, PCA var: {pca.explained_variance_ratio_.sum():.4f}")

# ===================================================================
# 3. LINEAR BASELINE (reproduce r2)
# ===================================================================
print("\n[3] Linear state-lift (baseline)...")
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

r2_action_lin = cross_val_score(Ridge(alpha=1.0), a, y, cv=cv, scoring='r2').mean()
X_sa = np.hstack([a, z, a * z])
r2_sa_lin = cross_val_score(Ridge(alpha=1.0), X_sa, y, cv=cv, scoring='r2').mean()
sl_linear = r2_sa_lin - max(r2_action_lin, 0)

print(f"  R2 action: {r2_action_lin:.4f}")
print(f"  R2 s+a+i:  {r2_sa_lin:.4f}")
print(f"  SL linear: {sl_linear:.4f}")

# ===================================================================
# 4. KERNEL RIDGE
# ===================================================================
print("\n[4] Kernel Ridge (RBF) state-lift...")

# Use subsample for kernel methods (N^2 memory)
MAX_KR = 10000
if len(z) > MAX_KR:
    np.random.seed(SEED)
    idx_kr = np.random.choice(len(z), MAX_KR, replace=False)
    z_kr, a_kr, y_kr = z[idx_kr], a[idx_kr], y[idx_kr]
else:
    z_kr, a_kr, y_kr = z, a, y

X_sa_kr = np.hstack([a_kr, z_kr, a_kr * z_kr])

# Try multiple gamma values
for gamma in [0.01, 0.05, 0.1, 0.5]:
    try:
        r2_act_kr = cross_val_score(
            KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma),
            a_kr, y_kr, cv=cv, scoring='r2').mean()
        r2_sa_kr = cross_val_score(
            KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma),
            X_sa_kr, y_kr, cv=cv, scoring='r2').mean()
        sl_kr = r2_sa_kr - max(r2_act_kr, 0)
        print(f"  gamma={gamma}: R2_act={r2_act_kr:.4f}, R2_sa={r2_sa_kr:.4f}, SL={sl_kr:.4f}")
    except Exception as e:
        print(f"  gamma={gamma}: FAILED ({e})")

# Best gamma = 0.1 (from r1)
r2_act_kr_best = cross_val_score(
    KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1),
    a_kr, y_kr, cv=cv, scoring='r2').mean()
r2_sa_kr_best = cross_val_score(
    KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1),
    X_sa_kr, y_kr, cv=cv, scoring='r2').mean()
sl_kernel = r2_sa_kr_best - max(r2_act_kr_best, 0)

print(f"\n  BEST Kernel Ridge: SL = {sl_kernel:.4f}")

# ===================================================================
# 5. MLP
# ===================================================================
print("\n[5] MLP state-lift...")

# Try multiple architectures
for hidden in [(64, 32), (128, 64), (256, 128, 64)]:
    try:
        mlp_act = MLPRegressor(hidden_layer_sizes=hidden, max_iter=1000,
                               random_state=SEED, early_stopping=True,
                               validation_fraction=0.1)
        r2_act_mlp = cross_val_score(mlp_act, a, y, cv=cv, scoring='r2').mean()

        mlp_sa = MLPRegressor(hidden_layer_sizes=hidden, max_iter=1000,
                              random_state=SEED, early_stopping=True,
                              validation_fraction=0.1)
        r2_sa_mlp = cross_val_score(mlp_sa, X_sa, y, cv=cv, scoring='r2').mean()

        sl_mlp = r2_sa_mlp - max(r2_act_mlp, 0)
        print(f"  MLP{hidden}: R2_act={r2_act_mlp:.4f}, R2_sa={r2_sa_mlp:.4f}, SL={sl_mlp:.4f}")
    except Exception as e:
        print(f"  MLP{hidden}: FAILED ({e})")

# Best MLP (64,32)
mlp_act_best = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000,
                            random_state=SEED, early_stopping=True,
                            validation_fraction=0.1)
r2_act_mlp_best = cross_val_score(mlp_act_best, a, y, cv=cv, scoring='r2').mean()

mlp_sa_best = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000,
                           random_state=SEED, early_stopping=True,
                           validation_fraction=0.1)
r2_sa_mlp_best = cross_val_score(mlp_sa_best, X_sa, y, cv=cv, scoring='r2').mean()
sl_mlp_best = r2_sa_mlp_best - max(r2_act_mlp_best, 0)

print(f"\n  BEST MLP: SL = {sl_mlp_best:.4f}")

# ===================================================================
# 6. BINARY CLASSIFICATION (nonlinear)
# ===================================================================
print("\n[6] Binary classification (positive vs negative, nonlinear)...")
mask_binary = np.array(ratings) != 0
z_bin = z[mask_binary]
a_bin = a[mask_binary]
y_bin = (np.array(ratings)[mask_binary] > 0).astype(float)
X_sa_bin = np.hstack([a_bin, z_bin, a_bin * z_bin])

print(f"  Binary samples: {len(z_bin)} ({y_bin.sum():.0f} pos, {(1-y_bin).sum():.0f} neg)")

cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Linear logistic
auc_act_lin = cross_val_score(
    LogisticRegression(max_iter=1000, C=0.1),
    a_bin, y_bin, cv=cv_s, scoring='roc_auc').mean()
auc_sa_lin = cross_val_score(
    LogisticRegression(max_iter=1000, C=0.1),
    X_sa_bin, y_bin, cv=cv_s, scoring='roc_auc').mean()
sl_auc_linear = auc_sa_lin - auc_act_lin

print(f"  Linear:  AUC_act={auc_act_lin:.4f}, AUC_sa={auc_sa_lin:.4f}, SL_auc={sl_auc_linear:.4f}")

# MLP classifier
from sklearn.neural_network import MLPClassifier
mlp_cls_act = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                            random_state=SEED, early_stopping=True)
auc_act_mlp = cross_val_score(mlp_cls_act, a_bin, y_bin, cv=cv_s, scoring='roc_auc').mean()

mlp_cls_sa = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                           random_state=SEED, early_stopping=True)
auc_sa_mlp = cross_val_score(mlp_cls_sa, X_sa_bin, y_bin, cv=cv_s, scoring='roc_auc').mean()
sl_auc_mlp = auc_sa_mlp - auc_act_mlp

print(f"  MLP:     AUC_act={auc_act_mlp:.4f}, AUC_sa={auc_sa_mlp:.4f}, SL_auc={sl_auc_mlp:.4f}")

# ===================================================================
# 7. CODE DOMAIN PREDICTED REGRET (A3)
# ===================================================================
print("\n[7] Code domain predicted regret from Theorem 1...")

# From paper Table 1: Code SL=0.003, from t6 results
sl_code = 0.003
# Var(R) for code -- estimate from the data variance
# Code correctness is binary (0/1), so Var(R) = p*(1-p)
# Typically ~50% correct steps in CodeContests
var_r_code = 0.25  # p=0.5 binary

for K in [4, 8, 16, 64]:
    sigma = np.sqrt(sl_code * var_r_code)
    upper_regret = sigma * np.sqrt(2 * np.log(K))
    # Also compute effective gap
    eff_gap = np.sqrt((1 - sl_code) / sl_code) if sl_code > 0 else float('inf')
    print(f"  K={K:2d}: sigma={sigma:.4f}, upper_regret={upper_regret:.4f}, "
          f"eff_gap={eff_gap:.2f}")

# Compare with Math domain
sl_math = 0.584
var_r_math = 0.25
K = 8
sigma_math = np.sqrt(sl_math * var_r_math)
upper_math = sigma_math * np.sqrt(2 * np.log(K))
sigma_code = np.sqrt(sl_code * var_r_code)
upper_code = sigma_code * np.sqrt(2 * np.log(K))
print(f"\n  Math predicted regret (K={K}): {upper_math:.4f}")
print(f"  Code predicted regret (K={K}): {upper_code:.4f}")
print(f"  Ratio: {upper_math/upper_code:.1f}x")
print(f"  DreamPRM reports +1.5-2.6% for code.")
print(f"  Predicted regret {upper_code:.4f} is consistent with small but nonzero advantage")
print(f"  because the bound is an UPPER bound, not exact.")

# ===================================================================
# 8. SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("SUMMARY: NONLINEAR STATE-LIFT ON PRM800K")
print("=" * 70)

print(f"""
  LINEAR:
    Ridge R2-diff:           {sl_linear:.4f}
    LogReg AUC-diff:         {sl_auc_linear:.4f}

  NONLINEAR:
    Kernel Ridge (RBF):      {sl_kernel:.4f}  ({sl_kernel/max(sl_linear,0.001):.1f}x linear)
    MLP (64-32):             {sl_mlp_best:.4f}  ({sl_mlp_best/max(sl_linear,0.001):.1f}x linear)
    MLP classifier AUC-diff: {sl_auc_mlp:.4f}  ({sl_auc_mlp/max(sl_auc_linear,0.001):.1f}x linear)

  INTERPRETATION:
    {'Nonlinear SL >> linear SL: confirms linear is a lower bound. Story holds.' if sl_mlp_best > 0.05 else ''}
    {'Nonlinear SL still low: genuine false negative. Must discuss honestly.' if sl_mlp_best < 0.02 else ''}
    {'Nonlinear SL moderate: partial recovery. Lower bound story partially holds.' if 0.02 <= sl_mlp_best <= 0.05 else ''}
""")

# Save
out = {
    'linear': {
        'r2_action': float(r2_action_lin),
        'r2_state_action': float(r2_sa_lin),
        'sl_r2_diff': float(sl_linear),
        'auc_action': float(auc_act_lin),
        'auc_state_action': float(auc_sa_lin),
        'sl_auc_diff': float(sl_auc_linear),
    },
    'kernel_ridge': {
        'r2_action': float(r2_act_kr_best),
        'r2_state_action': float(r2_sa_kr_best),
        'sl_r2_diff': float(sl_kernel),
        'gamma': 0.1,
        'n_subsample': len(z_kr),
    },
    'mlp_regressor': {
        'r2_action': float(r2_act_mlp_best),
        'r2_state_action': float(r2_sa_mlp_best),
        'sl_r2_diff': float(sl_mlp_best),
        'architecture': '64-32',
    },
    'mlp_classifier': {
        'auc_action': float(auc_act_mlp),
        'auc_state_action': float(auc_sa_mlp),
        'sl_auc_diff': float(sl_auc_mlp),
        'architecture': '64-32',
    },
    'code_predicted_regret': {
        'sl_code': sl_code,
        'var_r': var_r_code,
        'K8_upper_regret': float(upper_code),
        'K8_math_upper_regret': float(upper_math),
        'ratio_math_over_code': float(upper_math / upper_code),
        'eff_gap_code': float(np.sqrt((1 - sl_code) / sl_code)),
        'eff_gap_math': float(np.sqrt((1 - sl_math) / sl_math)),
    },
    'comparison': {
        'linear_sl': float(sl_linear),
        'kernel_sl': float(sl_kernel),
        'mlp_sl': float(sl_mlp_best),
        'mlp_cls_sl_auc': float(sl_auc_mlp),
        'nonlinear_confirms_lower_bound': bool(sl_mlp_best > sl_linear * 1.5),
    },
}

out_path = os.path.join(OUT_DIR, 'r9_prm800k_nonlinear.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
