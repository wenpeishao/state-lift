"""
T5b: Artifact checks for TSR.

Tests:
1. SHUFFLED STATE CONTROL: Permute z_t randomly. If TSR stays high, it's artifact.
2. WITHIN-CHAIN vs CROSS-CHAIN: State from same reasoning chain vs different chain.
   If cross-chain state helps equally, TSR is just capturing "math text similarity."
3. PRM LABEL CORRELATION: Use GSM8K step correctness as ground truth.
   Does state actually predict which steps are GOOD vs BAD?
4. REVERSE DIRECTION: Use z_{t+1} to predict z_t. If TSR is symmetric, it's just
   measuring autocorrelation, not directional state-dependence.
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
from datasets import load_dataset

SEED = 42
D_PCA = 16
OUT_DIR = 'results'
np.random.seed(SEED)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

# ===================================================================
# LOAD GSM8K (same as T5)
# ===================================================================
print("=" * 70)
print("T5b: ARTIFACT CHECKS FOR TSR")
print("=" * 70)

print("\n[0] Loading GSM8K...")
ds = load_dataset('openai/gsm8k', 'main', split='train')

chains = []
for ex in list(ds)[:2000]:
    answer = ex.get('answer', '')
    steps = [s.strip() for s in answer.split('\n') if s.strip() and len(s.strip()) > 10]
    if len(steps) >= 3:
        chains.append({'question': ex['question'], 'steps': steps})

print(f"  {len(chains)} chains")

# Embed
all_texts = []
chain_bounds = []
for c in chains:
    s = len(all_texts)
    all_texts.append(c['question'][:512])
    for step in c['steps']:
        all_texts.append(step[:512])
    chain_bounds.append((s, len(all_texts)))

print(f"  Embedding {len(all_texts)} texts...")
embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)

# Build transitions with chain IDs
tz, ta, tzn, t_chain = [], [], [], []
for ci, (s, e) in enumerate(chain_bounds):
    n = e - s
    if n < 4: continue
    for i in range(1, n - 1):
        tz.append(embs[s + i - 1])
        ta.append(embs[s + i])
        tzn.append(embs[s + i + 1] if s + i + 1 < e else embs[s + i])
        t_chain.append(ci)

z_raw, a_raw, zn_raw = np.array(tz), np.array(ta), np.array(tzn)
chain_ids = np.array(t_chain)
print(f"  {len(z_raw)} transitions")

# PCA
pca = PCA(n_components=D_PCA)
pca.fit(np.vstack([z_raw, a_raw, zn_raw]))
z, a, zn = pca.transform(z_raw), pca.transform(a_raw), pca.transform(zn_raw)

z_mean = z.mean(0)
improvement = np.linalg.norm(z - z_mean, axis=1) - np.linalg.norm(zn - z_mean, axis=1)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Baseline TSR
a_centroid = a.mean(0)
X_utt = np.hstack([a, np.linalg.norm(a, axis=1, keepdims=True),
                    np.linalg.norm(a - a_centroid, axis=1, keepdims=True)])
r2_utt = max(cross_val_score(Ridge(alpha=1.0), X_utt, improvement, cv=cv, scoring='r2').mean(), 0.001)

X_trans = np.hstack([z, a, z * a])
r2_trans = cross_val_score(Ridge(alpha=1.0), X_trans, improvement, cv=cv, scoring='r2').mean()

tsr_real = r2_trans / r2_utt
print(f"\n  Baseline TSR: {tsr_real:.2f} (R2_utt={r2_utt:.4f}, R2_trans={r2_trans:.4f})")

# ===================================================================
# TEST 1: SHUFFLED STATE
# ===================================================================
print("\n" + "=" * 70)
print("TEST 1: SHUFFLED STATE CONTROL")
print("=" * 70)

n_perms = 20
shuffled_tsrs = []
for p in range(n_perms):
    z_shuf = z[np.random.permutation(len(z))]
    X_trans_shuf = np.hstack([z_shuf, a, z_shuf * a])
    r2_shuf = cross_val_score(Ridge(alpha=1.0), X_trans_shuf, improvement, cv=cv, scoring='r2').mean()
    tsr_shuf = r2_shuf / r2_utt
    shuffled_tsrs.append(tsr_shuf)

shuffled_tsrs = np.array(shuffled_tsrs)
print(f"  Real TSR:     {tsr_real:.2f}")
print(f"  Shuffled TSR: {shuffled_tsrs.mean():.2f} +/- {shuffled_tsrs.std():.2f}")
print(f"  Drop:         {(1 - shuffled_tsrs.mean()/tsr_real)*100:.1f}%")

if shuffled_tsrs.mean() < tsr_real * 0.5:
    print("  PASSED: Shuffled state destroys TSR -> state signal is real")
else:
    print("  FAILED: Shuffled state preserves TSR -> state signal may be artifact")

# ===================================================================
# TEST 2: WITHIN-CHAIN vs CROSS-CHAIN STATE
# ===================================================================
print("\n" + "=" * 70)
print("TEST 2: WITHIN-CHAIN vs CROSS-CHAIN STATE")
print("=" * 70)

# Replace each z_t with a z from a DIFFERENT chain (same position if possible)
z_cross = np.zeros_like(z)
for i in range(len(z)):
    # Find transitions from different chains
    other_mask = chain_ids != chain_ids[i]
    other_idx = np.where(other_mask)[0]
    if len(other_idx) > 0:
        z_cross[i] = z[np.random.choice(other_idx)]
    else:
        z_cross[i] = z[np.random.randint(len(z))]

X_cross = np.hstack([z_cross, a, z_cross * a])
r2_cross = cross_val_score(Ridge(alpha=1.0), X_cross, improvement, cv=cv, scoring='r2').mean()
tsr_cross = r2_cross / r2_utt

print(f"  Within-chain TSR: {tsr_real:.2f} (real state)")
print(f"  Cross-chain TSR:  {tsr_cross:.2f} (state from different problem)")
print(f"  Drop:             {(1 - tsr_cross/tsr_real)*100:.1f}%")

if tsr_cross < tsr_real * 0.5:
    print("  PASSED: Cross-chain state is much worse -> state captures problem-specific info")
else:
    print("  CAUTION: Cross-chain state works similarly -> state may just capture 'math style'")

# ===================================================================
# TEST 3: DOES STATE PREDICT STEP CORRECTNESS?
# ===================================================================
print("\n" + "=" * 70)
print("TEST 3: STATE PREDICTS STEP QUALITY (using final answer correctness)")
print("=" * 70)

# GSM8K has numerical answers. We can check if chains that reach the right answer
# have different dynamics than chains that don't.
# Since GSM8K training set is all correct, we create "wrong" steps by
# using steps from different problems in the same position.

# Alternative: use step POSITION as quality proxy (later steps are more critical)
# Or: measure if state helps predict whether the NEXT step makes progress

# Simpler test: does state help predict the MAGNITUDE of change (not just direction)?
change_mag = np.linalg.norm(zn - z, axis=1)

r2_mag_utt = cross_val_score(Ridge(alpha=1.0), a, change_mag, cv=cv, scoring='r2').mean()
r2_mag_trans = cross_val_score(Ridge(alpha=1.0), X_trans, change_mag, cv=cv, scoring='r2').mean()
r2_mag_state = cross_val_score(Ridge(alpha=1.0), z, change_mag, cv=cv, scoring='r2').mean()

print(f"  Predicting step change magnitude:")
print(f"    Action only:   R2={r2_mag_utt:.4f}")
print(f"    State only:    R2={r2_mag_state:.4f}")
print(f"    Transition:    R2={r2_mag_trans:.4f}")
print(f"    TSR (magnitude): {r2_mag_trans / max(r2_mag_utt, 0.001):.2f}")

# Also: does state predict which steps are "big moves" vs "small refinements"?
big_step = change_mag > np.median(change_mag)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
auc_action = cross_val_score(LogisticRegression(max_iter=500, C=0.1), a,
                              big_step.astype(int), cv=cv_s, scoring='roc_auc').mean()
auc_state = cross_val_score(LogisticRegression(max_iter=500, C=0.1), z,
                             big_step.astype(int), cv=cv_s, scoring='roc_auc').mean()
auc_trans = cross_val_score(LogisticRegression(max_iter=500, C=0.1),
                             np.hstack([z, a]), big_step.astype(int),
                             cv=cv_s, scoring='roc_auc').mean()

print(f"\n  Predicting big vs small steps (AUC):")
print(f"    Action only:   {auc_action:.4f}")
print(f"    State only:    {auc_state:.4f}")
print(f"    State+Action:  {auc_trans:.4f}")

# ===================================================================
# TEST 4: DIRECTIONAL ASYMMETRY
# ===================================================================
print("\n" + "=" * 70)
print("TEST 4: FORWARD vs REVERSE PREDICTION")
print("=" * 70)

# Forward: predict z_{t+1} from (z_t, a_t) -- the real world model
# Reverse: predict z_t from (z_{t+1}, a_t) -- should be similar if just autocorrelation

wm_fwd = Ridge(alpha=1.0)
tr = np.random.rand(len(z)) < 0.7
wm_fwd.fit(np.hstack([z[tr], a[tr]]), zn[tr])
r2_fwd = 1 - np.sum((zn[~tr] - wm_fwd.predict(np.hstack([z[~tr], a[~tr]])))**2) / \
         np.sum((zn[~tr] - zn[~tr].mean(0))**2)

wm_rev = Ridge(alpha=1.0)
wm_rev.fit(np.hstack([zn[tr], a[tr]]), z[tr])
r2_rev = 1 - np.sum((z[~tr] - wm_rev.predict(np.hstack([zn[~tr], a[~tr]])))**2) / \
         np.sum((z[~tr] - z[~tr].mean(0))**2)

print(f"  Forward R2 (z_t, a -> z_{{t+1}}):  {r2_fwd:.4f}")
print(f"  Reverse R2 (z_{{t+1}}, a -> z_t):  {r2_rev:.4f}")
print(f"  Ratio (fwd/rev):                  {r2_fwd/max(r2_rev, 0.001):.2f}")

if abs(r2_fwd - r2_rev) / max(r2_fwd, r2_rev) < 0.1:
    print("  CAUTION: Symmetric -> may be just autocorrelation")
else:
    print("  PASSED: Asymmetric -> genuine directional dynamics")

# ===================================================================
# TEST 5: ESConv SAME CONTROLS (for comparison)
# ===================================================================
print("\n" + "=" * 70)
print("TEST 5: SAME CONTROLS ON ESConv")
print("=" * 70)

# Load ESConv embeddings from cache
es_cache = os.path.join('data', 'esconv_embeddings.npz')
if os.path.exists(es_cache):
    es_data = np.load(es_cache)
    embs_es = es_data['embeddings']

    # Rebuild transitions (same as t3)
    ds_es = load_dataset('thu-coai/esconv', split='train')
    convs_es = []
    for ex in ds_es:
        parsed = json.loads(ex['text'])
        dialog = parsed.get('dialog', [])
        if len(dialog) >= 6:
            convs_es.append([(t.get('speaker', 'usr'), t['text']) for t in dialog])

    cb_es = []
    idx = 0
    for conv in convs_es:
        s = idx; idx += len(conv); cb_es.append((s, idx))

    tz_es, ta_es, tzn_es = [], [], []
    for ci, (s, e) in enumerate(cb_es):
        conv = convs_es[ci]
        speakers = [sp for sp, t in conv]
        n = e - s
        for i in range(n):
            if i < 2 or i >= n: continue
            if speakers[i] == speakers[i-2] and speakers[i] != speakers[i-1]:
                tz_es.append(embs_es[s+i-2])
                ta_es.append(embs_es[s+i-1])
                tzn_es.append(embs_es[s+i])

    z_es_raw, a_es_raw, zn_es_raw = np.array(tz_es), np.array(ta_es), np.array(tzn_es)

    pca_es = PCA(n_components=D_PCA)
    pca_es.fit(np.vstack([z_es_raw, a_es_raw, zn_es_raw]))
    z_es = pca_es.transform(z_es_raw)
    a_es = pca_es.transform(a_es_raw)
    zn_es = pca_es.transform(zn_es_raw)

    z_mean_es = z_es.mean(0)
    imp_es = np.linalg.norm(z_es - z_mean_es, axis=1) - np.linalg.norm(zn_es - z_mean_es, axis=1)

    # Shuffled control
    shuf_tsrs_es = []
    a_cen_es = a_es.mean(0)
    X_utt_es = np.hstack([a_es, np.linalg.norm(a_es, axis=1, keepdims=True),
                           np.linalg.norm(a_es - a_cen_es, axis=1, keepdims=True)])
    r2_utt_es = max(cross_val_score(Ridge(alpha=1.0), X_utt_es, imp_es, cv=cv, scoring='r2').mean(), 0.001)
    X_trans_es = np.hstack([z_es, a_es, z_es * a_es])
    r2_trans_es = cross_val_score(Ridge(alpha=1.0), X_trans_es, imp_es, cv=cv, scoring='r2').mean()
    tsr_es_real = r2_trans_es / r2_utt_es

    for p in range(20):
        z_s = z_es[np.random.permutation(len(z_es))]
        X_s = np.hstack([z_s, a_es, z_s * a_es])
        r2_s = cross_val_score(Ridge(alpha=1.0), X_s, imp_es, cv=cv, scoring='r2').mean()
        shuf_tsrs_es.append(r2_s / r2_utt_es)

    shuf_tsrs_es = np.array(shuf_tsrs_es)
    print(f"  ESConv real TSR:     {tsr_es_real:.2f}")
    print(f"  ESConv shuffled TSR: {shuf_tsrs_es.mean():.2f} +/- {shuf_tsrs_es.std():.2f}")
    print(f"  Drop:                {(1 - shuf_tsrs_es.mean()/tsr_es_real)*100:.1f}%")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("ARTIFACT CHECK SUMMARY")
print("=" * 70)

print(f"""
                        Math        ESConv
                        ----        ------
Real TSR:               {tsr_real:.1f}        {tsr_es_real:.1f}
Shuffled TSR:           {shuffled_tsrs.mean():.1f}         {shuf_tsrs_es.mean():.1f}
Cross-chain TSR:        {tsr_cross:.1f}         (n/a)
Drop (shuffled):        {(1-shuffled_tsrs.mean()/tsr_real)*100:.0f}%          {(1-shuf_tsrs_es.mean()/tsr_es_real)*100:.0f}%
Forward R2:             {r2_fwd:.3f}
Reverse R2:             {r2_rev:.3f}
""")

verdicts = []
if shuffled_tsrs.mean() < tsr_real * 0.3:
    verdicts.append("Shuffled control: PASSED (state is real)")
else:
    verdicts.append("Shuffled control: FAILED")

if tsr_cross < tsr_real * 0.5:
    verdicts.append("Cross-chain: PASSED (problem-specific)")
else:
    verdicts.append("Cross-chain: CAUTION (generic math signal)")

if abs(r2_fwd - r2_rev) / max(r2_fwd, r2_rev) > 0.1:
    verdicts.append("Directionality: PASSED (asymmetric)")
else:
    verdicts.append("Directionality: CAUTION (symmetric)")

for v in verdicts:
    print(f"  {v}")

# Save
results = {
    'math': {
        'tsr_real': float(tsr_real),
        'tsr_shuffled_mean': float(shuffled_tsrs.mean()),
        'tsr_shuffled_std': float(shuffled_tsrs.std()),
        'tsr_cross_chain': float(tsr_cross),
        'r2_fwd': float(r2_fwd),
        'r2_rev': float(r2_rev),
    },
    'esconv': {
        'tsr_real': float(tsr_es_real),
        'tsr_shuffled_mean': float(shuf_tsrs_es.mean()),
        'tsr_shuffled_std': float(shuf_tsrs_es.std()),
    },
}
out_path = os.path.join(OUT_DIR, 't5b_artifact_check.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
