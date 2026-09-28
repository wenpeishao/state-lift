"""
R13b: Making state-lift work for math — without CrossEncoder (hangs on 5090).

Methods that capture cross-attention:
1. Bi-encoder baseline (encode separately) — SL ≈ 0 confirmed
2. Joint-embedding (concat text, single encoder) — SL = -0.27 (worse)
3. Manual cross-encoder using transformers AutoModel — replaces CE class
4. NLI via transformers pipeline
5. Similarity-enriched features
6. Larger encoder (mpnet) joint
"""

import numpy as np
import json
import os
import re
import warnings
import time
import sys

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
from scipy import stats as sp_stats
from datasets import load_dataset
import torch

SEED = 42
D_PCA = 16
np.random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_math_shepherd():
    print("Loading MathShepherd...")
    ds = load_dataset("peiyi9979/Math-Shepherd", split="train")
    states, actions, ratings = [], [], []
    n_parsed = 0
    for ex in ds:
        if n_parsed >= 30000:
            break
        inp, lbl = ex.get('input', ''), ex.get('label', '')
        if not inp or not lbl:
            continue
        ki_positions = [m.start() for m in re.finditer(re.escape('ки'), inp)]
        if len(ki_positions) < 2:
            continue
        step_match = re.search(r'Step \d+:', inp)
        problem = inp[:step_match.start()].strip() if step_match else inp[:ki_positions[0]].strip()
        if not problem:
            continue
        step_texts, step_labels = [], []
        for idx, ki_pos in enumerate(ki_positions):
            start = (step_match.start() if step_match else 0) if idx == 0 else ki_positions[idx-1] + 2
            step_text = re.sub(r'^Step \d+:\s*', '', inp[start:ki_pos].strip()).strip().lstrip('\n').strip()
            if not step_text:
                continue
            if ki_pos < len(lbl):
                c = lbl[ki_pos]
                step_labels.append(1 if c == '+' else (-1 if c == '-' else 0))
            else:
                step_labels.append(0)
            step_texts.append(step_text)
        valid = [(t, l) for t, l in zip(step_texts, step_labels) if l != 0]
        if len(valid) < 2:
            continue
        for i, (st, lb) in enumerate(valid):
            state = problem if i == 0 else problem + "\n" + "\n".join([v[0] for v in valid[:i]])
            states.append(state[:1024])
            actions.append(st[:512])
            ratings.append(lb)
        n_parsed += 1
    print(f"  {n_parsed} chains, {len(states)} steps, +{sum(1 for r in ratings if r>0)}/-{sum(1 for r in ratings if r<0)}")
    return states, actions, ratings


def compute_sl_ridge(features_a, features_sa, y, name):
    """Compute SL from two feature sets."""
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2_a = cross_val_score(Ridge(alpha=1.0), features_a, y, cv=cv, scoring='r2').mean()
    r2_sa = cross_val_score(Ridge(alpha=1.0), features_sa, y, cv=cv, scoring='r2').mean()
    sl = r2_sa - max(r2_a, 0)

    y_bin = (y > 0).astype(float)
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    try:
        auc_a = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                 features_a, y_bin, cv=cv_s, scoring='roc_auc').mean()
        auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                  features_sa, y_bin, cv=cv_s, scoring='roc_auc').mean()
        sl_auc = auc_sa - auc_a
    except:
        auc_a, auc_sa, sl_auc = 0, 0, 0

    print(f"  [{name}] R²_a={r2_a:.4f} R²_sa={r2_sa:.4f} SL={sl:.4f} | AUC_a={auc_a:.4f} AUC_sa={auc_sa:.4f} SL_AUC={sl_auc:.4f}")
    return {'r2_a': float(r2_a), 'r2_sa': float(r2_sa), 'sl': float(sl),
            'auc_a': float(auc_a), 'auc_sa': float(auc_sa), 'sl_auc': float(sl_auc)}


# ===================================================================
# Method 3: Manual cross-encoder using transformers
# ===================================================================
def manual_cross_encoder_scores(states, actions, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """Score (state, action) pairs using transformers directly."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()

    scores_sa = []
    scores_a = []
    batch_size = 32

    print(f"  Scoring {len(states)} (state, action) pairs...")
    for i in range(0, len(states), batch_size):
        batch_states = states[i:i+batch_size]
        batch_actions = actions[i:i+batch_size]

        # State + action
        inputs = tokenizer(batch_states, batch_actions, padding=True, truncation=True,
                           max_length=512, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
        scores_sa.extend(logits.cpu().numpy().tolist())

        # Action only (empty state)
        empty_states = [''] * len(batch_actions)
        inputs_a = tokenizer(empty_states, batch_actions, padding=True, truncation=True,
                             max_length=512, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits_a = model(**inputs_a).logits.squeeze(-1)
        scores_a.extend(logits_a.cpu().numpy().tolist())

        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i//batch_size}/{len(states)//batch_size}")

    return np.array(scores_sa), np.array(scores_a)


# ===================================================================
# Method 4: NLI via transformers
# ===================================================================
def nli_scores(states, actions, model_name='cross-encoder/nli-MiniLM2-L6-H768'):
    """NLI entailment scores using transformers directly."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"  Loading NLI model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()

    all_probs = []
    batch_size = 32

    print(f"  Scoring {len(states)} NLI pairs...")
    for i in range(0, len(states), batch_size):
        batch_s = states[i:i+batch_size]
        batch_a = actions[i:i+batch_size]

        inputs = tokenizer(batch_s, batch_a, padding=True, truncation=True,
                           max_length=512, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        all_probs.extend(probs.cpu().numpy().tolist())

        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i//batch_size}/{len(states)//batch_size}")

    probs = np.array(all_probs)
    return probs  # [contradiction, neutral, entailment]


# ===================================================================
# MAIN
# ===================================================================
def main():
    states, actions, ratings = load_math_shepherd()
    y = np.array(ratings, dtype=float)

    results = {}

    # =========================
    # 1. Bi-encoder baseline
    # =========================
    print("\n=== Method 1: Bi-encoder (baseline) ===")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    state_embs = encoder.encode(states, batch_size=256, show_progress_bar=True)
    action_embs = encoder.encode(actions, batch_size=256, show_progress_bar=True)

    pca = PCA(n_components=D_PCA)
    pca.fit(np.vstack([state_embs, action_embs]))
    z = pca.transform(state_embs)
    a = pca.transform(action_embs)
    X_sa = np.hstack([a, z, a * z])

    res1 = compute_sl_ridge(a, X_sa, y, 'bi-encoder')
    results['bi-encoder'] = res1

    # =========================
    # 2. Similarity-enriched
    # =========================
    print("\n=== Method 2: Similarity-enriched ===")
    cos_sim = np.sum(state_embs * action_embs, axis=1) / (
        np.linalg.norm(state_embs, axis=1) * np.linalg.norm(action_embs, axis=1) + 1e-8)
    l2_dist = np.linalg.norm(state_embs - action_embs, axis=1)

    # Hadamard interaction
    pca_h = PCA(n_components=D_PCA)
    h = pca_h.fit_transform(state_embs * action_embs)

    # Abs diff
    pca_d = PCA(n_components=D_PCA)
    d = pca_d.fit_transform(np.abs(state_embs - action_embs))

    X_enrich = np.hstack([a, cos_sim.reshape(-1,1), l2_dist.reshape(-1,1), h, d])

    rho_cos, p_cos = sp_stats.spearmanr(cos_sim, y)
    print(f"  Cosine sim ↔ correctness: rho={rho_cos:.4f} (p={p_cos:.4f})")

    res2 = compute_sl_ridge(a, X_enrich, y, 'similarity-enriched')
    results['similarity-enriched'] = {**res2, 'cos_corr': float(rho_cos), 'cos_p': float(p_cos)}

    # =========================
    # 3. Cross-encoder (manual)
    # =========================
    print("\n=== Method 3: Cross-encoder (manual transformers) ===")
    try:
        ce_scores_sa, ce_scores_a = manual_cross_encoder_scores(states, actions)
        rho_ce, p_ce = sp_stats.spearmanr(ce_scores_sa, y)
        print(f"  CE score ↔ correctness: rho={rho_ce:.4f} (p={p_ce:.6f})")
        print(f"  CE(s,a) mean={ce_scores_sa.mean():.4f}, CE(a) mean={ce_scores_a.mean():.4f}")

        features_a = ce_scores_a.reshape(-1, 1)
        features_sa = np.hstack([ce_scores_sa.reshape(-1,1), ce_scores_a.reshape(-1,1),
                                  (ce_scores_sa - ce_scores_a).reshape(-1,1)])

        res3 = compute_sl_ridge(features_a, features_sa, y, 'cross-encoder')
        results['cross-encoder'] = {**res3, 'corr': float(rho_ce), 'corr_p': float(p_ce)}

        # Also: cross-encoder features + action embeddings
        X_ce_full = np.hstack([a, features_sa])
        res3b = compute_sl_ridge(a, X_ce_full, y, 'cross-encoder+emb')
        results['cross-encoder+emb'] = res3b
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results['cross-encoder'] = {'status': 'error', 'error': str(e)}

    # =========================
    # 4. NLI
    # =========================
    print("\n=== Method 4: NLI ===")
    try:
        nli_probs = nli_scores(states, actions)
        if nli_probs.shape[1] >= 3:
            entail = nli_probs[:, 2]
            contra = nli_probs[:, 0]
        else:
            entail = nli_probs[:, 0]
            contra = np.zeros_like(entail)

        rho_e, p_e = sp_stats.spearmanr(entail, y)
        rho_c, p_c = sp_stats.spearmanr(contra, y)
        print(f"  Entailment ↔ correctness: rho={rho_e:.4f} (p={p_e:.6f})")
        print(f"  Contradiction ↔ correctness: rho={rho_c:.4f} (p={p_c:.6f})")

        features_nli = nli_probs
        X_nli_full = np.hstack([a, features_nli])

        res4_nli = compute_sl_ridge(a, X_nli_full, y, 'nli+emb')
        res4_nli_only = compute_sl_ridge(
            np.zeros((len(y), 1)),  # dummy baseline
            features_nli, y, 'nli-only'
        )
        results['nli'] = {**res4_nli,
                          'entail_corr': float(rho_e), 'contra_corr': float(rho_c)}
        results['nli-only'] = res4_nli_only
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results['nli'] = {'status': 'error', 'error': str(e)}

    # =========================
    # 5. Larger encoder (mpnet) with joint encoding
    # =========================
    print("\n=== Method 5: MPNet joint ===")
    try:
        enc2 = SentenceTransformer('all-mpnet-base-v2')
        a_mp = enc2.encode(actions, batch_size=128, show_progress_bar=True)
        j_mp = enc2.encode([s + " [SEP] " + a for s, a in zip(states, actions)],
                           batch_size=64, show_progress_bar=True)

        pca_amp = PCA(n_components=D_PCA)
        a_mp_r = pca_amp.fit_transform(a_mp)
        pca_jmp = PCA(n_components=D_PCA)
        j_mp_r = pca_jmp.fit_transform(j_mp)

        res5 = compute_sl_ridge(a_mp_r, j_mp_r, y, 'mpnet-joint')
        results['mpnet-joint'] = res5
    except Exception as e:
        print(f"  ERROR: {e}")
        results['mpnet-joint'] = {'status': 'error', 'error': str(e)}

    # =========================
    # 6. DeBERTa cross-encoder (better cross-attention)
    # =========================
    print("\n=== Method 6: DeBERTa cross-encoder ===")
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
        model = AutoModel.from_pretrained('microsoft/deberta-v3-small').to(DEVICE)
        model.eval()

        # Get [CLS] embeddings for action-only and state+action
        def get_cls_embeddings(texts, batch_size=32):
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors='pt').to(DEVICE)
                with torch.no_grad():
                    outputs = model(**inputs)
                    cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embs.append(cls)
            return np.vstack(all_embs)

        print("  Encoding actions...")
        deberta_a = get_cls_embeddings(actions)
        print("  Encoding state+action...")
        deberta_sa = get_cls_embeddings([s + " [SEP] " + a for s, a in zip(states, actions)])

        pca_da = PCA(n_components=D_PCA)
        da = pca_da.fit_transform(deberta_a)
        pca_dsa = PCA(n_components=D_PCA)
        dsa = pca_dsa.fit_transform(deberta_sa)

        res6 = compute_sl_ridge(da, dsa, y, 'deberta-joint')
        results['deberta-joint'] = res6
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results['deberta-joint'] = {'status': 'error', 'error': str(e)}

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n\n" + "=" * 80)
    print("R13b SUMMARY: CAN WE MAKE STATE-LIFT WORK FOR MATH?")
    print("=" * 80)

    print(f"\n{'Method':<25} {'SL(R²)':<10} {'SL(AUC)':<10} {'R²_action':<10} {'R²_s+a':<10}")
    print("-" * 65)

    for name, res in results.items():
        if isinstance(res, dict) and 'sl' in res:
            print(f"{name:<25} {res['sl']:<10.4f} {res.get('sl_auc',0):<10.4f} "
                  f"{res['r2_a']:<10.4f} {res['r2_sa']:<10.4f}")

    print(f"\nEmpirical PRM gap (math): ~0.35-0.47")

    # Best method
    best_name, best_sl = None, -999
    for name, res in results.items():
        if isinstance(res, dict) and 'sl' in res:
            if res['sl'] > best_sl:
                best_sl = res['sl']
                best_name = name

    print(f"\nBest method: {best_name} (SL={best_sl:.4f})")
    if best_sl > 0.05:
        print("  --> SUCCESS: Captures math state-dependence!")
    elif best_sl > 0.01:
        print("  --> PARTIAL: Some signal but far from empirical gap")
    else:
        print("  --> NOT ENOUGH: Small encoders can't capture math state-dependence")
        print("  --> Need LLM-scale models (8B+) to see interaction-level signal")

    # Save
    out_path = os.path.join(OUT_DIR, 'r13b_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
