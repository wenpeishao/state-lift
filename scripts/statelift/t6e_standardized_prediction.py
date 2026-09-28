"""
T6e: Standardized SDI + PRM advantage computation.

Fix: T6d had inconsistent transition construction across datasets.
- Dialogue: MUST use alternating-speaker transitions (z=turn_t, a=turn_{t+1}, zn=turn_{t+2}, step=2)
- Chains: use consecutive steps (z=step_i, a=step_{i+1}, zn=step_{i+2}, step=1)

Also: compute BOTH SDI and PRM advantage from the SAME standardized pipeline,
with chain-level train/test splits for PRM advantage.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from scipy import stats
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from collections import defaultdict

SEED = 42
D_PCA = 16
OUT_DIR = 'results'
np.random.seed(SEED)

encoder = SentenceTransformer('all-MiniLM-L6-v2')


def standardized_sdi_and_prm(z_raw, a_raw, zn_raw, chain_ids, name=""):
    """
    Standardized computation of SDI and held-out PRM advantage.
    Uses same PCA, same improvement proxy, same alpha.
    """
    n = len(z_raw)
    if n < 100:
        print(f"  {name}: too few ({n})")
        return None

    d = min(D_PCA, n // 4)

    # PCA fit on ALL data
    pca = PCA(n_components=d)
    pca.fit(np.vstack([z_raw, a_raw, zn_raw]))
    z = pca.transform(z_raw)
    a = pca.transform(a_raw)
    zn = pca.transform(zn_raw)

    # Global mean for improvement
    z_mean = z.mean(0)
    improvement = np.linalg.norm(z - z_mean, axis=1) - np.linalg.norm(zn - z_mean, axis=1)

    # ===== SDI (5-fold CV on all data) =====
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    alpha = 1.0

    a_centroid = a.mean(0)
    X_outcome = np.hstack([a, np.linalg.norm(a, axis=1, keepdims=True),
                            np.linalg.norm(a - a_centroid, axis=1, keepdims=True)])
    r2_outcome = max(cross_val_score(Ridge(alpha=alpha), X_outcome, improvement,
                                      cv=cv, scoring='r2').mean(), 0.001)

    X_process = np.hstack([z, a, z * a])
    r2_process = cross_val_score(Ridge(alpha=alpha), X_process, improvement,
                                  cv=cv, scoring='r2').mean()

    sdi = r2_process / max(r2_outcome, 0.001)

    # ===== PRM advantage (held-out, chain-level split) =====
    unique_chains = np.unique(chain_ids)
    np.random.seed(SEED)
    np.random.shuffle(unique_chains)
    n_train = int(0.7 * len(unique_chains))
    train_set = set(unique_chains[:n_train])
    tr = np.array([c in train_set for c in chain_ids])
    te = ~tr

    if tr.sum() < 30 or te.sum() < 30:
        print(f"  {name}: too few in train ({tr.sum()}) or test ({te.sum()})")
        # Fall back to random split
        np.random.seed(SEED)
        tr = np.random.rand(n) < 0.7
        te = ~tr

    # Blind model
    X_blind_tr = np.hstack([a[tr], np.linalg.norm(a[tr], axis=1, keepdims=True),
                             np.linalg.norm(a[tr] - a_centroid, axis=1, keepdims=True)])
    X_blind_te = np.hstack([a[te], np.linalg.norm(a[te], axis=1, keepdims=True),
                             np.linalg.norm(a[te] - a_centroid, axis=1, keepdims=True)])
    m_blind = Ridge(alpha=alpha).fit(X_blind_tr, improvement[tr])
    r2_blind = r2_score(improvement[te], m_blind.predict(X_blind_te))
    rho_blind = stats.spearmanr(improvement[te], m_blind.predict(X_blind_te))[0]

    # Conditioned model
    X_cond_tr = np.hstack([z[tr], a[tr], z[tr]*a[tr]])
    X_cond_te = np.hstack([z[te], a[te], z[te]*a[te]])
    m_cond = Ridge(alpha=alpha).fit(X_cond_tr, improvement[tr])
    r2_cond = r2_score(improvement[te], m_cond.predict(X_cond_te))
    rho_cond = stats.spearmanr(improvement[te], m_cond.predict(X_cond_te))[0]

    prm_adv_r2 = r2_cond - r2_blind
    prm_adv_rank = rho_cond - rho_blind

    print(f"  {name}: n={n}, SDI={sdi:.2f}, PRM_adv(R2)={prm_adv_r2:.4f}, PRM_adv(rank)={prm_adv_rank:.4f}")

    return {
        'name': name, 'n': n,
        'SDI': float(sdi),
        'r2_outcome': float(r2_outcome), 'r2_process': float(r2_process),
        'r2_blind_test': float(r2_blind), 'r2_cond_test': float(r2_cond),
        'rho_blind_test': float(rho_blind), 'rho_cond_test': float(rho_cond),
        'prm_advantage_r2': float(prm_adv_r2),
        'prm_advantage_rank': float(prm_adv_rank),
    }


def embed_and_build(texts_list, chain_ids_list):
    """Embed texts and return arrays."""
    embs = encoder.encode(texts_list, batch_size=256, show_progress_bar=False)
    return embs


# ===================================================================
# STANDARDIZED DATA LOADING
# ===================================================================
print("=" * 70)
print("T6e: STANDARDIZED SDI + PRM ADVANTAGE")
print("=" * 70)

results = {}

# Helper for dialogue datasets
def process_dialogue(convs, name, min_turns=6, step=2):
    """Standard dialogue transition builder: alternating-speaker, step=2."""
    all_texts = []
    conv_bounds = []
    for conv in convs:
        if len(conv) < min_turns:
            continue
        s = len(all_texts)
        for spk, txt in conv:
            all_texts.append(str(txt)[:512])
        conv_bounds.append((s, len(all_texts)))

    if not all_texts or len(conv_bounds) < 20:
        print(f"  {name}: too few conversations ({len(conv_bounds)})")
        return None

    print(f"  Embedding {len(all_texts)} turns from {len(conv_bounds)} conversations...")
    embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)

    tz, ta, tzn, cids = [], [], [], []
    for ci, (s, e) in enumerate(conv_bounds):
        n = e - s
        if n < min_turns:
            continue
        for i in range(step, n - step + 1, step):
            if s + i - step >= 0 and s + i < len(embs) and s + i + step - 1 < e:
                tz.append(embs[s + i - step])
                ta.append(embs[s + i - step + 1] if step == 2 else embs[s + i])
                tzn.append(embs[s + i])
                cids.append(ci)

    if len(tz) < 100:
        print(f"  {name}: too few transitions ({len(tz)})")
        return None

    return standardized_sdi_and_prm(np.array(tz), np.array(ta), np.array(tzn), np.array(cids), name)


def process_chains(chains, name):
    """Standard chain transition builder: consecutive steps."""
    all_texts, chain_bounds = [], []
    for chain in chains:
        steps = chain['steps']
        if len(steps) < 3:
            continue
        s = len(all_texts)
        if 'context' in chain and chain['context']:
            all_texts.append(chain['context'][:512])
        for step in steps:
            all_texts.append(str(step)[:512])
        chain_bounds.append((s, len(all_texts)))

    if not all_texts:
        return None

    print(f"  Embedding {len(all_texts)} texts from {len(chain_bounds)} chains...")
    embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)

    tz, ta, tzn, cids = [], [], [], []
    for ci, (s, e) in enumerate(chain_bounds):
        n = e - s
        if n < 3:
            continue
        for i in range(1, n - 1):
            tz.append(embs[s + i - 1])
            ta.append(embs[s + i])
            tzn.append(embs[s + i + 1])
            cids.append(ci)

    if len(tz) < 100:
        return None

    return standardized_sdi_and_prm(np.array(tz), np.array(ta), np.array(tzn), np.array(cids), name)


# ------------------------------------------------------------------
# 1. MATH REASONING (GSM8K) -- chain
# ------------------------------------------------------------------
print("\n[1] Math Reasoning (GSM8K)...")
ds = load_dataset('openai/gsm8k', 'main', split='train')
chains = []
for ex in list(ds)[:2000]:
    steps = [s.strip() for s in ex.get('answer', '').split('\n') if s.strip() and len(s.strip()) > 10]
    if len(steps) >= 3:
        chains.append({'context': ex.get('question', ''), 'steps': steps})
r = process_chains(chains, "Math-GSM8K")
if r: results['Math Reasoning'] = {**r, 'type': 'chain', 'sequential': True}

# ------------------------------------------------------------------
# 2. TUTORING (MathDial) -- dialogue, alternating
# ------------------------------------------------------------------
print("\n[2] Tutoring (MathDial)...")
ds = load_dataset('eth-nlped/mathdial', split='train')
convs = []
for ex in list(ds):
    conv_text = ex.get('conversation', '')
    if isinstance(conv_text, str):
        turns = conv_text.split('\n')
        conv = []
        for t in turns:
            t = t.strip()
            if not t: continue
            if ':' in t:
                role, text = t.split(':', 1)
                conv.append((role.strip(), text.strip()))
        if len(conv) >= 4:
            convs.append(conv)
r = process_dialogue(convs, "Tutoring-MathDial", min_turns=4, step=2)
if r: results['Tutoring'] = {**r, 'type': 'dialogue', 'sequential': True}

# ------------------------------------------------------------------
# 3. WEB NAVIGATION (Mind2Web) -- chain
# ------------------------------------------------------------------
print("\n[3] Web Navigation (Mind2Web)...")
ds = load_dataset('osunlp/Mind2Web', split='train')
chains = []
for ex in list(ds)[:2000]:
    actions = ex.get('action_reprs', [])
    if isinstance(actions, list) and len(actions) >= 3:
        chains.append({'context': str(ex.get('confirmed_task', ''))[:512],
                      'steps': [str(a)[:512] for a in actions]})
r = process_chains(chains, "WebNav-Mind2Web")
if r: results['Web Navigation'] = {**r, 'type': 'chain', 'sequential': True}

# ------------------------------------------------------------------
# 4. CODE REASONING (CodeContests) -- chain
# ------------------------------------------------------------------
print("\n[4] Code Reasoning (CodeContests)...")
ds = load_dataset('deepmind/code_contests', split='train')
chains = []
for ex in list(ds)[:1000]:
    solutions = ex.get('solutions', {})
    sol_list = solutions.get('solution', []) if isinstance(solutions, dict) else []
    for sol in sol_list[:2]:
        if isinstance(sol, str) and len(sol) > 50:
            lines = sol.split('\n')
            blocks, current = [], []
            for line in lines:
                current.append(line)
                if len('\n'.join(current)) > 100 and (
                    line.strip() == '' or line.strip().startswith('def ') or
                    line.strip().startswith('class ')):
                    blocks.append('\n'.join(current)); current = []
            if current: blocks.append('\n'.join(current))
            if len(blocks) >= 3:
                chains.append({'context': str(ex.get('description', ''))[:512],
                              'steps': [b[:512] for b in blocks]})
r = process_chains(chains, "Code-CodeContests")
if r: results['Code Reasoning'] = {**r, 'type': 'chain', 'sequential': True}

# ------------------------------------------------------------------
# 5. EMOTIONAL SUPPORT (ESConv) -- dialogue, alternating
# ------------------------------------------------------------------
print("\n[5] Emotional Support (ESConv)...")
ds = load_dataset('thu-coai/esconv', split='train')
convs = []
for ex in ds:
    parsed = json.loads(ex['text'])
    dialog = parsed.get('dialog', [])
    if len(dialog) >= 6:
        convs.append([(t.get('speaker', 'usr'), t['text']) for t in dialog])
r = process_dialogue(convs[:2000], "ESConv", min_turns=6, step=2)
if r: results['Emotional Support'] = {**r, 'type': 'dialogue', 'sequential': True}

# ------------------------------------------------------------------
# 6. TOOL-USE (Glaive) -- dialogue-like, multi-role
# ------------------------------------------------------------------
print("\n[6] Tool-Use (Glaive)...")
ds = load_dataset('glaiveai/glaive-function-calling-v2', split='train')
convs = []
for ex in list(ds)[:3000]:
    chat = ex.get('chat', '')
    if isinstance(chat, str) and len(chat) > 100:
        conv = []
        for line in chat.split('\n'):
            line = line.strip()
            if not line or len(line) < 10: continue
            for prefix in ['USER:', 'ASSISTANT:', 'FUNCTION RESPONSE:', 'SYSTEM:']:
                if line.upper().startswith(prefix):
                    text = line[len(prefix):].strip()
                    if text: conv.append((prefix[:-1].lower(), text[:512]))
                    break
        if len(conv) >= 4:
            convs.append(conv)
r = process_dialogue(convs, "ToolUse-Glaive", min_turns=4, step=2)
if r: results['Tool-Use'] = {**r, 'type': 'dialogue', 'sequential': True}

# ------------------------------------------------------------------
# 7. NEGOTIATION (CaSiNo) -- dialogue, alternating
# ------------------------------------------------------------------
print("\n[7] Negotiation (CaSiNo)...")
ds = load_dataset('casino', split='train')
convs = []
for ex in list(ds):
    chat = ex.get('chat_logs', [])
    if isinstance(chat, list) and len(chat) >= 4:
        conv = [(str(t.get('id', 'spk')), str(t.get('text', ''))[:512])
                for t in chat if isinstance(t, dict) and t.get('text')]
        if len(conv) >= 4:
            convs.append(conv)
r = process_dialogue(convs, "Negotiation-CaSiNo", min_turns=4, step=2)
if r: results['Negotiation'] = {**r, 'type': 'dialogue', 'sequential': True}

# ------------------------------------------------------------------
# 8. HH-RLHF -- dialogue, alternating
# ------------------------------------------------------------------
print("\n[8] Chat QA (HH-RLHF)...")
ds = load_dataset('Anthropic/hh-rlhf', split='train', data_dir='helpful-base')
convs = []
for ex in list(ds)[:3000]:
    text = ex.get('chosen', '')
    turns = text.split('\n\n')
    conv = []
    for t in turns:
        t = t.strip()
        if t.startswith('Human:'): conv.append(('human', t[6:].strip()))
        elif t.startswith('Assistant:'): conv.append(('assistant', t[10:].strip()))
    if len(conv) >= 6:
        convs.append(conv)
r = process_dialogue(convs, "HH-RLHF", min_turns=6, step=2)
if r: results['Chat QA'] = {**r, 'type': 'dialogue', 'sequential': False}

# ------------------------------------------------------------------
# 9. FACTUAL QA (Alpaca) -- fake transitions (non-sequential)
# ------------------------------------------------------------------
print("\n[9] Factual QA (Alpaca)...")
ds = load_dataset('tatsu-lab/alpaca', split='train')
qa = [(ex.get('instruction', '')[:512], ex.get('output', '')[:512])
      for ex in list(ds)[:3000]
      if len(ex.get('instruction', '')) > 10 and len(ex.get('output', '')) > 10]
all_qa = [t for pair in qa for t in pair]
embs = encoder.encode(all_qa, batch_size=256, show_progress_bar=False)
tz, ta, tzn, cids = [], [], [], []
for i in range(len(qa) - 1):
    tz.append(embs[i*2]); ta.append(embs[i*2+1]); tzn.append(embs[(i+1)*2]); cids.append(i)
r = standardized_sdi_and_prm(np.array(tz), np.array(ta), np.array(tzn), np.array(cids), "FactualQA")
if r: results['Factual QA'] = {**r, 'type': 'fake-sequential', 'sequential': False}

# ------------------------------------------------------------------
# 10. CODE GEN (single-turn) -- fake transitions
# ------------------------------------------------------------------
print("\n[10] Code Gen (single-turn)...")
ds = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
code = [(str(ex.get('prompt', ex.get('instruction', '')))[:512],
         str(ex.get('output', ex.get('response', '')))[:512])
        for ex in list(ds)[:3000]
        if len(str(ex.get('prompt', ex.get('instruction', '')))) > 10]
all_code = [t for pair in code for t in pair]
embs = encoder.encode(all_code, batch_size=256, show_progress_bar=False)
tz, ta, tzn, cids = [], [], [], []
for i in range(len(code) - 1):
    tz.append(embs[i*2]); ta.append(embs[i*2+1]); tzn.append(embs[(i+1)*2]); cids.append(i)
r = standardized_sdi_and_prm(np.array(tz), np.array(ta), np.array(tzn), np.array(cids), "CodeGen")
if r: results['Code Gen'] = {**r, 'type': 'fake-sequential', 'sequential': False}


# ===================================================================
# ANALYSIS
# ===================================================================
print("\n" + "=" * 70)
print("STANDARDIZED RESULTS")
print("=" * 70)

print(f"\n{'Domain':<25} {'SDI':>8} {'PRM(R2)':>9} {'PRM(rank)':>10} {'Type':<15} {'Seq?':>5}")
print("-" * 75)
for name, r in sorted(results.items(), key=lambda x: x[1]['SDI'], reverse=True):
    print(f"{name:<25} {r['SDI']:>8.2f} {r['prm_advantage_r2']:>9.4f} "
          f"{r['prm_advantage_rank']:>10.4f} {r.get('type','?'):<15} "
          f"{'YES' if r.get('sequential') else 'no':>5}")

# Correlations
sdi_vals = np.array([r['SDI'] for r in results.values()])
adv_r2 = np.array([r['prm_advantage_r2'] for r in results.values()])
adv_rank = np.array([r['prm_advantage_rank'] for r in results.values()])
log_sdi = np.log(np.maximum(sdi_vals, 0.1))

print(f"\n  N = {len(sdi_vals)} domains")

r1, p1 = stats.pearsonr(log_sdi, adv_r2)
rho1, prho1 = stats.spearmanr(sdi_vals, adv_r2)
print(f"\n  SDI vs PRM advantage (R2 gap):")
print(f"    Pearson r(log SDI, R2) = {r1:.3f} (p={p1:.4f})")
print(f"    Spearman rho = {rho1:.3f} (p={prho1:.4f})")

r2, p2 = stats.pearsonr(log_sdi, adv_rank)
rho2, prho2 = stats.spearmanr(sdi_vals, adv_rank)
print(f"\n  SDI vs PRM advantage (Rank gap):")
print(f"    Pearson r(log SDI, Rank) = {r2:.3f} (p={p2:.4f})")
print(f"    Spearman rho = {rho2:.3f} (p={prho2:.4f})")

# Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(log_sdi.reshape(-1, 1), adv_rank)
r2_fit = lr.score(log_sdi.reshape(-1, 1), adv_rank)
print(f"\n  Regression (rank advantage):")
print(f"    PRM_rank_adv = {lr.coef_[0]:.4f} * log(SDI) + {lr.intercept_:.4f}")
print(f"    R2 = {r2_fit:.3f}")

# Bootstrap CI
np.random.seed(SEED)
boot_rhos_r2, boot_rhos_rank = [], []
for _ in range(1000):
    idx = np.random.choice(len(sdi_vals), len(sdi_vals), replace=True)
    if len(np.unique(sdi_vals[idx])) >= 3:
        boot_rhos_r2.append(stats.spearmanr(sdi_vals[idx], adv_r2[idx])[0])
        boot_rhos_rank.append(stats.spearmanr(sdi_vals[idx], adv_rank[idx])[0])

print(f"\n  Bootstrap 95% CI (1000 resamples):")
print(f"    Spearman rho (R2 gap): {rho1:.3f} [{np.percentile(boot_rhos_r2, 2.5):.3f}, {np.percentile(boot_rhos_r2, 97.5):.3f}]")
print(f"    Spearman rho (Rank gap): {rho2:.3f} [{np.percentile(boot_rhos_rank, 2.5):.3f}, {np.percentile(boot_rhos_rank, 97.5):.3f}]")

# Prediction table
print(f"\n  PREDICTION TABLE:")
print(f"  {'SDI':>6} {'Predicted Rank Advantage':>25}")
print(f"  {'-'*35}")
for sdi_test in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    pred = lr.predict(np.array([[np.log(sdi_test)]]))[0]
    print(f"  {sdi_test:>6.1f} {pred:>25.4f}")

# Save
out_path = os.path.join(OUT_DIR, 't6e_standardized.json')
with open(out_path, 'w') as f:
    json.dump({
        'results': results,
        'correlations': {
            'n': len(sdi_vals),
            'pearson_log_sdi_vs_r2_gap': {'r': float(r1), 'p': float(p1)},
            'spearman_sdi_vs_r2_gap': {'rho': float(rho1), 'p': float(prho1)},
            'pearson_log_sdi_vs_rank_gap': {'r': float(r2), 'p': float(p2)},
            'spearman_sdi_vs_rank_gap': {'rho': float(rho2), 'p': float(prho2)},
            'bootstrap_95ci_r2': [float(np.percentile(boot_rhos_r2, 2.5)), float(np.percentile(boot_rhos_r2, 97.5))],
            'bootstrap_95ci_rank': [float(np.percentile(boot_rhos_rank, 2.5)), float(np.percentile(boot_rhos_rank, 97.5))],
        },
        'regression_rank': {
            'a': float(lr.coef_[0]), 'b': float(lr.intercept_), 'R2': float(r2_fit),
        },
    }, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
