"""
T10: Mechanism Alignment -- do state_lift, anti-correlation, and PRM gap
all line up on the SAME domains?

For each domain with real quality labels, compute:
1. state_lift (from T8 -- already have)
2. utterance-transition correlation (new)
3. PRM gap (from E4/E3 -- already have)

The mechanism claim: ORM fails BECAUSE it optimizes utterance quality,
which is anti-correlated with transition quality in high-state_lift domains.

Expected pattern:
  Math:    high state_lift, NEGATIVE utt-trans correlation, LARGE PRM gap
  ESConv:  moderate state_lift, NEGATIVE utt-trans correlation, moderate PRM gap
  HH-RLHF: zero state_lift, POSITIVE utt-trans correlation, zero PRM gap
  Code:    small state_lift, ~zero utt-trans correlation, small PRM gap
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

print("=" * 70)
print("T10: MECHANISM ALIGNMENT ACROSS DOMAINS")
print("=" * 70)

results = {}

def compute_mechanism(z, a, zn, y_quality, name, task='classification'):
    """
    Compute all three mechanism components:
    1. state_lift: how much state helps predict quality
    2. utt_trans_corr: correlation between utterance quality and transition quality
    3. The anti-correlation pattern
    """
    n = len(z)

    # PCA
    d = min(D_PCA, n // 4)
    pca = PCA(n_components=d)
    pca.fit(np.vstack([z, a, zn]))
    zp = pca.transform(z)
    ap = pca.transform(a)
    znp = pca.transform(zn)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # --- 1. STATE_LIFT ---
    r2_action = cross_val_score(Ridge(alpha=1.0), ap, y_quality, cv=cv, scoring='r2').mean()
    X_sa = np.hstack([ap, zp, ap * zp])
    r2_sa = cross_val_score(Ridge(alpha=1.0), X_sa, y_quality, cv=cv, scoring='r2').mean()
    state_lift = r2_sa - max(r2_action, 0)

    # --- 2. UTTERANCE vs TRANSITION QUALITY ---
    # Utterance quality: how "good" the action looks in isolation
    # = action's distance from action centroid (popular actions are typical/good)
    # + action norm (longer/more detailed)
    a_centroid = ap.mean(0)
    utt_quality = -np.linalg.norm(ap - a_centroid, axis=1)  # closer to centroid = more typical = "better looking"

    # Transition quality: how much the state improves
    z_mean = zp.mean(0)
    trans_quality = np.linalg.norm(zp - z_mean, axis=1) - np.linalg.norm(znp - z_mean, axis=1)

    # Correlation between utterance quality and transition quality
    utt_trans_r, utt_trans_p = stats.pearsonr(utt_quality, trans_quality)

    # Also: correlation of each with the real quality label
    if task == 'regression':
        utt_label_r, utt_label_p = stats.pearsonr(utt_quality, y_quality)
        trans_label_r, trans_label_p = stats.pearsonr(trans_quality, y_quality)
    else:
        # Point-biserial for binary labels
        utt_label_r, utt_label_p = stats.pointbiserialr(y_quality, utt_quality)
        trans_label_r, trans_label_p = stats.pointbiserialr(y_quality, trans_quality)

    # --- 3. BLIND vs CONDITIONED PREDICTION ---
    # How well can action-only features predict quality?
    a_c = ap.mean(0)
    X_utt = np.hstack([ap, np.linalg.norm(ap, axis=1, keepdims=True),
                        np.linalg.norm(ap - a_c, axis=1, keepdims=True)])
    r2_utt = cross_val_score(Ridge(alpha=1.0), X_utt, y_quality, cv=cv, scoring='r2').mean()

    # How well can state+action predict quality?
    X_proc = np.hstack([zp, ap, zp * ap])
    r2_proc = cross_val_score(Ridge(alpha=1.0), X_proc, y_quality, cv=cv, scoring='r2').mean()

    print(f"\n  {name}: n={n}")
    print(f"    state_lift:           {state_lift:.4f}")
    print(f"    utt-trans corr:       r={utt_trans_r:.4f} (p={utt_trans_p:.2e})")
    print(f"    utt-label corr:       r={utt_label_r:.4f} (p={utt_label_p:.2e})")
    print(f"    trans-label corr:     r={trans_label_r:.4f} (p={trans_label_p:.2e})")
    print(f"    R2 action->quality:   {r2_utt:.4f}")
    print(f"    R2 process->quality:  {r2_proc:.4f}")

    return {
        'name': name, 'n': n,
        'state_lift': float(state_lift),
        'utt_trans_correlation': float(utt_trans_r),
        'utt_trans_p': float(utt_trans_p),
        'utt_label_correlation': float(utt_label_r),
        'trans_label_correlation': float(trans_label_r),
        'r2_action_quality': float(r2_utt),
        'r2_process_quality': float(r2_proc),
    }


# ===================================================================
# 1. MATH -- step-in-correct-chain vs wrong-context
# ===================================================================
print("\n[1] Math Reasoning (GSM8K)...")
ds = load_dataset('openai/gsm8k', 'main', split='train')
chains = []
for ex in list(ds)[:2000]:
    steps = [s.strip() for s in ex.get('answer', '').split('\n') if s.strip() and len(s.strip()) > 10]
    if len(steps) >= 3:
        chains.append({'question': ex.get('question', ''), 'steps': steps})

all_texts, chain_bounds = [], []
for chain in chains:
    s = len(all_texts)
    all_texts.append(chain['question'][:512])
    for step in chain['steps']:
        all_texts.append(step[:512])
    chain_bounds.append((s, len(all_texts)))

embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)

tz, ta, tzn, labels = [], [], [], []
np.random.seed(SEED)
for ci, (s, e) in enumerate(chain_bounds):
    n = e - s
    if n < 3: continue
    for i in range(1, n - 1):
        # Correct
        tz.append(embs[s+i-1]); ta.append(embs[s+i]); tzn.append(embs[s+i+1]); labels.append(1)
        # Wrong context
        other = np.random.randint(0, len(chain_bounds))
        while other == ci: other = np.random.randint(0, len(chain_bounds))
        os_, oe = chain_bounds[other]
        on = oe - os_
        if on >= 3:
            oi = np.random.randint(1, on-1)
            tz.append(embs[os_+oi-1]); ta.append(embs[s+i]); tzn.append(embs[os_+oi+1]); labels.append(0)

r = compute_mechanism(np.array(tz), np.array(ta), np.array(tzn), np.array(labels), "Math", task='classification')
r['llm_prm_gap'] = 0.474  # from E4
results['Math'] = r

# ===================================================================
# 2. ESConv -- therapeutic progress
# ===================================================================
print("\n[2] Emotional Support (ESConv)...")
ds_es = load_dataset('thu-coai/esconv', split='train')
convs = []
for ex in ds_es:
    parsed = json.loads(ex['text'])
    dialog = parsed.get('dialog', [])
    if len(dialog) >= 6:
        convs.append([(t.get('speaker', 'usr'), t['text']) for t in dialog])

all_texts, conv_bounds = [], []
for conv in convs[:2000]:
    s = len(all_texts)
    for spk, txt in conv:
        all_texts.append(txt[:512])
    conv_bounds.append((s, len(all_texts)))

embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)

tz, ta, tzn, quality = [], [], [], []
for s, e in conv_bounds:
    n = e - s
    if n < 6: continue
    target = embs[e - 2] if (e - s) % 2 == 0 else embs[e - 1]
    for i in range(2, n, 2):
        if s+i < len(embs):
            tz.append(embs[s+i-2]); ta.append(embs[s+i-1]); tzn.append(embs[s+i])
            quality.append(np.linalg.norm(embs[s+i-2] - target) - np.linalg.norm(embs[s+i] - target))

r = compute_mechanism(np.array(tz), np.array(ta), np.array(tzn), np.array(quality), "ESConv", task='regression')
r['llm_prm_gap'] = 0.017  # from E3
results['ESConv'] = r

# ===================================================================
# 3. HH-RLHF -- preference labels
# ===================================================================
print("\n[3] Chat QA (HH-RLHF)...")
ds_hh = load_dataset('Anthropic/hh-rlhf', split='train', data_dir='helpful-base')

hh_pairs = []
for ex in list(ds_hh)[:5000]:
    def parse_hh(text):
        turns = []
        for p in text.split('\n\n'):
            p = p.strip()
            if p.startswith('Human:'): turns.append(('human', p[6:].strip()))
            elif p.startswith('Assistant:'): turns.append(('assistant', p[10:].strip()))
        return turns
    chosen = parse_hh(ex['chosen'])
    rejected = parse_hh(ex['rejected'])
    if len(chosen) >= 4 and len(rejected) >= 4:
        prefix_len = 0
        for j in range(min(len(chosen), len(rejected))):
            if j < len(chosen) and j < len(rejected) and chosen[j] == rejected[j]:
                prefix_len = j + 1
            else: break
        if prefix_len >= 2:
            state_text = chosen[prefix_len-1][1][:512]
            c_asst = [t for r, t in chosen[prefix_len:] if r == 'assistant']
            r_asst = [t for r, t in rejected[prefix_len:] if r == 'assistant']
            if c_asst and r_asst:
                hh_pairs.append({'state': state_text, 'chosen': c_asst[0][:512], 'rejected': r_asst[0][:512]})

# For HH-RLHF, build z=state, a=action_diff, y=preference
state_texts = [p['state'] for p in hh_pairs]
chosen_texts = [p['chosen'] for p in hh_pairs]
rejected_texts = [p['rejected'] for p in hh_pairs]
all_hh = state_texts + chosen_texts + rejected_texts
embs_hh = encoder.encode(all_hh, batch_size=256, show_progress_bar=False)
n_p = len(hh_pairs)

# Build as: state, chosen_action, next_state_proxy (use chosen text as proxy)
# Quality label: 1 for chosen, 0 for rejected
tz, ta, tzn, labels = [], [], [], []
np.random.seed(SEED)
for i in range(n_p):
    # Chosen entry
    tz.append(embs_hh[i])             # state
    ta.append(embs_hh[n_p + i])       # chosen action
    tzn.append(embs_hh[n_p + i])      # proxy next state
    labels.append(1)
    # Rejected entry
    tz.append(embs_hh[i])             # same state
    ta.append(embs_hh[2*n_p + i])     # rejected action
    tzn.append(embs_hh[2*n_p + i])    # proxy next state
    labels.append(0)

r = compute_mechanism(np.array(tz), np.array(ta), np.array(tzn), np.array(labels), "HH-RLHF", task='classification')
r['llm_prm_gap'] = 0.034  # from E4
results['HH-RLHF'] = r

# ===================================================================
# 4. CODE -- correct vs incorrect solutions
# ===================================================================
print("\n[4] Code Reasoning (CodeContests)...")
ds_code = load_dataset('deepmind/code_contests', split='train')

tz, ta, tzn, labels = [], [], [], []
all_code = []
chain_bounds_code = []

for ex in list(ds_code)[:1000]:
    solutions = ex.get('solutions', {})
    incorrect = ex.get('incorrect_solutions', {})
    correct_sols = solutions.get('solution', []) if isinstance(solutions, dict) else []
    incorrect_sols = incorrect.get('solution', []) if isinstance(incorrect, dict) else []
    desc = str(ex.get('description', ''))[:512]

    for sol, label in [(s, 1) for s in correct_sols[:2]] + [(s, 0) for s in incorrect_sols[:2]]:
        if isinstance(sol, str) and len(sol) > 50:
            lines = sol.split('\n')
            blocks, current = [], []
            for line in lines:
                current.append(line)
                if len('\n'.join(current)) > 100 and (line.strip() == '' or line.strip().startswith('def ')):
                    blocks.append('\n'.join(current)); current = []
            if current: blocks.append('\n'.join(current))
            if len(blocks) >= 3:
                s_idx = len(all_code)
                all_code.append(desc)
                for b in blocks: all_code.append(b[:512])
                chain_bounds_code.append((s_idx, len(all_code), label))

embs_code = encoder.encode(all_code, batch_size=256, show_progress_bar=False)
for s, e, label in chain_bounds_code:
    n = e - s
    if n < 3: continue
    for i in range(1, n-1):
        tz.append(embs_code[s+i-1]); ta.append(embs_code[s+i]); tzn.append(embs_code[s+i+1]); labels.append(label)

r = compute_mechanism(np.array(tz), np.array(ta), np.array(tzn), np.array(labels), "Code", task='classification')
r['llm_prm_gap'] = 0.02  # literature: 1.5-2.6%
results['Code'] = r


# ===================================================================
# ALIGNMENT TABLE
# ===================================================================
print("\n" + "=" * 70)
print("MECHANISM ALIGNMENT TABLE")
print("=" * 70)

print(f"\n{'Domain':<12} {'state_lift':>11} {'utt-trans r':>12} {'utt-label r':>12} {'trans-label r':>14} {'LLM PRM gap':>12}")
print("-" * 75)
for name in ['Math', 'ESConv', 'HH-RLHF', 'Code']:
    r = results[name]
    print(f"  {name:<12} {r['state_lift']:>10.4f} {r['utt_trans_correlation']:>12.4f} "
          f"{r['utt_label_correlation']:>12.4f} {r['trans_label_correlation']:>14.4f} {r['llm_prm_gap']:>11.3f}")

print(f"""
MECHANISM CLAIM CHECK:
  For the story to hold, we need three things to align:

  1. High state_lift  -->  PRM needed (theory)
  2. Negative utt-trans correlation  -->  ORM optimizes wrong direction (mechanism)
  3. Large LLM PRM gap  -->  empirical confirmation

  Math:    state_lift={results['Math']['state_lift']:.3f}, utt-trans={results['Math']['utt_trans_correlation']:+.3f}, PRM_gap={results['Math']['llm_prm_gap']:.3f}
  ESConv:  state_lift={results['ESConv']['state_lift']:.3f}, utt-trans={results['ESConv']['utt_trans_correlation']:+.3f}, PRM_gap={results['ESConv']['llm_prm_gap']:.3f}
  HH-RLHF: state_lift={results['HH-RLHF']['state_lift']:.3f}, utt-trans={results['HH-RLHF']['utt_trans_correlation']:+.3f}, PRM_gap={results['HH-RLHF']['llm_prm_gap']:.3f}
  Code:    state_lift={results['Code']['state_lift']:.3f}, utt-trans={results['Code']['utt_trans_correlation']:+.3f}, PRM_gap={results['Code']['llm_prm_gap']:.3f}
""")

# Check: does negative utt-trans correlation predict PRM gap?
domains = list(results.keys())
utt_trans = [results[d]['utt_trans_correlation'] for d in domains]
prm_gaps = [results[d]['llm_prm_gap'] for d in domains]
state_lifts = [results[d]['state_lift'] for d in domains]

r_ut_prm, p_ut_prm = stats.pearsonr(utt_trans, prm_gaps)
r_sl_prm, p_sl_prm = stats.pearsonr(state_lifts, prm_gaps)
r_ut_sl, p_ut_sl = stats.pearsonr(utt_trans, state_lifts)

print(f"  Correlations (n={len(domains)}):")
print(f"    r(utt_trans, PRM_gap) = {r_ut_prm:.3f} (p={p_ut_prm:.3f})")
print(f"    r(state_lift, PRM_gap) = {r_sl_prm:.3f} (p={p_sl_prm:.3f})")
print(f"    r(utt_trans, state_lift) = {r_ut_sl:.3f} (p={p_ut_sl:.3f})")

# Save
out_path = os.path.join(OUT_DIR, 't10_mechanism_alignment.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
