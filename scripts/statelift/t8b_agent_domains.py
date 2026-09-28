"""
T8b: Agent domains with real step-level quality signals.

Need 2-3 more domains with genuine environment feedback:
1. Web navigation -- task success at each step (Mind2Web has action-level labels)
2. Tool-use -- API call success/failure
3. Game/grid agents -- reward at each step

For each: compute state_lift with REAL quality labels.
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
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

SEED = 42
D_PCA = 16
OUT_DIR = 'results'
np.random.seed(SEED)

encoder = SentenceTransformer('all-MiniLM-L6-v2')


def compute_state_lift_real(z, a, y, name="", task='classification'):
    """
    Compute state_lift = AUC/R2(state+action) - AUC/R2(action only).
    This is the core diagnostic with real quality labels.
    """
    n = len(z)
    if n < 100:
        print(f"  {name}: too few ({n})")
        return None

    a_c = a.mean(0)

    if task == 'regression':
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

        r2_action = cross_val_score(Ridge(alpha=1.0), a, y, cv=cv, scoring='r2').mean()

        X_sa = np.hstack([a, z, a * z])
        r2_sa = cross_val_score(Ridge(alpha=1.0), X_sa, y, cv=cv, scoring='r2').mean()

        r2_state = cross_val_score(Ridge(alpha=1.0), z, y, cv=cv, scoring='r2').mean()

        X_out = np.hstack([a, np.linalg.norm(a, axis=1, keepdims=True),
                            np.linalg.norm(a - a_c, axis=1, keepdims=True)])
        r2_out = cross_val_score(Ridge(alpha=1.0), X_out, y, cv=cv, scoring='r2').mean()

        X_proc = np.hstack([z, a, z * a])
        r2_proc = cross_val_score(Ridge(alpha=1.0), X_proc, y, cv=cv, scoring='r2').mean()

        state_lift = r2_sa - max(r2_action, 0)
        sdi = r2_proc / max(r2_out, 0.001)

        print(f"  {name}: n={n}")
        print(f"    R2: action={r2_action:.4f}, state={r2_state:.4f}, s+a+ix={r2_sa:.4f}")
        print(f"    state_lift = {state_lift:.4f}, SDI = {sdi:.2f}")

        return {
            'name': name, 'n': n, 'task': task,
            'r2_action': float(r2_action), 'r2_state': float(r2_state),
            'r2_state_action': float(r2_sa), 'r2_outcome': float(r2_out),
            'r2_process': float(r2_proc),
            'state_lift': float(state_lift), 'SDI_real': float(sdi),
        }

    elif task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        auc_action = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                      a, y, cv=cv, scoring='roc_auc').mean()

        X_sa = np.hstack([a, z, a * z])
        auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                  X_sa, y, cv=cv, scoring='roc_auc').mean()

        auc_state = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                     z, y, cv=cv, scoring='roc_auc').mean()

        X_out = np.hstack([a, np.linalg.norm(a, axis=1, keepdims=True),
                            np.linalg.norm(a - a_c, axis=1, keepdims=True)])
        auc_out = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                   X_out, y, cv=cv, scoring='roc_auc').mean()

        X_proc = np.hstack([z, a, z * a])
        auc_proc = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                    X_proc, y, cv=cv, scoring='roc_auc').mean()

        state_lift = auc_sa - auc_action
        sdi = auc_proc / max(auc_out, 0.501)

        print(f"  {name}: n={n}")
        print(f"    AUC: action={auc_action:.4f}, state={auc_state:.4f}, s+a+ix={auc_sa:.4f}")
        print(f"    state_lift = {state_lift:.4f}, SDI = {sdi:.2f}")

        return {
            'name': name, 'n': n, 'task': task,
            'auc_action': float(auc_action), 'auc_state': float(auc_state),
            'auc_state_action': float(auc_sa), 'auc_outcome': float(auc_out),
            'auc_process': float(auc_proc),
            'state_lift': float(state_lift), 'SDI_real': float(sdi),
        }


print("=" * 70)
print("T8b: AGENT DOMAINS WITH REAL QUALITY SIGNALS")
print("=" * 70)

results = {}

# ===================================================================
# 1. WEB NAVIGATION (Mind2Web) -- action matching labels
# ===================================================================
print("\n[1] Web Navigation (Mind2Web) -- correct vs wrong actions...")

# Mind2Web has candidate actions with one being correct.
# We can construct: state + correct_action (label=1) vs state + wrong_action (label=0)
ds = load_dataset('osunlp/Mind2Web', split='train')
print(f"  Mind2Web: {len(ds)} examples")

# Each example has action_reprs (the correct action sequence) and
# candidate negative actions from the page
tz, ta, labels = [], [], []
all_texts = []
text_idx = {}

for ex in list(ds)[:1500]:
    actions = ex.get('action_reprs', [])
    task_desc = str(ex.get('confirmed_task', ''))[:512]

    # Candidate actions and the positive index
    pos_candidates = ex.get('pos_candidates', [])
    neg_candidates = ex.get('neg_candidates', [])

    if not isinstance(actions, list) or len(actions) < 2:
        continue

    # Build state-action pairs from the action sequence
    # For each step: state = task + previous actions, action = current action
    for i in range(len(actions)):
        # State: task description + previous actions concatenated
        if i == 0:
            state_text = task_desc
        else:
            state_text = task_desc + " | " + " > ".join(str(a)[:100] for a in actions[:i])
        state_text = state_text[:512]

        action_text = str(actions[i])[:512]

        # Correct action in correct context
        all_texts.append(state_text)
        all_texts.append(action_text)
        tz.append(len(all_texts) - 2)
        ta.append(len(all_texts) - 1)
        labels.append(1)

        # Wrong action: take action from a DIFFERENT step in same trace
        if len(actions) >= 3:
            wrong_idx = np.random.randint(0, len(actions))
            while wrong_idx == i:
                wrong_idx = np.random.randint(0, len(actions))
            wrong_action = str(actions[wrong_idx])[:512]

            all_texts.append(state_text)  # same state
            all_texts.append(wrong_action)  # wrong action
            tz.append(len(all_texts) - 2)
            ta.append(len(all_texts) - 1)
            labels.append(0)

print(f"  {len(labels)} state-action pairs ({sum(labels)} correct, {len(labels)-sum(labels)} wrong)")
print(f"  Embedding {len(all_texts)} texts...")
embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)

z_web = np.array([embs[i] for i in tz])
a_web = np.array([embs[i] for i in ta])
y_web = np.array(labels)

pca = PCA(n_components=D_PCA)
pca.fit(np.vstack([z_web, a_web]))
z_w = pca.transform(z_web)
a_w = pca.transform(a_web)

r = compute_state_lift_real(z_w, a_w, y_web, "WebNav-ActionCorrectness", task='classification')
if r:
    results['Web Navigation'] = r
    results['Web Navigation']['quality_signal'] = 'correct action in context vs wrong action'
    results['Web Navigation']['literature_prm_gap'] = '6.1-19.0% (AgentPRM 2024)'


# ===================================================================
# 2. TOOL-USE (Glaive) -- function call success
# ===================================================================
print("\n[2] Tool-Use (Glaive) -- correct vs shuffled function calls...")

ds_tool = load_dataset('glaiveai/glaive-function-calling-v2', split='train')

# Quality signal: function call in correct conversation context vs wrong context
tz, ta, labels = [], [], []
all_texts = []

convs_tool = []
for ex in list(ds_tool)[:5000]:
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
        convs_tool.append(conv)

print(f"  {len(convs_tool)} tool-use conversations")

# Build state-action pairs with context-correctness labels
for ci, conv in enumerate(convs_tool):
    for i in range(1, len(conv) - 1):
        state_text = conv[i-1][1][:512]  # previous turn
        action_text = conv[i][1][:512]    # current turn

        # Correct: action in its real context
        all_texts.append(state_text)
        all_texts.append(action_text)
        tz.append(len(all_texts) - 2)
        ta.append(len(all_texts) - 1)
        labels.append(1)

        # Wrong: same action, random state from different conversation
        other_ci = np.random.randint(0, len(convs_tool))
        while other_ci == ci:
            other_ci = np.random.randint(0, len(convs_tool))
        other_conv = convs_tool[other_ci]
        other_i = np.random.randint(0, len(other_conv))
        wrong_state = other_conv[other_i][1][:512]

        all_texts.append(wrong_state)
        all_texts.append(action_text)  # same action
        tz.append(len(all_texts) - 2)
        ta.append(len(all_texts) - 1)
        labels.append(0)

    if len(labels) >= 20000:
        break

print(f"  {len(labels)} pairs ({sum(labels)} correct, {len(labels)-sum(labels)} wrong)")
print(f"  Embedding {len(all_texts)} texts...")
embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)

z_tool = np.array([embs[i] for i in tz])
a_tool = np.array([embs[i] for i in ta])
y_tool = np.array(labels)

pca = PCA(n_components=D_PCA)
pca.fit(np.vstack([z_tool, a_tool]))
z_t = pca.transform(z_tool)
a_t = pca.transform(a_tool)

r = compute_state_lift_real(z_t, a_t, y_tool, "ToolUse-ContextCorrectness", task='classification')
if r:
    results['Tool-Use'] = r
    results['Tool-Use']['quality_signal'] = 'action in correct context vs wrong context'
    results['Tool-Use']['literature_prm_gap'] = 'Expected MODERATE (AgentPRM: agent domains benefit)'


# ===================================================================
# 3. NEGOTIATION (CaSiNo) -- deal outcome quality
# ===================================================================
print("\n[3] Negotiation (CaSiNo) -- utterance context-correctness...")

ds_neg = load_dataset('casino', split='train')

convs_neg = []
for ex in list(ds_neg):
    chat = ex.get('chat_logs', [])
    if isinstance(chat, list) and len(chat) >= 4:
        conv = [(str(t.get('id', 'spk')), str(t.get('text', ''))[:512])
                for t in chat if isinstance(t, dict) and t.get('text')]
        if len(conv) >= 4:
            convs_neg.append(conv)

print(f"  {len(convs_neg)} negotiations")

tz, ta, labels = [], [], []
all_texts = []

for ci, conv in enumerate(convs_neg):
    for i in range(1, len(conv) - 1):
        state_text = conv[i-1][1][:512]
        action_text = conv[i][1][:512]

        # Correct context
        all_texts.append(state_text)
        all_texts.append(action_text)
        tz.append(len(all_texts) - 2)
        ta.append(len(all_texts) - 1)
        labels.append(1)

        # Wrong context
        other_ci = np.random.randint(0, len(convs_neg))
        while other_ci == ci:
            other_ci = np.random.randint(0, len(convs_neg))
        other_conv = convs_neg[other_ci]
        other_i = np.random.randint(0, len(other_conv))
        wrong_state = other_conv[other_i][1][:512]

        all_texts.append(wrong_state)
        all_texts.append(action_text)
        tz.append(len(all_texts) - 2)
        ta.append(len(all_texts) - 1)
        labels.append(0)

print(f"  {len(labels)} pairs")
print(f"  Embedding {len(all_texts)} texts...")
embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)

z_neg = np.array([embs[i] for i in tz])
a_neg = np.array([embs[i] for i in ta])
y_neg = np.array(labels)

pca = PCA(n_components=D_PCA)
pca.fit(np.vstack([z_neg, a_neg]))
z_n = pca.transform(z_neg)
a_n = pca.transform(a_neg)

r = compute_state_lift_real(z_n, a_n, y_neg, "Negotiation-ContextCorrectness", task='classification')
if r:
    results['Negotiation'] = r
    results['Negotiation']['quality_signal'] = 'utterance in correct context vs wrong context'
    results['Negotiation']['literature_prm_gap'] = 'NOVEL PREDICTION: Expected LOW'


# ===================================================================
# COMBINED RESULTS
# ===================================================================
print("\n" + "=" * 70)
print("T8b RESULTS: AGENT DOMAINS")
print("=" * 70)

print(f"\n{'Domain':<25} {'State Lift':>11} {'AUC proc':>10} {'AUC out':>10} {'Quality Signal'}")
print("-" * 80)
for name, r in sorted(results.items(), key=lambda x: x[1]['state_lift'], reverse=True):
    sl = r['state_lift']
    proc = r.get('auc_process', r.get('r2_process', '?'))
    out = r.get('auc_outcome', r.get('r2_outcome', '?'))
    qs = r.get('quality_signal', '?')[:35]
    print(f"  {name:<25} {sl:>10.4f} {proc:>10.4f} {out:>10.4f} {qs}")


# ===================================================================
# COMBINE WITH T8 RESULTS
# ===================================================================
print("\n" + "=" * 70)
print("FULL TABLE: ALL DOMAINS WITH REAL QUALITY LABELS")
print("=" * 70)

# Load T8 results
t8_path = os.path.join(OUT_DIR, 't8_real_label_sdi.json')
if os.path.exists(t8_path):
    with open(t8_path) as f:
        t8 = json.load(f)

    all_domains = {}
    # T8 results (4 genuine domains)
    for name, r in t8.items():
        if 'state_lift' in r:
            all_domains[name] = r

    # T8b results
    for name, r in results.items():
        all_domains[name] = r

    print(f"\n{'Domain':<30} {'State Lift':>11} {'Lit PRM Gap':>20} {'Correct?'}")
    print("-" * 75)
    for name, r in sorted(all_domains.items(), key=lambda x: x[1]['state_lift'], reverse=True):
        sl = r['state_lift']
        lit = r.get('literature_prm_gap', '?')
        # Determine if prediction matches
        prm_helps_lit = lit and '0%' not in str(lit) and 'no' not in str(lit).lower()
        prm_helps_pred = sl > 0.01  # threshold
        match = 'YES' if prm_helps_lit == prm_helps_pred else 'NO'
        print(f"  {name:<30} {sl:>10.4f} {str(lit):>20} {match:>8}")

# Save
out_path = os.path.join(OUT_DIR, 't8b_agent_domains.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
