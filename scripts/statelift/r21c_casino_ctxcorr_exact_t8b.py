"""
R21: CaSiNo context-correctness SL under grouped CV (t8b replication + split test).
Completes the corrected Table 2 story: for BOTH negotiation domains we then have
  - genuine deal-outcome SL (r19/r19b: CaSiNo ~0.00, DoND ~0.06)
  - synthetic context-correctness SL (t8b CaSiNo: 0.22; DoND: this script)
Construction (t8b): for each turn, (prev-turn state, action, label=1) plus
(random other-dialogue state, same action, label=0). SL = AUC(s+a) - AUC(a),
StratifiedKFold logistic regression, MiniLM + PCA16. Also reports the
GroupKFold-by-dialogue variant.
"""

import numpy as np
import re, json
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sentence_transformers import SentenceTransformer

SEED = 42
D_PCA = 16
N_MAX_DIALOGUES = 3000
OUT = "results/r21c_casino_ctxcorr_exact_t8b.json"
np.random.seed(SEED)

enc = SentenceTransformer('all-MiniLM-L6-v2')

# ---- load CaSiNo dialogues (t8b construction) ----
from datasets import load_dataset
ds = load_dataset('casino', split='train')
convs = []
for ex in ds:
    chat = ex.get('chat_logs', [])
    if isinstance(chat, list) and len(chat) >= 4:
        conv = [str(t.get('text', ''))[:512] for t in chat
                if isinstance(t, dict) and t.get('text')
                ]
        if len(conv) >= 4:
            convs.append(conv)

print(f"{len(convs)} dialogues")
if len(convs) > N_MAX_DIALOGUES:
    idx = np.random.choice(len(convs), N_MAX_DIALOGUES, replace=False)
    convs = [convs[i] for i in idx]
    print(f"sampled {len(convs)}")

# ---- t8b-style pair construction ----
tz_texts, ta_texts, labels, groups = [], [], [], []
for ci, conv in enumerate(convs):
    for i in range(1, len(conv) - 1):
        state_text = conv[i-1][:512]
        action_text = conv[i][:512]
        # correct context
        tz_texts.append(state_text); ta_texts.append(action_text)
        labels.append(1); groups.append(ci)
        # wrong context (random other dialogue, random turn)
        other_ci = np.random.randint(0, len(convs))
        while other_ci == ci:
            other_ci = np.random.randint(0, len(convs))
        other = convs[other_ci]
        wrong_state = other[np.random.randint(0, len(other))][:512]
        tz_texts.append(wrong_state); ta_texts.append(action_text)
        labels.append(0); groups.append(ci)

y = np.array(labels)
groups = np.array(groups)
print(f"{len(y)} pairs")

print("embedding...")
ez = enc.encode(tz_texts, batch_size=128, show_progress_bar=False)
ea = enc.encode(ta_texts, batch_size=128, show_progress_bar=False)
pca = PCA(n_components=D_PCA, random_state=SEED)
pca.fit(np.vstack([ez, ea]))
z = pca.transform(ez)
a = pca.transform(ea)
Xsa = np.hstack([z, a])

results = {}
for mode in ['stratified_random', 'grouped']:
    if mode == 'stratified_random':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        auc_a = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), a, y, cv=cv, scoring='roc_auc').mean()
        auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), Xsa, y, cv=cv, scoring='roc_auc').mean()
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        auc_a = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), a, y, cv=cv, groups=groups, scoring='roc_auc').mean()
        auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), Xsa, y, cv=cv, groups=groups, scoring='roc_auc').mean()
    sl = auc_sa - auc_a
    results[mode] = {'auc_action': float(auc_a), 'auc_state_action': float(auc_sa), 'state_lift_auc': float(sl)}
    print(f"  {mode:18s}: AUC(a)={auc_a:.4f} AUC(s+a)={auc_sa:.4f} SL={sl:+.4f}")

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved {OUT}")
