"""
R11: State-lift using frozen LLM hidden states as features.

Tests the resolution hypothesis: SL_MiniLM(PRM800K) = 0.018 but LLM lift = +0.238.
If SL_Llama8B >> SL_MiniLM, state-dependence lives at a resolution sentence
embeddings can't see. SL becomes jointly (domain, encoder), not just domain.

No training -- just frozen forward passes + Ridge regression.
"""

import torch
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

SEED = 42
D_PCA = 16
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = os.environ.get("LLAMA_PATH", "meta-llama/Llama-3.1-8B-Instruct")
DATA_DIR = "data/PRM800K"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===================================================================
# 1. PARSE PRM800K (same as r2)
# ===================================================================
print("\n[1/6] Parsing PRM800K...")
t0 = time.time()

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

# Build transitions
states, actions, ratings = [], [], []
for chain in step_data:
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

print(f"  {len(states)} transitions from {len(step_data)} chains")
print(f"  +1={ratings.count(1)}, 0={ratings.count(0)}, -1={ratings.count(-1)}")
print(f"  Parse time: {time.time()-t0:.1f}s")

# ===================================================================
# 2. LOAD FROZEN LLM
# ===================================================================
print("\n[2/6] Loading Llama-3.1-8B-Instruct (frozen)...")
t0 = time.time()

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()

print(f"  Model loaded in {time.time()-t0:.1f}s")
print(f"  Hidden dim: {model.config.hidden_size}")

# ===================================================================
# 3. EXTRACT HIDDEN STATES
# ===================================================================
print("\n[3/6] Extracting hidden states...")

BATCH_SIZE = 16
MAX_LEN = 512  # must capture full state text (1024 chars ≈ 250-400 tokens)


@torch.no_grad()
def get_hidden_states(texts, batch_size=BATCH_SIZE):
    """Mean-pool last-layer hidden states over non-padding tokens."""
    all_hidden = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=False)
        # Last hidden state: (batch, seq_len, hidden_dim)
        hidden = out.last_hidden_state

        # Mean pool over non-padding tokens
        mask = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_hidden.append(pooled.float().cpu().numpy())

        if (start // batch_size) % 200 == 0:
            print(f"    {start + len(batch)}/{len(texts)} "
                  f"({100*(start+len(batch))/len(texts):.0f}%)")

    return np.vstack(all_hidden)


# Subsample for speed if needed (78K * 2 passes is ~10K batches)
MAX_SAMPLES = 20000
if len(states) > MAX_SAMPLES:
    np.random.seed(SEED)
    idx = np.random.choice(len(states), MAX_SAMPLES, replace=False)
    states_sub = [states[i] for i in idx]
    actions_sub = [actions[i] for i in idx]
    ratings_sub = [ratings[i] for i in idx]
    print(f"  Subsampled {MAX_SAMPLES} from {len(states)}")
else:
    states_sub = states
    actions_sub = actions
    ratings_sub = ratings
    idx = np.arange(len(states))

t0 = time.time()
print(f"  Encoding {len(states_sub)} states...")
state_hidden = get_hidden_states(states_sub)
print(f"  States done in {time.time()-t0:.1f}s, shape: {state_hidden.shape}")

t1 = time.time()
print(f"  Encoding {len(actions_sub)} actions...")
action_hidden = get_hidden_states(actions_sub)
print(f"  Actions done in {time.time()-t1:.1f}s, shape: {action_hidden.shape}")

# Free GPU memory
del model
torch.cuda.empty_cache()

# ===================================================================
# 4. PCA + STATE-LIFT (LLM features)
# ===================================================================
print("\n[4/6] Computing state-lift with LLM features...")

y = np.array(ratings_sub, dtype=float)

pca = PCA(n_components=D_PCA)
pca.fit(np.vstack([state_hidden, action_hidden]))
z_llm = pca.transform(state_hidden)
a_llm = pca.transform(action_hidden)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Action only
r2_act = cross_val_score(Ridge(alpha=1.0), a_llm, y, cv=cv, scoring='r2').mean()
# State + action + interaction
X_sa = np.hstack([a_llm, z_llm, a_llm * z_llm])
r2_sa = cross_val_score(Ridge(alpha=1.0), X_sa, y, cv=cv, scoring='r2').mean()
sl_llm = r2_sa - max(r2_act, 0)

print(f"  R2 action:    {r2_act:.4f}")
print(f"  R2 s+a+i:     {r2_sa:.4f}")
print(f"  SL (LLM):     {sl_llm:.4f}")

# Also try more PCA dims
for d in [32, 64, 128]:
    pca_d = PCA(n_components=d)
    pca_d.fit(np.vstack([state_hidden, action_hidden]))
    z_d = pca_d.transform(state_hidden)
    a_d = pca_d.transform(action_hidden)
    X_d = np.hstack([a_d, z_d, a_d * z_d])
    r2_act_d = cross_val_score(Ridge(alpha=1.0), a_d, y, cv=cv, scoring='r2').mean()
    r2_sa_d = cross_val_score(Ridge(alpha=1.0), X_d, y, cv=cv, scoring='r2').mean()
    sl_d = r2_sa_d - max(r2_act_d, 0)
    print(f"  D={d:3d}: R2_act={r2_act_d:.4f}, R2_sa={r2_sa_d:.4f}, SL={sl_d:.4f}, "
          f"var_explained={pca_d.explained_variance_ratio_.sum():.4f}")

# ===================================================================
# 5. BINARY CLASSIFICATION (LLM features)
# ===================================================================
print("\n[5/6] Binary classification with LLM features...")
mask_binary = np.array(ratings_sub) != 0
z_bin = z_llm[mask_binary]
a_bin = a_llm[mask_binary]
y_bin = (np.array(ratings_sub)[mask_binary] > 0).astype(float)
X_sa_bin = np.hstack([a_bin, z_bin, a_bin * z_bin])

cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
auc_act = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                           a_bin, y_bin, cv=cv_s, scoring='roc_auc').mean()
auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                          X_sa_bin, y_bin, cv=cv_s, scoring='roc_auc').mean()
sl_auc_llm = auc_sa - auc_act

print(f"  Binary: {mask_binary.sum()} samples ({y_bin.sum():.0f} pos)")
print(f"  AUC action:    {auc_act:.4f}")
print(f"  AUC s+a+i:     {auc_sa:.4f}")
print(f"  SL_AUC (LLM):  {sl_auc_llm:.4f}")

# ===================================================================
# 6. COMPARISON
# ===================================================================
print("\n" + "=" * 70)
print("RESOLUTION HYPOTHESIS TEST")
print("=" * 70)

# From r9 results
sl_miniml_r2 = 0.015
sl_miniml_auc = 0.071

print(f"""
  MiniLM (384d, PCA-16):
    SL (R2):   {sl_miniml_r2:.4f}
    SL (AUC):  {sl_miniml_auc:.4f}

  Llama-8B frozen (4096d, PCA-16):
    SL (R2):   {sl_llm:.4f}  ({sl_llm/max(sl_miniml_r2, 0.001):.1f}x MiniLM)
    SL (AUC):  {sl_auc_llm:.4f}  ({sl_auc_llm/max(sl_miniml_auc, 0.001):.1f}x MiniLM)

  LLM training lift (from r2):  +0.238 AUC

  INTERPRETATION:
    {'RESOLUTION CONFIRMED: LLM features recover state-dependence that MiniLM misses.' if sl_llm > 0.05 else ''}
    {'ATTENTION HYPOTHESIS: Signal lives in cross-attention, not representations.' if sl_llm < 0.03 else ''}
    {'PARTIAL: LLM features help but gap remains -- both resolution and attention matter.' if 0.03 <= sl_llm <= 0.05 else ''}
""")

# Save
out = {
    'n_samples': len(states_sub),
    'n_chains': len(step_data),
    'llm_model': 'Llama-3.1-8B-Instruct',
    'llm_hidden_dim': 4096,
    'pca_dim': D_PCA,
    'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
    'llm_features': {
        'r2_action': float(r2_act),
        'r2_state_action': float(r2_sa),
        'sl_r2': float(sl_llm),
        'auc_action': float(auc_act),
        'auc_state_action': float(auc_sa),
        'sl_auc': float(sl_auc_llm),
    },
    'miniml_baseline': {
        'sl_r2': sl_miniml_r2,
        'sl_auc': sl_miniml_auc,
    },
    'resolution_ratio_r2': float(sl_llm / max(sl_miniml_r2, 0.001)),
    'resolution_ratio_auc': float(sl_auc_llm / max(sl_miniml_auc, 0.001)),
    'resolution_confirmed': bool(sl_llm > 0.05),
}

out_path = os.path.join(OUT_DIR, 'r11_llm_feature_sl.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved to {out_path}")
