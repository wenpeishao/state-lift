"""R14d: More math datasets with step labels."""
import numpy as np, json, sys, re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
sys.stdout.reconfigure(line_buffering=True)

SEED, D_PCA = 42, 16
np.random.seed(SEED)
OUT = 'results'
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def compute_sl(states, actions, ratings, name, max_n=15000):
    n = len(states)
    if n > max_n:
        np.random.seed(SEED); idx = np.random.choice(n, max_n, replace=False)
        states = [states[i] for i in idx]; actions = [actions[i] for i in idx]; ratings = [ratings[i] for i in idx]; n = max_n
    y = np.array(ratings, dtype=float)
    u, c = np.unique(y, return_counts=True)
    print(f"  [{name}] n={n}, labels={dict(zip(u.tolist(), c.tolist()))}")
    if len(u) <= 1: return {'status': 'single label'}
    se = encoder.encode([s[:1024] for s in states], batch_size=256, show_progress_bar=True)
    ae = encoder.encode([a[:512] for a in actions], batch_size=256, show_progress_bar=True)
    pca = PCA(n_components=D_PCA); pca.fit(np.vstack([se, ae]))
    z, a = pca.transform(se), pca.transform(ae)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2_a = cross_val_score(Ridge(alpha=1.0), a, y, cv=cv, scoring='r2').mean()
    X_sa = np.hstack([a, z, a * z])
    r2_sa = cross_val_score(Ridge(alpha=1.0), X_sa, y, cv=cv, scoring='r2').mean()
    sl = r2_sa - max(r2_a, 0)
    boot = []
    for bi in range(200):
        np.random.seed(SEED + bi); idx = np.random.choice(n, min(2000, n), replace=True)
        cv_b = KFold(n_splits=5, shuffle=True, random_state=SEED)
        r2ab = cross_val_score(Ridge(alpha=1.0), a[idx], y[idx], cv=cv_b, scoring='r2').mean()
        Xb = np.hstack([a[idx], z[idx], a[idx]*z[idx]])
        r2sb = cross_val_score(Ridge(alpha=1.0), Xb, y[idx], cv=cv_b, scoring='r2').mean()
        boot.append(r2sb - max(r2ab, 0))
    boot = np.array(boot); ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    # AUC
    y_bin = (y > 0).astype(float)
    auc_a, auc_sa = 0.5, 0.5
    if y_bin.sum() > 10 and (1-y_bin).sum() > 10:
        cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        try:
            auc_a = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), a, y_bin, cv=cv_s, scoring='roc_auc').mean()
            auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), X_sa, y_bin, cv=cv_s, scoring='roc_auc').mean()
        except: pass
    print(f"  SL={sl:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}] R2a={r2_a:.4f} R2sa={r2_sa:.4f} AUCa={auc_a:.4f} AUCsa={auc_sa:.4f}")
    return {'dataset': name, 'sl': float(sl), 'ci': [float(ci_lo), float(ci_hi)],
            'r2_a': float(r2_a), 'r2_sa': float(r2_sa), 'auc_a': float(auc_a), 'auc_sa': float(auc_sa)}


# ===================================================================
# 1. trl-lib/math_shepherd (clean format: prompt, completions[], labels[])
# ===================================================================
def load_trl_math_shepherd():
    print("\n[1] trl-lib/math_shepherd...")
    ds = load_dataset("trl-lib/math_shepherd", split="train", streaming=True)
    states, actions, ratings = [], [], []
    n_parsed = 0
    for ex in ds:
        if n_parsed >= 20000: break
        prompt = ex.get('prompt', '')
        completions = ex.get('completions', [])
        labels = ex.get('labels', [])
        if not prompt or not completions or not labels: continue
        n_steps = min(len(completions), len(labels))
        if n_steps < 2: continue
        for i in range(n_steps):
            if i == 0:
                state = prompt
            else:
                state = prompt + "\n" + "\n".join(completions[:i])
            states.append(state[:1024])
            actions.append(completions[i][:512])
            ratings.append(1 if labels[i] else -1)
        n_parsed += 1
    pos = sum(1 for r in ratings if r > 0)
    neg = sum(1 for r in ratings if r < 0)
    print(f"  {n_parsed} chains, {len(states)} steps, +{pos}/-{neg}")
    return states, actions, ratings


# ===================================================================
# 2. Qwen/ProcessBench (splits: gsm8k, math, olympiadbench, omnimath)
# ===================================================================
def load_processbench():
    print("\n[2] Qwen/ProcessBench...")
    all_states, all_actions, all_ratings = [], [], []

    for split_name in ['gsm8k', 'math']:
        try:
            ds = load_dataset("Qwen/ProcessBench", split=split_name, streaming=True)
        except Exception as e:
            print(f"  {split_name}: {e}")
            continue

        first = next(iter(ds))
        print(f"  {split_name} keys: {list(first.keys())}")
        for k, v in first.items():
            print(f"    {k}: {str(v)[:150]}")

        states, actions, ratings = [], [], []
        n_parsed = 0
        ds = load_dataset("Qwen/ProcessBench", split=split_name, streaming=True)
        for ex in ds:
            if n_parsed >= 10000: break

            problem = ex.get('problem', '') or ex.get('question', '')
            steps = ex.get('steps', None) or ex.get('solution', None)
            step_labels = ex.get('labels', None) or ex.get('step_labels', None)
            error_step = ex.get('error_step', None)

            if not problem: continue

            # Handle different formats
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split('\n') if s.strip()]
            elif isinstance(steps, list):
                steps = [str(s) for s in steps]

            if steps is None: continue

            # If we have error_step but no step_labels, create labels
            if step_labels is None and error_step is not None:
                step_labels = []
                for j in range(len(steps)):
                    if error_step == -1:  # no error
                        step_labels.append(1)
                    elif j < error_step:
                        step_labels.append(1)
                    elif j == error_step:
                        step_labels.append(-1)
                    else:
                        step_labels.append(-1)

            if step_labels is None: continue
            if isinstance(step_labels, (int, float)):
                step_labels = [step_labels]

            n_steps = min(len(steps), len(step_labels))
            if n_steps < 2: continue

            for i in range(n_steps):
                if i == 0:
                    state = problem
                else:
                    state = problem + "\n" + "\n".join(steps[:i])
                label = step_labels[i]
                if isinstance(label, bool):
                    label = 1 if label else -1
                elif isinstance(label, (int, float)):
                    label = 1 if label > 0 else -1
                states.append(state[:1024])
                actions.append(steps[i][:512])
                ratings.append(label)
            n_parsed += 1

        print(f"  {split_name}: {n_parsed} chains, {len(states)} steps")
        all_states.extend(states)
        all_actions.extend(actions)
        all_ratings.extend(ratings)

    print(f"  Total: {len(all_states)} steps")
    return all_states, all_actions, all_ratings


# ===================================================================
# 3. RLHFlow/Mistral-PRM-Data (conversations format with step labels)
# ===================================================================
def load_rlhflow():
    print("\n[3] RLHFlow/Mistral-PRM-Data...")
    ds = load_dataset("RLHFlow/Mistral-PRM-Data", split="train", streaming=True)

    states, actions, ratings = [], [], []
    n_parsed = 0

    for ex in ds:
        if n_parsed >= 15000: break
        convs = ex.get('conversations', [])
        if len(convs) < 2: continue

        # First message is usually the problem
        problem = convs[0].get('content', '')
        if not problem: continue

        # Subsequent messages are steps with labels
        step_texts = []
        step_labels = []
        for msg in convs[1:]:
            content = msg.get('content', '')
            role = msg.get('role', '')

            # Check if content has step + label markers
            # RLHFlow format: "Step N: ... +\n" or "Step N: ... -\n"
            if 'Step ' in content or role == 'assistant':
                # Look for +/- at end
                stripped = content.rstrip()
                if stripped.endswith('+'):
                    step_texts.append(stripped[:-1].strip())
                    step_labels.append(1)
                elif stripped.endswith('-'):
                    step_texts.append(stripped[:-1].strip())
                    step_labels.append(-1)
                else:
                    # Check for label in a separate field or embedded
                    label = msg.get('label', None)
                    if label is not None:
                        step_texts.append(content)
                        step_labels.append(1 if label > 0 else -1)
                    else:
                        step_texts.append(content)
                        step_labels.append(0)  # unknown

        valid = [(t, l) for t, l in zip(step_texts, step_labels) if l != 0]
        if len(valid) < 2: continue

        for i, (st, lb) in enumerate(valid):
            if i == 0:
                state = problem
            else:
                state = problem + "\n" + "\n".join([v[0] for v in valid[:i]])
            states.append(state[:1024])
            actions.append(st[:512])
            ratings.append(lb)
        n_parsed += 1

    pos = sum(1 for r in ratings if r > 0)
    neg = sum(1 for r in ratings if r < 0)
    print(f"  {n_parsed} chains, {len(states)} steps, +{pos}/-{neg}")
    return states, actions, ratings


# ===================================================================
# MAIN
# ===================================================================
def main():
    results = {}

    for name, loader in [('trl-MathShepherd', load_trl_math_shepherd),
                          ('ProcessBench', load_processbench),
                          ('RLHFlow-PRM', load_rlhflow)]:
        print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
        try:
            s, a, r = loader()
            if s and len(s) >= 100:
                res = compute_sl(s, a, r, name)
                results[name] = res
                with open(f'{OUT}/r14d_{name.lower().replace("-","_")}.json', 'w') as f:
                    json.dump(res, f, indent=2)
            else:
                print(f"  SKIPPED: {0 if s is None else len(s)} samples")
                results[name] = {'status': 'skipped'}
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'status': 'error', 'error': str(e)}

    # Summary
    print("\n\n" + "=" * 80)
    print("R14d: MATH DATASETS WITH REAL STEP LABELS")
    print("=" * 80)
    baselines = {'PRM800K (r2)': 0.015, 'MathShepherd (r12c)': -0.002}
    print(f"\n{'Dataset':<25} {'SL':<10} {'CI':<25} {'AUC_a':<8} {'AUC_sa':<8}")
    print("-" * 76)
    for nm, sl in baselines.items():
        print(f"{nm:<25} {sl:<10.4f} {'(prior)':<25}")
    for nm, res in results.items():
        if 'sl' in res:
            ci = res.get('ci', [None, None])
            cis = f"[{ci[0]:.4f},{ci[1]:.4f}]" if ci[0] is not None else "N/A"
            print(f"{nm:<25} {res['sl']:<10.4f} {cis:<25} {res.get('auc_a',0):<8.4f} {res.get('auc_sa',0):<8.4f}")

    with open(f'{OUT}/r14d_combined.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved.")

if __name__ == '__main__':
    main()
