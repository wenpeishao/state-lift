"""
R12c: Proper MathShepherd parsing + PRM800K all-completions.

Key discovery: MathShepherd label format:
  - input:  "...step text ки\\nStep 2:..."
  - label:  "...step text +\\nStep 2:..."  or "...step text -\\nStep 2:..."
  Labels are embedded by replacing ки markers with + or - in the label field.

Also: re-try PRM800K with ALL completions at each step (not just chosen).
"""

import numpy as np
import json
import os
import re
import warnings
import time

warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

SEED = 42
D_PCA = 16
np.random.seed(SEED)

OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

encoder = SentenceTransformer('all-MiniLM-L6-v2')


def compute_state_lift(states, actions, ratings, dataset_name, max_samples=20000):
    """Compute state-lift (same methodology as r2)."""
    print(f"\n{'='*70}")
    print(f"Computing state-lift for: {dataset_name}")
    print(f"{'='*70}")

    n = len(states)
    if n > max_samples:
        np.random.seed(SEED)
        idx = np.random.choice(n, max_samples, replace=False)
        states = [states[i] for i in idx]
        actions = [actions[i] for i in idx]
        ratings = [ratings[i] for i in idx]
        n = max_samples

    y = np.array(ratings, dtype=float)
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Samples: {n}")
    print(f"  Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    if len(unique) <= 1:
        print("  ERROR: All same label")
        return {'dataset': dataset_name, 'status': 'error', 'reason': 'all same label'}

    states_t = [s[:1024] for s in states]
    actions_t = [a[:512] for a in actions]

    print("  Embedding states...")
    state_embs = encoder.encode(states_t, batch_size=256, show_progress_bar=True)
    print("  Embedding actions...")
    action_embs = encoder.encode(actions_t, batch_size=256, show_progress_bar=True)

    pca = PCA(n_components=D_PCA)
    pca.fit(np.vstack([state_embs, action_embs]))
    z = pca.transform(state_embs)
    a = pca.transform(action_embs)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    r2_action = cross_val_score(Ridge(alpha=1.0), a, y, cv=cv, scoring='r2').mean()
    r2_state = cross_val_score(Ridge(alpha=1.0), z, y, cv=cv, scoring='r2').mean()
    X_sa = np.hstack([a, z, a * z])
    r2_sa = cross_val_score(Ridge(alpha=1.0), X_sa, y, cv=cv, scoring='r2').mean()

    a_c = a.mean(0)
    X_out = np.hstack([a, np.linalg.norm(a, axis=1, keepdims=True),
                        np.linalg.norm(a - a_c, axis=1, keepdims=True)])
    r2_outcome = cross_val_score(Ridge(alpha=1.0), X_out, y, cv=cv, scoring='r2').mean()
    X_proc = np.hstack([z, a, z * a])
    r2_process = cross_val_score(Ridge(alpha=1.0), X_proc, y, cv=cv, scoring='r2').mean()

    state_lift = r2_sa - max(r2_action, 0)
    sdi = r2_process / max(r2_outcome, 0.001)

    # Bootstrap
    boot_sls = []
    for bi in range(200):
        np.random.seed(SEED + bi)
        bidx = np.random.choice(n, min(2000, n), replace=True)
        cv_b = KFold(n_splits=5, shuffle=True, random_state=SEED)
        r2_a_b = cross_val_score(Ridge(alpha=1.0), a[bidx], y[bidx], cv=cv_b, scoring='r2').mean()
        X_b = np.hstack([a[bidx], z[bidx], a[bidx]*z[bidx]])
        r2_sa_b = cross_val_score(Ridge(alpha=1.0), X_b, y[bidx], cv=cv_b, scoring='r2').mean()
        boot_sls.append(r2_sa_b - max(r2_a_b, 0))
    boot_sls = np.array(boot_sls)
    ci_lo, ci_hi = np.percentile(boot_sls, [2.5, 97.5])

    # AUC
    auc_results = None
    y_bin = (y > 0).astype(float)
    if y_bin.sum() > 10 and (1 - y_bin).sum() > 10:
        cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        try:
            auc_a = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                     a, y_bin, cv=cv_s, scoring='roc_auc').mean()
            auc_sa = cross_val_score(LogisticRegression(max_iter=1000, C=0.1),
                                      X_sa, y_bin, cv=cv_s, scoring='roc_auc').mean()
            auc_results = {
                'auc_action': float(auc_a),
                'auc_state_action': float(auc_sa),
                'state_lift_auc': float(auc_sa - auc_a),
                'n_positive': int(y_bin.sum()),
                'n_negative': int((1 - y_bin).sum()),
            }
        except:
            pass

    # Context-correctness
    n_cc = min(5000, len(state_embs))
    np.random.seed(SEED)
    tz_cc, ta_cc, labels_cc = [], [], []
    for i in range(n_cc):
        tz_cc.append(state_embs[i]); ta_cc.append(action_embs[i]); labels_cc.append(1)
        other = np.random.randint(0, len(state_embs))
        tz_cc.append(state_embs[other]); ta_cc.append(action_embs[i]); labels_cc.append(0)

    z_cc = np.array(tz_cc); a_cc = np.array(ta_cc); y_cc = np.array(labels_cc, dtype=float)
    pca_cc = PCA(n_components=D_PCA)
    pca_cc.fit(np.vstack([z_cc, a_cc]))
    zp = pca_cc.transform(z_cc); ap = pca_cc.transform(a_cc)
    r2_a_cc = cross_val_score(Ridge(alpha=1.0), ap, y_cc, cv=cv, scoring='r2').mean()
    X_cc = np.hstack([ap, zp, ap * zp])
    r2_sa_cc = cross_val_score(Ridge(alpha=1.0), X_cc, y_cc, cv=cv, scoring='r2').mean()
    sl_cc = r2_sa_cc - max(r2_a_cc, 0)

    results = {
        'dataset': dataset_name, 'n_samples': n,
        'r2_action': float(r2_action), 'r2_state': float(r2_state),
        'r2_state_action': float(r2_sa), 'r2_outcome': float(r2_outcome),
        'r2_process': float(r2_process), 'state_lift': float(state_lift),
        'sdi': float(sdi), 'bootstrap_ci': [float(ci_lo), float(ci_hi)],
        'context_correctness_sl': float(sl_cc),
        'ratio_real_vs_cc': float(state_lift / max(sl_cc, 0.001)),
    }
    if auc_results:
        results['binary_analysis'] = auc_results

    print(f"\n  State-lift (real):   {state_lift:.4f}  CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  State-lift (CC):     {sl_cc:.4f}")
    print(f"  Ratio (real/CC):     {state_lift / max(sl_cc, 0.001):.3f}")
    if auc_results:
        print(f"  AUC action:          {auc_results['auc_action']:.4f}")
        print(f"  AUC state+action:    {auc_results['auc_state_action']:.4f}")
        print(f"  State-lift (AUC):    {auc_results['state_lift_auc']:.4f}")
    return results


# ===================================================================
# Dataset 1: MathShepherd (PROPER parsing)
# ===================================================================
def load_math_shepherd_proper():
    """
    Proper parsing: compare input vs label fields.
    input has ки markers, label has + or - at same positions.
    """
    print("\n[1] Loading MathShepherd (proper parsing)...")
    ds = load_dataset("peiyi9979/Math-Shepherd", split="train")
    print(f"  Total: {len(ds)} examples")

    states, actions, ratings = [], [], []
    n_parsed = 0
    n_pos, n_neg = 0, 0

    for ex in ds:
        if n_parsed >= 20000:
            break

        inp = ex.get('input', '')
        lbl = ex.get('label', '')

        if not inp or not lbl:
            continue

        # Find ки positions in input
        ki_marker = 'ки'
        ki_positions = [m.start() for m in re.finditer(re.escape(ki_marker), inp)]

        if len(ki_positions) < 2:
            continue

        # Extract problem (before first "Step")
        step_match = re.search(r'Step \d+:', inp)
        if step_match:
            problem = inp[:step_match.start()].strip()
        else:
            problem = inp[:ki_positions[0]].strip()

        if not problem:
            continue

        # Extract step texts and labels
        step_texts = []
        step_labels = []

        for idx, ki_pos in enumerate(ki_positions):
            # Step text: from after previous ки (or start) to this ки
            if idx == 0:
                start = step_match.start() if step_match else 0
            else:
                start = ki_positions[idx-1] + len(ki_marker)

            step_text = inp[start:ki_pos].strip()
            # Remove "Step N:" prefix
            step_text = re.sub(r'^Step \d+:\s*', '', step_text).strip()
            # Remove leading newline
            step_text = step_text.lstrip('\n').strip()

            if not step_text:
                continue

            # Get label from label field at same position
            # In label field, ки is replaced with + or -
            if ki_pos < len(lbl):
                label_char = lbl[ki_pos]
                if label_char == '+':
                    step_labels.append(1)
                    n_pos += 1
                elif label_char == '-':
                    step_labels.append(-1)
                    n_neg += 1
                else:
                    step_labels.append(0)  # unknown
            else:
                step_labels.append(0)

            step_texts.append(step_text)

        # Filter out unknowns and require variation
        valid = [(t, l) for t, l in zip(step_texts, step_labels) if l != 0]
        if len(valid) < 2:
            continue

        for i, (step_text, label) in enumerate(valid):
            if i == 0:
                state = problem
            else:
                prev = [v[0] for v in valid[:i]]
                state = problem + "\n" + "\n".join(prev)

            states.append(state[:1024])
            actions.append(step_text[:512])
            ratings.append(label)

        n_parsed += 1

    print(f"  Parsed {n_parsed} chains, {len(states)} steps")
    print(f"  Positive: {n_pos}, Negative: {n_neg}")
    print(f"  Negative rate: {n_neg / max(n_pos + n_neg, 1):.3f}")

    return states, actions, ratings


# ===================================================================
# Dataset 2: MathShepherd by task type (GSM8K vs MATH)
# ===================================================================
def load_math_shepherd_by_task():
    """
    MathShepherd has a 'task' field (GSM8K or MATH).
    Compute SL separately for each to see if difficulty matters.
    """
    print("\n[2] Loading MathShepherd by task...")
    ds = load_dataset("peiyi9979/Math-Shepherd", split="train")

    results_by_task = {}

    for task_name in ['GSM8K', 'MATH']:
        print(f"\n  === Task: {task_name} ===")
        states, actions, ratings = [], [], []
        n_parsed = 0

        for ex in ds:
            if n_parsed >= 10000:
                break
            if ex.get('task', '') != task_name:
                continue

            inp = ex.get('input', '')
            lbl = ex.get('label', '')
            if not inp or not lbl:
                continue

            ki_marker = 'ки'
            ki_positions = [m.start() for m in re.finditer(re.escape(ki_marker), inp)]
            if len(ki_positions) < 2:
                continue

            step_match = re.search(r'Step \d+:', inp)
            if step_match:
                problem = inp[:step_match.start()].strip()
            else:
                problem = inp[:ki_positions[0]].strip()

            step_texts, step_labels = [], []
            for idx, ki_pos in enumerate(ki_positions):
                if idx == 0:
                    start = step_match.start() if step_match else 0
                else:
                    start = ki_positions[idx-1] + len(ki_marker)

                step_text = inp[start:ki_pos].strip()
                step_text = re.sub(r'^Step \d+:\s*', '', step_text).strip().lstrip('\n').strip()

                if not step_text:
                    continue

                if ki_pos < len(lbl):
                    label_char = lbl[ki_pos]
                    if label_char == '+':
                        step_labels.append(1)
                    elif label_char == '-':
                        step_labels.append(-1)
                    else:
                        step_labels.append(0)
                else:
                    step_labels.append(0)
                step_texts.append(step_text)

            valid = [(t, l) for t, l in zip(step_texts, step_labels) if l != 0]
            if len(valid) < 2:
                continue

            for i, (step_text, label) in enumerate(valid):
                if i == 0:
                    state = problem
                else:
                    prev = [v[0] for v in valid[:i]]
                    state = problem + "\n" + "\n".join(prev)
                states.append(state[:1024])
                actions.append(step_text[:512])
                ratings.append(label)
            n_parsed += 1

        print(f"  {task_name}: {n_parsed} chains, {len(states)} steps")
        neg_rate = sum(1 for r in ratings if r < 0) / max(len(ratings), 1)
        print(f"  Negative rate: {neg_rate:.3f}")

        if len(states) >= 200:
            result = compute_state_lift(states, actions, ratings,
                                        f"MathShepherd-{task_name}")
            results_by_task[task_name] = result

    return results_by_task


# ===================================================================
# Dataset 3: PRM800K ALL completions (proper HF loading)
# ===================================================================
def load_prm800k_all():
    """
    PRM800K with all completions. Try loading from HF properly.
    The key difference: at each step, multiple completions exist with
    different ratings. This gives within-state variation.
    """
    print("\n[3] Loading PRM800K (all completions)...")

    # Check if local data exists
    local_path = 'data'
    prm_local = os.path.join(local_path, 'prm800k')

    # Try downloading from HF
    for name in ["openai/prm800k"]:
        try:
            ds = load_dataset(name, "phase2_train", split="train", streaming=True)
            print(f"  Loaded from {name}")
            break
        except Exception as e:
            print(f"  {name}: {e}")
            try:
                ds = load_dataset(name, split="train", streaming=True)
                print(f"  Loaded from {name} (no config)")
                break
            except Exception as e2:
                print(f"  {name} no config: {e2}")
    else:
        # Try the DigitalLearning mirror
        try:
            ds = load_dataset("DigitalLearningGmbH/PRM800K", split="train",
                              streaming=True)
            print("  Loaded from DigitalLearningGmbH/PRM800K")
        except Exception as e:
            print(f"  Mirror: {e}")
            print("  Trying Hugging Face search...")
            # Try another known mirror
            for mirror in ["tasksource/PRM800K", "llemma/PRM800K"]:
                try:
                    ds = load_dataset(mirror, split="train", streaming=True)
                    print(f"  Loaded from {mirror}")
                    break
                except:
                    continue
            else:
                print("  Cannot load PRM800K from any source")
                return None, None, None

    # Inspect
    first = next(iter(ds))
    print(f"  Keys: {list(first.keys())}")
    for k, v in first.items():
        val = str(v)[:300]
        print(f"    {k}: {val}")

    states, actions, ratings = [], [], []
    n_chains = 0
    n_total_completions = 0

    for ex in ds:
        if n_chains >= 10000:
            break

        # Extract based on format found
        question = ex.get('question', {})
        if isinstance(question, dict):
            problem = question.get('problem', '')
        elif isinstance(question, str):
            problem = question
        else:
            continue

        label = ex.get('label', {})
        if isinstance(label, dict):
            steps = label.get('steps', [])
        elif isinstance(label, list):
            steps = label
        else:
            continue

        if not problem or not steps or len(steps) < 2:
            continue

        prev_texts = []

        for step_i, step in enumerate(steps):
            if isinstance(step, dict):
                completions = step.get('completions', [])
                chosen_idx = step.get('chosen_completion', 0)
            elif isinstance(step, list):
                completions = step
                chosen_idx = 0
            else:
                continue

            if not completions:
                continue

            # State = problem + previous chosen steps
            if step_i == 0:
                state = problem
            else:
                state = problem + "\n" + "\n".join(prev_texts)

            # Record ALL completions at this step
            for comp in completions:
                if isinstance(comp, dict):
                    text = comp.get('text', '')
                    rating = comp.get('rating', None)
                elif isinstance(comp, str):
                    text = comp
                    rating = None
                else:
                    continue

                if text and rating is not None:
                    states.append(state[:1024])
                    actions.append(text[:512])
                    ratings.append(int(rating))
                    n_total_completions += 1

            # Update prev_texts with chosen completion
            if isinstance(chosen_idx, int) and chosen_idx < len(completions):
                chosen = completions[chosen_idx]
                chosen_text = chosen.get('text', '') if isinstance(chosen, dict) else str(chosen)
            else:
                chosen_text = completions[0].get('text', '') if isinstance(completions[0], dict) else str(completions[0])
            prev_texts.append(chosen_text)

        n_chains += 1

    print(f"  Parsed {n_chains} chains, {len(states)} step-completion pairs")
    print(f"  Average completions per position: {n_total_completions / max(n_chains, 1):.1f}")

    if len(states) < 100:
        return None, None, None
    return states, actions, ratings


# ===================================================================
# MAIN
# ===================================================================
def main():
    all_results = {}

    # 1. MathShepherd proper
    print("#" * 70)
    print("# MathShepherd (proper parsing)")
    print("#" * 70)
    try:
        s, a, r = load_math_shepherd_proper()
        if s and len(s) >= 100:
            result = compute_state_lift(s, a, r, "MathShepherd-proper")
            all_results['MathShepherd-proper'] = result
            with open(os.path.join(OUT_DIR, 'r12c_mathshepherd_proper.json'), 'w') as f:
                json.dump(result, f, indent=2)
        else:
            all_results['MathShepherd-proper'] = {'status': 'skipped'}
    except Exception as e:
        import traceback
        traceback.print_exc()
        all_results['MathShepherd-proper'] = {'status': 'error', 'error': str(e)}

    # 2. MathShepherd by task
    print("\n" + "#" * 70)
    print("# MathShepherd by task (GSM8K vs MATH)")
    print("#" * 70)
    try:
        task_results = load_math_shepherd_by_task()
        for task, res in task_results.items():
            all_results[f'MathShepherd-{task}'] = res
            with open(os.path.join(OUT_DIR, f'r12c_mathshepherd_{task.lower()}.json'), 'w') as f:
                json.dump(res, f, indent=2)
    except Exception as e:
        import traceback
        traceback.print_exc()

    # 3. PRM800K all completions
    print("\n" + "#" * 70)
    print("# PRM800K (all completions)")
    print("#" * 70)
    try:
        s, a, r = load_prm800k_all()
        if s and len(s) >= 100:
            result = compute_state_lift(s, a, r, "PRM800K-AllComp")
            all_results['PRM800K-AllComp'] = result
            with open(os.path.join(OUT_DIR, 'r12c_prm800k_allcomp.json'), 'w') as f:
                json.dump(result, f, indent=2)
        else:
            all_results['PRM800K-AllComp'] = {'status': 'skipped'}
    except Exception as e:
        import traceback
        traceback.print_exc()
        all_results['PRM800K-AllComp'] = {'status': 'error', 'error': str(e)}

    # Summary
    print("\n\n" + "=" * 80)
    print("R12c FINAL SUMMARY: STATE-LIFT ON REAL STEP LABELS")
    print("=" * 80)

    baselines = {
        'PRM800K (r2, chosen)': {'sl': 0.015, 'ci': [-0.010, 0.017], 'source': 'Original r2'},
        'PRM800K (r9, nonlinear)': {'sl': 0.018, 'ci': None, 'source': 'MLP extension'},
    }

    print(f"\n{'Dataset':<30} {'SL(real)':<10} {'SL(CC)':<10} {'CI':<25} {'Notes'}")
    print("-" * 95)

    for name, info in baselines.items():
        ci = info['ci']
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A"
        print(f"{name:<30} {info['sl']:<10.4f} {'--':<10} {ci_str:<25} {info['source']}")

    for name, res in all_results.items():
        if isinstance(res, dict) and 'state_lift' in res:
            ci = res.get('bootstrap_ci', [None, None])
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else "N/A"
            cc = res.get('context_correctness_sl', 0)
            notes = f"n={res['n_samples']}"
            if 'binary_analysis' in res:
                notes += f", AUC_SL={res['binary_analysis']['state_lift_auc']:.4f}"
            print(f"{name:<30} {res['state_lift']:<10.4f} {cc:<10.4f} {ci_str:<25} {notes}")

    print("\n\nKEY FINDING:")
    math_sls = []
    for name, res in all_results.items():
        if isinstance(res, dict) and 'state_lift' in res:
            math_sls.append(res['state_lift'])

    if math_sls:
        mean_sl = np.mean(math_sls)
        print(f"  Mean SL across new datasets: {mean_sl:.4f}")
        if mean_sl < 0.05:
            print("  CONSISTENT WITH PRM800K: Step correctness is state-independent")
            print("  in embedding space across math datasets.")
            print("  The PRM advantage comes from cross-attention (interaction-level),")
            print("  not from state representations (R11 finding confirmed).")
        else:
            print("  DIVERGENT FROM PRM800K: Some math datasets show measurable SL.")
            print("  The difference may come from annotation methodology or difficulty.")

    # Save
    out_path = os.path.join(OUT_DIR, 'r12c_combined.json')
    with open(out_path, 'w') as f:
        json.dump({
            'baselines': {k: v for k, v in baselines.items()},
            'datasets': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
