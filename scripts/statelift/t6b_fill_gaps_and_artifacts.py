"""
T6b: Fill missing domains + artifact checks on new high-SDI results.

Missing from T6: negotiation, persuasion, task-oriented dialogue.
Try alternative datasets. Also artifact-check the new domains
(MathDial, Mind2Web, CodeContests) to confirm signal is real.
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


def compute_sdi(z_raw, a_raw, zn_raw, name="", verbose=True):
    """Compute Sequential Dependence Index."""
    if len(z_raw) < 50:
        if verbose:
            print(f"  Too few transitions ({len(z_raw)})")
        return None

    pca = PCA(n_components=min(D_PCA, len(z_raw)//3))
    pca.fit(np.vstack([z_raw, a_raw, zn_raw]))
    z = pca.transform(z_raw)
    a = pca.transform(a_raw)
    zn = pca.transform(zn_raw)
    d = z.shape[1]

    z_mean = z.mean(0)
    improvement = np.linalg.norm(z - z_mean, axis=1) - np.linalg.norm(zn - z_mean, axis=1)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    a_centroid = a.mean(0)
    X_outcome = np.hstack([a, np.linalg.norm(a, axis=1, keepdims=True),
                            np.linalg.norm(a - a_centroid, axis=1, keepdims=True)])
    r2_outcome = max(cross_val_score(Ridge(alpha=1.0), X_outcome, improvement,
                                      cv=cv, scoring='r2').mean(), 0.001)

    X_process = np.hstack([z, a, z * a])
    r2_process = cross_val_score(Ridge(alpha=1.0), X_process, improvement,
                                  cv=cv, scoring='r2').mean()

    sdi = r2_process / max(r2_outcome, 0.001)

    # State lift
    r2_action = cross_val_score(Ridge(alpha=1.0), a, improvement, cv=cv, scoring='r2').mean()
    X_sa = np.hstack([a, z, a * z])
    r2_sa = cross_val_score(Ridge(alpha=1.0), X_sa, improvement, cv=cv, scoring='r2').mean()
    state_lift = r2_sa - max(r2_action, 0)

    # World model
    tr = np.random.rand(len(z)) < 0.7
    if tr.sum() > 10 and (~tr).sum() > 10:
        wm = Ridge(alpha=1.0)
        wm.fit(np.hstack([z[tr], a[tr]]), zn[tr])
        pred = wm.predict(np.hstack([z[~tr], a[~tr]]))
        r2_wm = 1 - np.sum((zn[~tr]-pred)**2) / np.sum((zn[~tr]-zn[~tr].mean(0))**2)
    else:
        r2_wm = float('nan')

    ac = np.mean([np.corrcoef(z[:-1,dd], z[1:,dd])[0,1] for dd in range(min(5, d))])

    if verbose:
        print(f"  {name}: {len(z)} trans, SDI={sdi:.2f}, lift={state_lift:.4f}, AC={ac:.3f}")

    return {
        'name': name, 'n_transitions': len(z),
        'r2_outcome_reward': float(r2_outcome), 'r2_process_reward': float(r2_process),
        'SDI': float(sdi), 'state_conditioning_lift': float(state_lift),
        'r2_dynamics': float(r2_wm), 'state_persistence': float(ac),
    }


def build_dialogue_transitions(conversations, min_turns=4):
    all_texts = []
    conv_bounds = []
    for conv in conversations:
        if len(conv) < min_turns:
            continue
        s = len(all_texts)
        for spk, txt in conv:
            all_texts.append(str(txt)[:512])
        conv_bounds.append((s, len(all_texts)))
    if not all_texts:
        return None, None, None
    print(f"  Embedding {len(all_texts)} turns from {len(conv_bounds)} conversations...")
    embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)
    tz, ta, tzn = [], [], []
    for s, e in conv_bounds:
        n = e - s
        if n < 3:
            continue
        for i in range(1, n - 1):
            tz.append(embs[s + i - 1])
            ta.append(embs[s + i])
            tzn.append(embs[s + i + 1])
    if len(tz) < 50:
        return None, None, None
    return np.array(tz), np.array(ta), np.array(tzn)


def build_chain_transitions(chains):
    all_texts = []
    chain_bounds = []
    for chain in chains:
        steps = chain['steps']
        if len(steps) < 3:
            continue
        s = len(all_texts)
        if 'context' in chain:
            all_texts.append(chain['context'][:512])
        for step in steps:
            all_texts.append(str(step)[:512])
        chain_bounds.append((s, len(all_texts)))
    if not all_texts:
        return None, None, None
    print(f"  Embedding {len(all_texts)} texts from {len(chain_bounds)} chains...")
    embs = encoder.encode(all_texts, batch_size=256, show_progress_bar=False)
    tz, ta, tzn = [], [], []
    for s, e in chain_bounds:
        n = e - s
        if n < 3:
            continue
        for i in range(1, n - 1):
            tz.append(embs[s + i - 1])
            ta.append(embs[s + i])
            tzn.append(embs[s + i + 1])
    if len(tz) < 50:
        return None, None, None
    return np.array(tz), np.array(ta), np.array(tzn)


print("=" * 70)
print("T6b: FILL GAPS + ARTIFACT CHECKS")
print("=" * 70)

results = {}

# ------------------------------------------------------------------
# PART A: FILL MISSING DOMAINS
# ------------------------------------------------------------------

# A1: Negotiation -- try casino_negotiation or another dataset
print("\n[A1] Negotiation -- trying alternatives...")
tried = []
for ds_name, loader_fn in [
    ('casino_negotiation', lambda: load_dataset('casino', split='train')),
    ('multi_turn_negotiation', lambda: load_dataset('Andyrasika/multi_turn_chat_negotiation', split='train')),
]:
    try:
        print(f"  Trying {ds_name}...")
        ds = loader_fn()
        print(f"  {ds_name} loaded: {len(ds)} examples")

        # Inspect structure
        ex = ds[0]
        print(f"  Keys: {list(ex.keys())[:10]}")

        convs = []
        for ex in list(ds)[:2000]:
            # Try common conversation fields
            for key in ['chat_logs', 'dialogue', 'conversation', 'dialog', 'turns', 'messages']:
                data = ex.get(key, None)
                if data is None:
                    continue
                if isinstance(data, str):
                    turns = [t.strip() for t in data.split('\n') if t.strip() and len(t.strip()) > 5]
                    if len(turns) >= 4:
                        conv = [(f'spk{i%2}', t) for i, t in enumerate(turns)]
                        convs.append(conv)
                        break
                elif isinstance(data, list):
                    conv = []
                    for i, t in enumerate(data):
                        if isinstance(t, dict):
                            text = t.get('text', t.get('content', t.get('message', str(t))))
                            role = t.get('role', t.get('speaker', f'spk{i%2}'))
                            conv.append((str(role), str(text)[:512]))
                        else:
                            conv.append((f'spk{i%2}', str(t)[:512]))
                    if len(conv) >= 4:
                        convs.append(conv)
                        break

        print(f"  {len(convs)} conversations extracted")
        if len(convs) >= 30:
            z, a, zn = build_dialogue_transitions(convs, min_turns=4)
            if z is not None:
                r = compute_sdi(z, a, zn, f"Negotiation-{ds_name}")
                if r:
                    results[f'Negotiation ({ds_name})'] = r
                    results[f'Negotiation ({ds_name})']['domain_type'] = 'negotiation'
                    results[f'Negotiation ({ds_name})']['genuinely_sequential'] = True
                    break
    except Exception as e:
        tried.append(f"{ds_name}: {e}")
        continue

if not any('Negotiation' in k for k in results):
    print(f"  All negotiation datasets failed: {tried}")

# A2: Task-oriented -- try DSTC or KETOD
print("\n[A2] Task-Oriented Dialogue...")
for ds_name, loader_fn in [
    ('KETOD', lambda: load_dataset('nicholasKluge/KETOD', split='train')),
    ('DSTC2', lambda: load_dataset('dstc2', split='train')),
    ('SGD', lambda: load_dataset('schema_guided_dstc8', split='train')),
]:
    try:
        print(f"  Trying {ds_name}...")
        ds = loader_fn()
        print(f"  {ds_name} loaded: {len(ds)} examples")
        ex = ds[0]
        print(f"  Keys: {list(ex.keys())[:10]}")

        convs = []
        for ex in list(ds)[:2000]:
            for key in ['dialogue', 'conversation', 'dialog', 'turns', 'messages', 'utterances']:
                data = ex.get(key, None)
                if data is None:
                    continue
                if isinstance(data, str):
                    turns = [t.strip() for t in data.split('\n') if t.strip() and len(t.strip()) > 5]
                    if len(turns) >= 6:
                        conv = [(f'spk{i%2}', t) for i, t in enumerate(turns)]
                        convs.append(conv)
                        break
                elif isinstance(data, list):
                    conv = []
                    for i, t in enumerate(data):
                        if isinstance(t, dict):
                            text = t.get('text', t.get('content', t.get('utterance', str(t))))
                            role = t.get('role', t.get('speaker', f'spk{i%2}'))
                            conv.append((str(role), str(text)[:512]))
                        else:
                            conv.append((f'spk{i%2}', str(t)[:512]))
                    if len(conv) >= 6:
                        convs.append(conv)
                        break

        print(f"  {len(convs)} conversations extracted")
        if len(convs) >= 30:
            z, a, zn = build_dialogue_transitions(convs, min_turns=6)
            if z is not None:
                r = compute_sdi(z, a, zn, f"TaskOriented-{ds_name}")
                if r:
                    results[f'Task-Oriented ({ds_name})'] = r
                    results[f'Task-Oriented ({ds_name})']['domain_type'] = 'task-oriented'
                    results[f'Task-Oriented ({ds_name})']['genuinely_sequential'] = True
                    break
    except Exception as e:
        print(f"  {ds_name} failed: {e}")

# A3: Persuasion -- try alternative
print("\n[A3] Persuasion...")
for ds_name, loader_fn in [
    ('ElecDeb60to20', lambda: load_dataset('theblackcat102/elec-deb60to20', split='train')),
    ('CMV_pairs', lambda: load_dataset('webis/change-my-view', split='train')),
]:
    try:
        print(f"  Trying {ds_name}...")
        ds = loader_fn()
        print(f"  {ds_name} loaded: {len(ds)} examples")
        ex = ds[0]
        print(f"  Keys: {list(ex.keys())[:10]}")

        convs = []
        for ex in list(ds)[:2000]:
            for key in ['dialogue', 'conversation', 'dialog', 'text', 'post']:
                data = ex.get(key, None)
                if data is None:
                    continue
                if isinstance(data, str) and len(data) > 100:
                    # Split into paragraphs as turns
                    paras = [p.strip() for p in data.split('\n\n') if p.strip() and len(p.strip()) > 20]
                    if len(paras) < 4:
                        paras = [p.strip() for p in data.split('\n') if p.strip() and len(p.strip()) > 20]
                    if len(paras) >= 4:
                        conv = [(f'spk{i%2}', p[:512]) for i, p in enumerate(paras)]
                        convs.append(conv)
                        break

        print(f"  {len(convs)} conversations extracted")
        if len(convs) >= 30:
            z, a, zn = build_dialogue_transitions(convs, min_turns=4)
            if z is not None:
                r = compute_sdi(z, a, zn, f"Persuasion-{ds_name}")
                if r:
                    results[f'Persuasion ({ds_name})'] = r
                    results[f'Persuasion ({ds_name})']['domain_type'] = 'persuasion'
                    results[f'Persuasion ({ds_name})']['genuinely_sequential'] = True
                    break
    except Exception as e:
        print(f"  {ds_name} failed: {e}")

# A4: Multi-turn tool use -- try glaive or gorilla
print("\n[A4] Multi-turn Tool Use...")
for ds_name, loader_fn in [
    ('glaive_v2', lambda: load_dataset('glaiveai/glaive-function-calling-v2', split='train')),
    ('gorilla', lambda: load_dataset('gorilla-llm/APIBench', split='train')),
]:
    try:
        print(f"  Trying {ds_name}...")
        ds = loader_fn()
        print(f"  {ds_name} loaded: {len(ds)} examples")
        ex = ds[0]
        print(f"  Keys: {list(ex.keys())[:10]}")

        convs = []
        for ex in list(ds)[:3000]:
            for key in ['chat', 'conversations', 'messages', 'system_prompt', 'text']:
                data = ex.get(key, None)
                if data is None:
                    continue
                if isinstance(data, str) and len(data) > 100:
                    # Parse USER: / ASSISTANT: / FUNCTION pattern
                    lines = data.split('\n')
                    conv = []
                    for line in lines:
                        line = line.strip()
                        if not line or len(line) < 10:
                            continue
                        for prefix in ['USER:', 'ASSISTANT:', 'FUNCTION RESPONSE:', 'SYSTEM:']:
                            if line.upper().startswith(prefix):
                                role = prefix.replace(':', '').lower()
                                text = line[len(prefix):].strip()
                                if text:
                                    conv.append((role, text[:512]))
                                break
                        else:
                            if len(line) > 20:
                                conv.append(('unknown', line[:512]))
                    if len(conv) >= 4:
                        convs.append(conv)
                        break
                elif isinstance(data, list) and len(data) >= 4:
                    conv = []
                    for i, t in enumerate(data):
                        if isinstance(t, dict):
                            text = t.get('value', t.get('content', t.get('text', str(t))))
                            role = t.get('from', t.get('role', f'spk{i%3}'))
                            conv.append((str(role), str(text)[:512]))
                        else:
                            conv.append((f'spk{i%3}', str(t)[:512]))
                    convs.append(conv)
                    break

        print(f"  {len(convs)} multi-turn tool-use conversations")
        if len(convs) >= 50:
            z, a, zn = build_dialogue_transitions(convs, min_turns=4)
            if z is not None:
                r = compute_sdi(z, a, zn, f"ToolUse-{ds_name}")
                if r:
                    results[f'Tool-Use ({ds_name})'] = r
                    results[f'Tool-Use ({ds_name})']['domain_type'] = 'tool-use'
                    results[f'Tool-Use ({ds_name})']['genuinely_sequential'] = True
                    break
    except Exception as e:
        print(f"  {ds_name} failed: {e}")

# ------------------------------------------------------------------
# PART B: ARTIFACT CHECKS ON NEW HIGH-SDI DOMAINS
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART B: ARTIFACT CHECKS ON NEW DOMAINS")
print("=" * 70)

# Reload the chain/dialogue data for artifact checks
# We'll re-load the key datasets and run shuffled-state controls

# B1: MathDial artifact check
print("\n[B1] MathDial -- shuffled state control...")
try:
    ds = load_dataset('eth-nlped/mathdial', split='train')
    convs = []
    for ex in list(ds)[:2000]:
        conv_text = ex.get('conversation', '')
        if isinstance(conv_text, str):
            turns = conv_text.split('\n')
            conv = []
            for t in turns:
                t = t.strip()
                if not t:
                    continue
                if ':' in t:
                    role, text = t.split(':', 1)
                    conv.append((role.strip(), text.strip()))
                else:
                    conv.append(('unknown', t))
            if len(conv) >= 4:
                convs.append(conv)

    z, a, zn = build_dialogue_transitions(convs, min_turns=4)
    if z is not None:
        # Real SDI
        real = compute_sdi(z, a, zn, "MathDial-REAL")

        # Shuffled state: randomize z while keeping a, zn paired
        np.random.seed(SEED)
        z_shuf = z[np.random.permutation(len(z))]
        shuf = compute_sdi(z_shuf, a, zn, "MathDial-SHUFFLED-STATE")

        # Cross-conversation: pair z from different conversations with a, zn
        np.random.seed(SEED+1)
        z_cross = z[np.random.permutation(len(z))]
        zn_cross = zn[np.random.permutation(len(zn))]
        cross = compute_sdi(z_cross, a, zn_cross, "MathDial-CROSS-CONV")

        if real and shuf:
            print(f"\n  MathDial artifact check:")
            print(f"    Real SDI:          {real['SDI']:.2f}")
            print(f"    Shuffled state:    {shuf['SDI']:.2f}")
            if cross:
                print(f"    Cross-conv:        {cross['SDI']:.2f}")
            drop_pct = (1 - shuf['SDI'] / real['SDI']) * 100 if real['SDI'] > 0 else 0
            print(f"    Drop: {drop_pct:.0f}% --> {'PASS' if drop_pct > 50 else 'CONCERN'}")
            results['MathDial_artifact'] = {
                'real_SDI': real['SDI'], 'shuffled_SDI': shuf['SDI'],
                'cross_SDI': cross['SDI'] if cross else None,
                'drop_pct': drop_pct, 'verdict': 'PASS' if drop_pct > 50 else 'CONCERN'
            }
except Exception as e:
    print(f"  MathDial artifact check failed: {e}")

# B2: Mind2Web artifact check
print("\n[B2] Mind2Web -- shuffled state control...")
try:
    ds = load_dataset('osunlp/Mind2Web', split='train')
    chains = []
    for ex in list(ds)[:2000]:
        actions = ex.get('action_reprs', [])
        if isinstance(actions, list) and len(actions) >= 3:
            steps = [str(a)[:512] for a in actions]
            chains.append({'context': str(ex.get('confirmed_task', ''))[:512], 'steps': steps})

    z, a, zn = build_chain_transitions(chains)
    if z is not None:
        real = compute_sdi(z, a, zn, "Mind2Web-REAL")

        np.random.seed(SEED)
        z_shuf = z[np.random.permutation(len(z))]
        shuf = compute_sdi(z_shuf, a, zn, "Mind2Web-SHUFFLED-STATE")

        np.random.seed(SEED+1)
        z_cross = z[np.random.permutation(len(z))]
        zn_cross = zn[np.random.permutation(len(zn))]
        cross = compute_sdi(z_cross, a, zn_cross, "Mind2Web-CROSS-CHAIN")

        if real and shuf:
            print(f"\n  Mind2Web artifact check:")
            print(f"    Real SDI:          {real['SDI']:.2f}")
            print(f"    Shuffled state:    {shuf['SDI']:.2f}")
            if cross:
                print(f"    Cross-chain:       {cross['SDI']:.2f}")
            drop_pct = (1 - shuf['SDI'] / real['SDI']) * 100 if real['SDI'] > 0 else 0
            print(f"    Drop: {drop_pct:.0f}% --> {'PASS' if drop_pct > 50 else 'CONCERN'}")
            results['Mind2Web_artifact'] = {
                'real_SDI': real['SDI'], 'shuffled_SDI': shuf['SDI'],
                'cross_SDI': cross['SDI'] if cross else None,
                'drop_pct': drop_pct, 'verdict': 'PASS' if drop_pct > 50 else 'CONCERN'
            }
except Exception as e:
    print(f"  Mind2Web artifact check failed: {e}")

# B3: CodeContests artifact check
print("\n[B3] CodeContests -- shuffled state control...")
try:
    ds = load_dataset('deepmind/code_contests', split='train')
    chains = []
    for ex in list(ds)[:1000]:
        solutions = ex.get('solutions', {})
        if isinstance(solutions, dict):
            sol_list = solutions.get('solution', [])
        elif isinstance(solutions, list):
            sol_list = solutions
        else:
            continue
        for sol in sol_list[:2]:
            if isinstance(sol, str) and len(sol) > 50:
                lines = sol.split('\n')
                blocks = []
                current = []
                for line in lines:
                    current.append(line)
                    if len('\n'.join(current)) > 100 and (
                        line.strip() == '' or line.strip().startswith('def ') or
                        line.strip().startswith('class ')):
                        blocks.append('\n'.join(current))
                        current = []
                if current:
                    blocks.append('\n'.join(current))
                if len(blocks) >= 3:
                    chains.append({
                        'context': str(ex.get('description', ''))[:512],
                        'steps': [b[:512] for b in blocks]
                    })

    z, a, zn = build_chain_transitions(chains)
    if z is not None:
        real = compute_sdi(z, a, zn, "CodeContests-REAL")

        np.random.seed(SEED)
        z_shuf = z[np.random.permutation(len(z))]
        shuf = compute_sdi(z_shuf, a, zn, "CodeContests-SHUFFLED-STATE")

        if real and shuf:
            print(f"\n  CodeContests artifact check:")
            print(f"    Real SDI:          {real['SDI']:.2f}")
            print(f"    Shuffled state:    {shuf['SDI']:.2f}")
            drop_pct = (1 - shuf['SDI'] / real['SDI']) * 100 if real['SDI'] > 0 else 0
            print(f"    Drop: {drop_pct:.0f}% --> {'PASS' if drop_pct > 50 else 'CONCERN'}")
            results['CodeContests_artifact'] = {
                'real_SDI': real['SDI'], 'shuffled_SDI': shuf['SDI'],
                'drop_pct': drop_pct, 'verdict': 'PASS' if drop_pct > 50 else 'CONCERN'
            }
except Exception as e:
    print(f"  CodeContests artifact check failed: {e}")

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("T6b SUMMARY")
print("=" * 70)

print("\nNew domains added:")
for name, r in results.items():
    if '_artifact' not in name:
        print(f"  {name}: SDI={r.get('SDI', '?'):.2f}")

print("\nArtifact checks:")
for name, r in results.items():
    if '_artifact' in name:
        print(f"  {name}: real={r['real_SDI']:.2f}, shuffled={r['shuffled_SDI']:.2f}, "
              f"drop={r['drop_pct']:.0f}%, verdict={r['verdict']}")

# Save
out_path = os.path.join(OUT_DIR, 't6b_fill_gaps_artifacts.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
