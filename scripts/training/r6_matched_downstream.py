"""
R6: Matched-Distribution Downstream Evaluation

Fix for R5: train reward models on Llama-8B's OWN generations (not PRM800K).
Eliminates distribution shift between training and scoring.

Pipeline (all on SSCC, single script):
1. Generate 16 solutions per GSM8K train problem (500 problems)
2. Auto-label: check final answer, steps from correct = positive, incorrect = negative
3. Train blind (ORM) and conditioned (PRM) step-level reward models
4. Step-level beam search on GSM8K test (100 problems)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import json
import os
import re
import time

SEED = 42
MODEL_PATH = os.environ.get("LLAMA_PATH", "meta-llama/Llama-3.1-8B-Instruct")
OUT_DIR = "results"
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-5
LORA_R = 8
LORA_ALPHA = 16
MAX_LEN = 512

# Generation params
N_TRAIN_PROBLEMS = 400
N_SOLUTIONS_PER = 16
N_TEST_PROBLEMS = 100
N_STEP_CANDIDATES = 4
MAX_STEPS = 8

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

os.makedirs(OUT_DIR, exist_ok=True)


def extract_answer(text):
    match = re.search(r'####\s*(.+)', text)
    if match:
        ans = match.group(1).strip().replace(',', '').replace('$', '')
        num = re.search(r'-?\d+\.?\d*', ans)
        return num.group(0) if num else ans
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None


# ===================================================================
# PHASE 1: Load GSM8K and generate solutions with Llama-8B
# ===================================================================
print("=" * 60)
print("PHASE 1: GENERATE TRAINING DATA WITH LLAMA-8B")
print("=" * 60)

# Load GSM8K from pre-saved .pt (SSCC has no internet)
GSM8K_PATH = "data/r6_gsm8k_problems.pt"
gsm8k_data = torch.load(GSM8K_PATH, weights_only=False)
train_problems = gsm8k_data['train'][:N_TRAIN_PROBLEMS]
test_problems = gsm8k_data['test'][:N_TEST_PROBLEMS]

print(f"  Train problems: {len(train_problems)}")
print(f"  Test problems: {len(test_problems)}")

# Generate solutions
print(f"\n  Generating {N_SOLUTIONS_PER} solutions per train problem...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto')

generated = []  # (question, solution_text, steps, correct)
for pi, prob in enumerate(train_problems):
    if (pi + 1) % 50 == 0:
        print(f"    Problem {pi+1}/{len(train_problems)}...")

    prompt = (f"Solve this math problem step by step. Show each step on a new line. "
              f"End with #### followed by the final numeric answer.\n\n"
              f"Problem: {prob['question']}\n\nSolution:")
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256).to(device)

    for si in range(N_SOLUTIONS_PER):
        with torch.no_grad():
            output = gen_model.generate(
                **inputs, max_new_tokens=300, temperature=0.8, top_p=0.95,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1)
        response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        pred = extract_answer(response)
        correct = pred == prob['final_answer'] if pred else False

        steps = [s.strip() for s in response.split('\n') if s.strip() and len(s.strip()) > 5]
        if len(steps) >= 2:
            generated.append({
                'question': prob['question'],
                'solution': response,
                'steps': steps,
                'correct': correct,
            })

del gen_model
torch.cuda.empty_cache()

n_correct = sum(1 for g in generated if g['correct'])
n_incorrect = sum(1 for g in generated if not g['correct'])
print(f"\n  Generated {len(generated)} solutions ({n_correct} correct, {n_incorrect} incorrect)")
print(f"  Solve rate: {n_correct/len(generated):.1%}")

# ===================================================================
# PHASE 2: Build step-level training data (matched distribution)
# ===================================================================
print("\n" + "=" * 60)
print("PHASE 2: BUILD STEP-LEVEL TRAINING DATA")
print("=" * 60)

# Steps from correct solutions = positive, from incorrect = negative
step_data = []
for g in generated:
    question = g['question']
    steps = g['steps']
    label = 1 if g['correct'] else 0

    for i, step in enumerate(steps):
        # State = question + previous steps
        if i == 0:
            state = question
        else:
            state = question + "\n" + "\n".join(steps[:i])

        step_data.append({
            'state': state[:768],
            'step': step[:256],
            'label': label,
        })

n_pos = sum(1 for s in step_data if s['label'] == 1)
n_neg = sum(1 for s in step_data if s['label'] == 0)
print(f"  Total steps: {len(step_data)} ({n_pos} positive, {n_neg} negative)")

# Balance
min_class = min(n_pos, n_neg)
pos_steps = [s for s in step_data if s['label'] == 1]
neg_steps = [s for s in step_data if s['label'] == 0]
np.random.shuffle(pos_steps)
np.random.shuffle(neg_steps)

MAX_PER_CLASS = 5000
n_per = min(min_class, MAX_PER_CLASS)
balanced = pos_steps[:n_per] + neg_steps[:n_per]
np.random.shuffle(balanced)

n_train = int(0.85 * len(balanced))
train_data = balanced[:n_train]
test_data = balanced[n_train:]

print(f"  Balanced: {len(balanced)} ({n_per} per class)")
print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

# Build prompts (SAME format for training and scoring)
train_cond = [f"Problem: {s['state']}\n\nNext step: {s['step']}\n\nIs this step correct?" for s in train_data]
train_blind = [f"Reasoning step: {s['step']}\n\nIs this step correct?" for s in train_data]
train_labels = [s['label'] for s in train_data]

test_cond = [f"Problem: {s['state']}\n\nNext step: {s['step']}\n\nIs this step correct?" for s in test_data]
test_blind = [f"Reasoning step: {s['step']}\n\nIs this step correct?" for s in test_data]
test_labels = [s['label'] for s in test_data]

# ===================================================================
# PHASE 3: Train reward models
# ===================================================================
print("\n" + "=" * 60)
print("PHASE 3: TRAIN REWARD MODELS (MATCHED DISTRIBUTION)")
print("=" * 60)

class TextDS(Dataset):
    def __init__(self, texts, labels, tok, ml):
        self.texts, self.labels, self.tok, self.ml = texts, labels, tok, ml
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        e = self.tok(self.texts[i], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {'input_ids': e['input_ids'].squeeze(), 'attention_mask': e['attention_mask'].squeeze(),
                'labels': torch.tensor(self.labels[i], dtype=torch.float32)}

def train_rm(name, texts, labels, t_texts, t_labels):
    print(f"\n  TRAINING: {name} ({len(texts)} train)")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
                                              lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))
    model.to(device)
    loader = DataLoader(TextDS(texts, labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    t_loader = DataLoader(TextDS(t_texts, t_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_st = 0, None
    for ep in range(EPOCHS):
        model.train(); tl, nc, nt = 0, 0, 0; opt.zero_grad(); t0 = time.time()
        for step, b in enumerate(loader):
            lo = model(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device)).logits.squeeze(-1)
            l = loss_fn(lo, b['labels'].to(device)) / GRAD_ACCUM; l.backward()
            if (step+1) % GRAD_ACCUM == 0: opt.step(); opt.zero_grad()
            tl += l.item()*GRAD_ACCUM; nc += ((lo>0).float().cpu()==b['labels']).sum().item(); nt += len(b['labels'])
        opt.step(); opt.zero_grad()
        model.eval(); al, alb = [], []
        with torch.no_grad():
            for b in t_loader:
                lo = model(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device)).logits.squeeze(-1)
                al.append(lo.cpu()); alb.append(b['labels'])
        al = torch.cat(al).float().numpy(); alb = torch.cat(alb).float().numpy()
        try: auc = roc_auc_score(alb, 1/(1+np.exp(-al)))
        except: auc = 0.5
        acc = accuracy_score(alb, (al>0).astype(float))
        print(f"    Ep {ep+1}: loss={tl/len(loader):.4f} train_acc={nc/nt:.3f} eval_acc={acc:.3f} auc={auc:.3f} t={time.time()-t0:.0f}s")
        if auc > best_auc: best_auc = auc; best_st = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_st); model.to(device)
    print(f"    Best AUC: {best_auc:.4f}")
    return model, best_auc

blind_m, blind_auc = train_rm("blind (ORM)", train_blind, train_labels, test_blind, test_labels)
prm_m, prm_auc = train_rm("conditioned (PRM)", train_cond, train_labels, test_cond, test_labels)

print(f"\n  Reward model AUC: blind={blind_auc:.4f}, PRM={prm_auc:.4f}, lift={prm_auc-blind_auc:+.4f}")

# ===================================================================
# PHASE 4: Step-level beam search on GSM8K test
# ===================================================================
print("\n" + "=" * 60)
print("PHASE 4: STEP-LEVEL BEAM SEARCH ON GSM8K TEST")
print("=" * 60)

blind_m.cpu(); prm_m.cpu(); torch.cuda.empty_cache()
gen_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto')

def score_step(model, text):
    e = tokenizer(text, truncation=True, max_length=MAX_LEN, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        return model(input_ids=e['input_ids'].to(device), attention_mask=e['attention_mask'].to(device)).logits.squeeze().item()

results_all = []
for pi, prob in enumerate(test_problems):
    if (pi+1) % 10 == 0:
        print(f"  Problem {pi+1}/{len(test_problems)}...")

    question = prob['question']
    gold = prob['final_answer']
    sols = {}

    for method in ['greedy', 'random', 'orm', 'prm']:
        if method == 'orm': blind_m.to(device)
        elif method == 'prm': prm_m.to(device)

        state = question
        steps = []
        for si in range(MAX_STEPS):
            prompt = f"Continue solving step by step. Write the next step only.\n\n{state}\n\nNext step:"
            inp = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=384).to(device)

            if method == 'greedy':
                with torch.no_grad():
                    out = gen_model.generate(**inp, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                step = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip().split('\n')[0].strip()
            else:
                cands = []
                for _ in range(N_STEP_CANDIDATES):
                    with torch.no_grad():
                        out = gen_model.generate(**inp, max_new_tokens=100, temperature=0.8, top_p=0.95,
                                                  do_sample=True, pad_token_id=tokenizer.eos_token_id)
                    c = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip().split('\n')[0].strip()
                    if c: cands.append(c)
                if not cands: break

                if method == 'random':
                    step = cands[np.random.randint(len(cands))]
                elif method == 'orm':
                    scores = [score_step(blind_m, f"Reasoning step: {c}\n\nIs this step correct?") for c in cands]
                    step = cands[np.argmax(scores)]
                elif method == 'prm':
                    scores = [score_step(prm_m, f"Problem: {state}\n\nNext step: {c}\n\nIs this step correct?") for c in cands]
                    step = cands[np.argmax(scores)]

            if not step: break
            steps.append(step)
            state = state + "\n" + step
            if '####' in step: break

        if method == 'orm': blind_m.cpu()
        elif method == 'prm': prm_m.cpu(); torch.cuda.empty_cache()

        full = "\n".join(steps)
        pred = extract_answer(full)
        sols[method] = {'correct': pred == gold if pred else False, 'answer': pred, 'nsteps': len(steps)}

    results_all.append({
        'question': question, 'gold': gold,
        **{f'{m}_correct': sols[m]['correct'] for m in sols},
        **{f'{m}_nsteps': sols[m]['nsteps'] for m in sols},
    })

del gen_model; torch.cuda.empty_cache()

# ===================================================================
# RESULTS
# ===================================================================
print("\n" + "=" * 60)
print("STEP-LEVEL BEAM SEARCH RESULTS (MATCHED DISTRIBUTION)")
print("=" * 60)

n = len(results_all)
for m in ['greedy', 'random', 'orm', 'prm']:
    acc = sum(r[f'{m}_correct'] for r in results_all) / n
    avg_s = np.mean([r[f'{m}_nsteps'] for r in results_all])
    print(f"  {m:>8}: accuracy={acc:.1%}, avg_steps={avg_s:.1f}")

g = sum(r['greedy_correct'] for r in results_all) / n
ra = sum(r['random_correct'] for r in results_all) / n
o = sum(r['orm_correct'] for r in results_all) / n
p = sum(r['prm_correct'] for r in results_all) / n

print(f"\n  PRM - ORM:    {p-o:+.1%}")
print(f"  PRM - Greedy: {p-g:+.1%}")
print(f"  ORM - Greedy: {o-g:+.1%}")

out = {
    'matched_downstream': {
        'n_test': n, 'n_candidates': N_STEP_CANDIDATES, 'max_steps': MAX_STEPS,
        'n_train_generated': len(generated), 'n_correct_generated': n_correct,
        'solve_rate': n_correct/len(generated),
        'n_train_steps': len(train_data), 'n_test_steps': len(test_data),
        'greedy': float(g), 'random': float(ra), 'orm': float(o), 'prm': float(p),
        'prm_minus_orm': float(p-o), 'prm_minus_greedy': float(p-g),
    },
    'reward_auc': {'blind': float(blind_auc), 'prm': float(prm_auc), 'lift': float(prm_auc-blind_auc)},
    'per_problem': results_all,
}

with open(os.path.join(OUT_DIR, 'r6_matched_downstream.json'), 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {OUT_DIR}/r6_matched_downstream.json")
