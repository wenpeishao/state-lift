"""
E4: LLM Reward Model Training on SSCC (L40S) -- HH-RLHF Negative Control
Uses Llama-3.1-8B-Instruct with LoRA.
All data pre-processed -- no internet/HF downloads needed.

NEGATIVE CONTROL: We expect zero PRM advantage on chat quality preference.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import json
import os
import time

SEED = 42
MODEL_PATH = os.environ.get("LLAMA_PATH", "meta-llama/Llama-3.1-8B-Instruct")
DATA_PATH = "data/e4_hh_data.pt"
OUT_DIR = "data"
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-5
LORA_R = 8
LORA_ALPHA = 16
MAX_LEN = 512

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name()}")
total_mem = getattr(torch.cuda.get_device_properties(0), 'total_memory',
                    getattr(torch.cuda.get_device_properties(0), 'total_mem', 0))
print(f"VRAM: {total_mem / 1e9:.1f} GB")

# ===================================================================
# 1. LOAD PRE-PROCESSED DATA
# ===================================================================
print("\n[1/5] Loading pre-processed data...")
data = torch.load(DATA_PATH, weights_only=False)
print(f"  Train pairs: {len(data['train_chosen_blind'])}")
print(f"  Test pairs: {len(data['test_chosen_blind'])}")

# ===================================================================
# 2. DATASET
# ===================================================================
class PairDataset(Dataset):
    def __init__(self, chosen_texts, rejected_texts):
        self.chosen = chosen_texts
        self.rejected = rejected_texts

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, i):
        return self.chosen[i], self.rejected[i]

def collate_fn(batch, tokenizer):
    chosen = [b[0] for b in batch]
    rejected = [b[1] for b in batch]
    c_enc = tokenizer(chosen, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors='pt')
    r_enc = tokenizer(rejected, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors='pt')
    return {
        'c_ids': c_enc['input_ids'], 'c_mask': c_enc['attention_mask'],
        'r_ids': r_enc['input_ids'], 'r_mask': r_enc['attention_mask'],
    }

# ===================================================================
# 3. TRAINING FUNCTION
# ===================================================================
def train_and_eval(condition, train_chosen, train_rejected,
                   test_chosen, test_rejected, tokenizer):
    print(f"\n{'='*60}")
    print(f"  TRAINING: {condition}")
    print(f"{'='*60}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = PairDataset(train_chosen, train_rejected)
    test_ds = PairDataset(test_chosen, test_rejected)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_acc = 0
    best_epoch = -1
    results_by_epoch = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_correct = 0
        n_total = 0

        for step, batch in enumerate(train_loader):
            r_c = model(input_ids=batch['c_ids'].to(device),
                        attention_mask=batch['c_mask'].to(device)).logits.squeeze(-1)
            r_r = model(input_ids=batch['r_ids'].to(device),
                        attention_mask=batch['r_mask'].to(device)).logits.squeeze(-1)

            loss = -torch.log(torch.sigmoid(r_c - r_r) + 1e-8).mean() / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM
            n_correct += (r_c > r_r).sum().item()
            n_total += len(r_c)

            if (step + 1) % 200 == 0:
                print(f"    Step {step+1}/{len(train_loader)}, "
                      f"loss={total_loss/(step+1):.4f}, acc={n_correct/n_total:.4f}")

        # Eval
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in test_loader:
                r_c = model(input_ids=batch['c_ids'].to(device),
                            attention_mask=batch['c_mask'].to(device)).logits.squeeze(-1)
                r_r = model(input_ids=batch['r_ids'].to(device),
                            attention_mask=batch['r_mask'].to(device)).logits.squeeze(-1)
                eval_correct += (r_c > r_r).sum().item()
                eval_total += len(r_c)

        eval_acc = eval_correct / eval_total
        train_acc = n_correct / n_total
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.4f}, eval_acc={eval_acc:.4f}, time={elapsed:.0f}s")

        results_by_epoch.append({
            'epoch': epoch+1, 'train_acc': train_acc, 'eval_acc': eval_acc,
            'loss': total_loss/len(train_loader), 'time': elapsed,
        })

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch + 1
            model.save_pretrained(os.path.join(OUT_DIR, f"best_{condition}"))

    # State-stratified eval with best model
    print(f"\n  Best: epoch {best_epoch}, eval_acc={best_acc:.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        'best_eval_acc': best_acc,
        'best_epoch': best_epoch,
        'epochs': results_by_epoch,
    }

# ===================================================================
# 4. RUN BOTH CONDITIONS
# ===================================================================
print("\n[2/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

results = {}

# State-blind
results['blind'] = train_and_eval(
    'blind',
    data['train_chosen_blind'], data['train_rejected_blind'],
    data['test_chosen_blind'], data['test_rejected_blind'],
    tokenizer,
)

# State-conditioned
results['state_conditioned'] = train_and_eval(
    'state_conditioned',
    data['train_chosen_state'], data['train_rejected_state'],
    data['test_chosen_state'], data['test_rejected_state'],
    tokenizer,
)

# ===================================================================
# 5. STATE-STRATIFIED EVALUATION
# ===================================================================
print("\n[4/5] State-stratified evaluation...")

test_z_dist = data['test_z_dist'].numpy()

for condition, use_state_key in [('blind', 'blind'), ('state_conditioned', 'state')]:
    print(f"\n  {condition}:")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.load_adapter(os.path.join(OUT_DIR, f"best_{condition}"), adapter_name="default")
    model.eval()

    chosen_key = f'test_chosen_{use_state_key}'
    rejected_key = f'test_rejected_{use_state_key}'

    for q_label, q_low, q_high in [
        ("Q1 near-mean", 0, 25), ("Q2-Q3 moderate", 25, 75), ("Q4 extreme", 75, 100),
    ]:
        low_t = np.percentile(test_z_dist, q_low)
        high_t = np.percentile(test_z_dist, q_high)
        mask = (test_z_dist >= low_t) & (test_z_dist < high_t)

        if mask.sum() < 20:
            print(f"    {q_label}: too few ({mask.sum()})")
            continue

        q_chosen = [data[chosen_key][i] for i in range(len(mask)) if mask[i]]
        q_rejected = [data[rejected_key][i] for i in range(len(mask)) if mask[i]]

        q_ds = PairDataset(q_chosen, q_rejected)
        q_loader = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, tokenizer))

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in q_loader:
                r_c = model(input_ids=batch['c_ids'].to(device),
                            attention_mask=batch['c_mask'].to(device)).logits.squeeze(-1)
                r_r = model(input_ids=batch['r_ids'].to(device),
                            attention_mask=batch['r_mask'].to(device)).logits.squeeze(-1)
                correct += (r_c > r_r).sum().item()
                total += len(r_c)

        acc = correct / total
        results[f'{condition}_{q_label}'] = acc
        print(f"    {q_label}: acc={acc:.4f} (n={mask.sum()})")

    del model
    torch.cuda.empty_cache()

# ===================================================================
# SAVE
# ===================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS (HH-RLHF NEGATIVE CONTROL)")
print("=" * 60)

blind_acc = results['blind']['best_eval_acc']
cond_acc = results['state_conditioned']['best_eval_acc']
lift = cond_acc - blind_acc

print(f"  Blind eval acc:           {blind_acc:.4f}")
print(f"  State-conditioned acc:    {cond_acc:.4f}")
print(f"  Lift:                     {lift:+.4f}")
print(f"\n  EXPECTATION: Lift should be ~0 (no PRM advantage on chat quality)")

out_path = os.path.join(OUT_DIR, 'e4_hh_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
