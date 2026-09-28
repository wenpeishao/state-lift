"""
E6: Multi-seed LLM Reward Model Training.
Usage: python e6_multiseed_train.py --domain casino --seed 42
Domains: casino, dealornodeal, esconv, hhrlhf
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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

DOMAIN = args.domain
SEED = args.seed

MODEL_PATH = os.environ.get("LLAMA_PATH", "meta-llama/Llama-3.1-8B-Instruct")
OUT_DIR = "results/multiseed"
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
print(f"Domain: {DOMAIN}, Seed: {SEED}")
print(f"GPU: {torch.cuda.get_device_name()}")

# Find data file
data_files = {
    'casino': 'e4_casino_data.pt',
    'dealornodeal': 'e5_dealornodeal_data.pt',
    'esconv': 'e4_esconv_data.pt',
    'hhrlhf': 'e4_hh_data.pt',
}

data_file = data_files.get(DOMAIN)
if not data_file:
    raise ValueError(f"Unknown domain: {DOMAIN}")

# Try local payload first, then NFS
local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file)
nfs_path = f"data/{data_file}"
DATA_PATH = local_path if os.path.exists(local_path) else nfs_path

print(f"Data: {DATA_PATH}")
data = torch.load(DATA_PATH, weights_only=False)

meta = data.get('metadata', {})
print(f"  Dataset: {meta.get('dataset', DOMAIN)}")
print(f"  Train: {meta.get('n_train', len(data['train_chosen_blind']))}")
print(f"  Test: {meta.get('n_test', len(data['test_chosen_blind']))}")


class PairDataset(Dataset):
    def __init__(self, chosen, rejected):
        self.chosen = chosen
        self.rejected = rejected
    def __len__(self):
        return len(self.chosen)
    def __getitem__(self, i):
        return self.chosen[i], self.rejected[i]


def collate_fn(batch, tokenizer):
    chosen = [b[0] for b in batch]
    rejected = [b[1] for b in batch]
    c = tokenizer(chosen, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
    r = tokenizer(rejected, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
    return {'c_ids': c['input_ids'], 'c_mask': c['attention_mask'],
            'r_ids': r['input_ids'], 'r_mask': r['attention_mask']}


def train_and_eval(condition, train_chosen, train_rejected,
                   test_chosen, test_rejected, tokenizer):
    print(f"\n{'='*60}")
    print(f"  {DOMAIN} / {condition} / seed={SEED}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

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

    train_ds = PairDataset(train_chosen, train_rejected)
    test_ds = PairDataset(test_chosen, test_rejected)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_acc = 0
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
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: train_acc={n_correct/n_total:.4f}, eval_acc={eval_acc:.4f}, time={elapsed:.0f}s")

        if eval_acc > best_acc:
            best_acc = eval_acc

    del model
    torch.cuda.empty_cache()
    print(f"  Best eval_acc: {best_acc:.4f}")
    return best_acc


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

blind_acc = train_and_eval('blind',
    data['train_chosen_blind'], data['train_rejected_blind'],
    data['test_chosen_blind'], data['test_rejected_blind'], tokenizer)

cond_acc = train_and_eval('state_conditioned',
    data['train_chosen_state'], data['train_rejected_state'],
    data['test_chosen_state'], data['test_rejected_state'], tokenizer)

lift = cond_acc - blind_acc

print(f"\n{'='*60}")
print(f"RESULT: {DOMAIN} seed={SEED}")
print(f"  Blind:  {blind_acc:.4f}")
print(f"  PRM:    {cond_acc:.4f}")
print(f"  Lift:   {lift:+.4f}")
print(f"{'='*60}")

result = {
    'domain': DOMAIN, 'seed': SEED,
    'blind_acc': blind_acc, 'cond_acc': cond_acc, 'lift': lift,
}
out_path = os.path.join(OUT_DIR, f'{DOMAIN}_seed{SEED}.json')
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_path}")
