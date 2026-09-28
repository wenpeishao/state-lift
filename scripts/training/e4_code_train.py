"""
E4: PRM vs ORM Code Reward Model Training on SSCC (L40S)
Uses Llama-3.1-8B-Instruct with LoRA for binary classification.
Compares state-blind (ORM) vs state-conditioned (PRM) on math step quality.
All data pre-processed -- no internet/HF downloads needed.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import json
import os
import time

SEED = 42
MODEL_PATH = os.environ.get("LLAMA_PATH", "meta-llama/Llama-3.1-8B-Instruct")
DATA_PATH = "data/e4_code_data.pt"
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
total_mem = torch.cuda.get_device_properties(0).total_memory
print(f"VRAM: {total_mem / 1e9:.1f} GB")

# ===================================================================
# 1. LOAD PRE-PROCESSED DATA
# ===================================================================
print("\n[1/4] Loading pre-processed data...")
data = torch.load(DATA_PATH, weights_only=False)

meta = data['metadata']
print(f"  Train pairs: {meta['n_train']}")
print(f"  Test pairs: {meta['n_test']}")
print(f"  Train chains: {meta.get('n_chains_train', 'N/A')}")
print(f"  Test chains: {meta.get('n_chains_test', 'N/A')}")
print(f"  Dataset: {meta['dataset']}")

# ===================================================================
# 2. DATASET
# ===================================================================
class BinaryDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i].item()


def collate_fn(batch, tokenizer):
    texts = [b[0] for b in batch]
    labels = torch.FloatTensor([b[1] for b in batch])
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors='pt')
    return {
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'labels': labels,
    }


# ===================================================================
# 3. TRAINING FUNCTION
# ===================================================================
def train_and_eval(condition, train_texts, train_labels,
                   test_texts, test_labels, tokenizer):
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

    train_ds = BinaryDataset(train_texts, train_labels)
    test_ds = BinaryDataset(test_texts, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    bce_loss = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_epoch = -1
    results_by_epoch = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_correct = 0
        n_total = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
            ).logits.squeeze(-1)

            labels = batch['labels'].to(device)
            loss = bce_loss(logits, labels) / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM
            preds = (logits > 0).float()
            n_correct += (preds == labels).sum().item()
            n_total += len(labels)

            if (step + 1) % 100 == 0:
                print(f"    Step {step+1}/{len(train_loader)}, "
                      f"loss={total_loss/(step+1):.4f}, "
                      f"acc={n_correct/n_total:.4f}")

        # Final gradient step if leftover
        if len(train_loader) % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Eval
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                ).logits.squeeze(-1)
                all_logits.append(logits.cpu())
                all_labels.append(batch['labels'])

        all_logits = torch.cat(all_logits).float().numpy()
        all_labels_np = torch.cat(all_labels).float().numpy()

        probs = 1.0 / (1.0 + np.exp(-all_logits))
        auc = roc_auc_score(all_labels_np, probs)
        preds = (all_logits > 0).astype(float)
        acc = accuracy_score(all_labels_np, preds)

        train_acc = n_correct / n_total
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.4f}, eval_acc={acc:.4f}, "
              f"eval_auc={auc:.4f}, time={elapsed:.0f}s")

        results_by_epoch.append({
            'epoch': epoch + 1,
            'train_acc': float(train_acc),
            'eval_acc': float(acc),
            'eval_auc': float(auc),
            'loss': float(total_loss / len(train_loader)),
            'time': float(elapsed),
        })

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            model.save_pretrained(os.path.join(OUT_DIR, f"best_{condition}"))

    print(f"\n  Best: epoch {best_epoch}, AUC={best_auc:.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        'best_eval_auc': float(best_auc),
        'best_eval_acc': float(results_by_epoch[best_epoch - 1]['eval_acc']),
        'best_epoch': best_epoch,
        'epochs': results_by_epoch,
    }


# ===================================================================
# 4. RUN BOTH CONDITIONS
# ===================================================================
print("\n[2/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

results = {}

# Model A: ORM (state-blind)
results['blind'] = train_and_eval(
    'blind',
    data['train_texts_blind'], data['train_labels'],
    data['test_texts_blind'], data['test_labels'],
    tokenizer,
)

# Model B: PRM (state-conditioned)
results['conditioned'] = train_and_eval(
    'conditioned',
    data['train_texts_conditioned'], data['train_labels'],
    data['test_texts_conditioned'], data['test_labels'],
    tokenizer,
)

# ===================================================================
# RESULTS SUMMARY
# ===================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS (CODE)")
print("=" * 60)

blind_auc = results['blind']['best_eval_auc']
blind_acc = results['blind']['best_eval_acc']
cond_auc = results['conditioned']['best_eval_auc']
cond_acc = results['conditioned']['best_eval_acc']
lift_auc = cond_auc - blind_auc
lift_acc = cond_acc - blind_acc

print(f"  Blind (ORM):        AUC={blind_auc:.4f}  Acc={blind_acc:.4f}")
print(f"  Conditioned (PRM):  AUC={cond_auc:.4f}  Acc={cond_acc:.4f}")
print(f"  Lift:               AUC={lift_auc:+.4f}  Acc={lift_acc:+.4f}")

results['summary'] = {
    'blind_auc': blind_auc,
    'blind_acc': blind_acc,
    'conditioned_auc': cond_auc,
    'conditioned_acc': cond_acc,
    'lift_auc': lift_auc,
    'lift_acc': lift_acc,
}
results['metadata'] = data['metadata']

out_path = os.path.join(OUT_DIR, 'e4_code_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
