"""
E6: Multi-seed LLM Reward Model Training -- home 5090 version.
Usage: python e6_multiseed_5090.py --domain casino --seed 42
Domains: casino, dealornodeal, esconv, hhrlhf
Adapted from scripts/training/e6_multiseed_train.py
(paths made relative to the repo root; torch_dtype -> dtype for transformers 5.x)
"""

import torch
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
parser.add_argument('--smoke', action='store_true', help='tiny run to verify pipeline')
parser.add_argument('--save_adapter', action='store_true', help='save LoRA adapter after training each arm')
parser.add_argument('--model_path', type=str, default=None, help='override base model path')
parser.add_argument('--tag', type=str, default='', help='suffix for result filename (e.g. model family)')
parser.add_argument('--eval_every', type=int, default=0, help='if >0, eval on test every N optimizer steps and record the learning curve')
parser.add_argument('--n_train', type=int, default=0,
                    help='if >0, subsample the training pairs to this many. Stage 2 of the '
                         'protocol is cheap along the STEP axis; this tests the DATA axis, '
                         'i.e. whether the blind-vs-conditioned gap survives when the label '
                         'budget rather than the compute budget is what is scarce.')
parser.add_argument('--shuffle_state', action='store_true',
                    help='NULL CONTROL: permute state prefixes across pairs, keeping action text, '
                         'labels, input format and sequence length intact. The conditioned arm then '
                         'has no valid state information; any surviving lift is a format artifact.')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate (default 2e-5 = the value used for every reported run). '
                         'Per-domain HP check for the camera-ready (Reviewer MmQK): both arms '
                         'get the SAME lr, so the comparison stays matched.')
parser.add_argument('--epochs', type=int, default=3, help='training epochs (default 3 = reported runs)')
args = parser.parse_args()

DOMAIN = args.domain
SEED = args.seed

MODEL_PATH = args.model_path or os.environ.get("LLAMA_PATH", "meta-llama/Llama-3.1-8B-Instruct")
OUT_DIR = "results/multiseed"
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = args.epochs
LR = args.lr
# Non-default HPs get an automatic filename tag so they never overwrite the reported runs.
if (args.lr != 2e-5 or args.epochs != 3) and not args.tag:
    args.tag = f"_hp_lr{args.lr:g}_ep{args.epochs}"
LORA_R = 8
LORA_ALPHA = 16
MAX_LEN = 512

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda')
print(f"Domain: {DOMAIN}, Seed: {SEED}", flush=True)
print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

data_files = {
    'casino': 'e4_casino_data.pt',
    'dealornodeal': 'e5_dealornodeal_data.pt',
    'esconv': 'e4_esconv_data.pt',
    'hhrlhf': 'e4_hh_data.pt',
    'casino_dsplit': 'e8_casino_dsplit_data.pt',
    'dealornodeal_dsplit': 'e8_dealornodeal_dsplit_data.pt',
    'p4g': 'e9_p4g_data.pt',
    'p4g_censored': 'e9b_p4g_censored_data.pt',
    'p4g_truncated': 'e9c_p4g_truncated_data.pt',
    'p4g_lastk': 'e9d_p4g_lastk_data.pt',
    'craigslist': 'e10_craigslist_data.pt',
    'p4g_pdisjoint': 'e9e_p4g_pdisjoint_data.pt',
    'craigslist_ldisjoint': 'e10b_craigslist_ldisjoint_data.pt',
    'multiwoz': 'e11_multiwoz_data.pt',
    'constraints': 'e12_constraints_data.pt',
    'constraints_v2': 'e13_constraints_v2_data.pt',
    'arithmetic': 'e14_arithmetic_data.pt',
    'constraints_v3': 'e15_constraints_v3_data.pt',
    'constraints_bal': 'e16_constraints_balanced.pt',
    'multiwoz_precode': 'e11d_multiwoz_precode_data.pt',
    'prm800k': 'e17_prm800k_pairs.pt',
    'additive_pooled': 'e18_additive_pooled.pt',
    'additive_within': 'e18_additive_within.pt',
    'salient_pooled': 'e19_salient_pooled.pt',
    'salient_within': 'e19_salient_within.pt',
}

DATA_PATH = f"data/{data_files[DOMAIN]}"
print(f"Data: {DATA_PATH}", flush=True)
data = torch.load(DATA_PATH, weights_only=False)

meta = data.get('metadata', {})
print(f"  Train pairs: {len(data['train_chosen_blind'])}", flush=True)
print(f"  Test pairs:  {len(data['test_chosen_blind'])}", flush=True)

if args.shuffle_state:
    # NULL CONTROL for the early-training-signal (ETS) diagnostic. The conditioned
    # arm keeps its prompt format, its length, and its action text; only the pairing
    # between state prefix and action is destroyed. A conditioned model can therefore
    # still exploit any format/length/capacity advantage, but no genuine state
    # information. Any surviving gap over the blind arm is an artifact of the arm
    # itself rather than evidence of state-dependent quality.
    def _split_prefix(blind, cond):
        i = 0
        while i < min(len(blind), len(cond)) and blind[len(blind) - 1 - i] == cond[len(cond) - 1 - i]:
            i += 1
        return cond[:len(cond) - i], cond[len(cond) - i:]

    _rng = np.random.default_rng(SEED)
    for _split in ('train', 'test'):
        _bc = data[f'{_split}_chosen_blind']; _sc = data[f'{_split}_chosen_state']
        _br = data[f'{_split}_rejected_blind']; _sr = data[f'{_split}_rejected_state']
        _n = min(len(_bc), len(_sc), len(_br), len(_sr))
        _pre = [_split_prefix(_bc[i], _sc[i])[0] for i in range(_n)]
        _act_c = [_split_prefix(_bc[i], _sc[i])[1] for i in range(_n)]
        _act_r = [_split_prefix(_br[i], _sr[i])[1] for i in range(_n)]
        _perm = _rng.permutation(_n)
        data[f'{_split}_chosen_state'] = [_pre[_perm[i]] + _act_c[i] for i in range(_n)]
        data[f'{_split}_rejected_state'] = [_pre[_perm[i]] + _act_r[i] for i in range(_n)]
        for _k in ('chosen_blind', 'rejected_blind'):
            data[f'{_split}_{_k}'] = data[f'{_split}_{_k}'][:_n]
    print(f"SHUFFLE_STATE null control active: state prefixes permuted across pairs "
          f"(mean prefix chars {np.mean([len(p) for p in _pre]):.0f})", flush=True)

if args.n_train and args.n_train < len(data['train_chosen_blind']):
    # DATA-budget lever for Stage 2. Both arms get the SAME subsample, so the comparison
    # stays matched; only the number of labelled pairs changes.
    _idx = np.random.default_rng(SEED).choice(len(data['train_chosen_blind']),
                                              args.n_train, replace=False)
    for _k in ('train_chosen_blind', 'train_rejected_blind',
               'train_chosen_state', 'train_rejected_state'):
        data[_k] = [data[_k][i] for i in _idx]
    print(f"N_TRAIN: subsampled to {args.n_train} training pairs", flush=True)

if args.smoke:
    for k in ['train_chosen_blind', 'train_rejected_blind',
              'train_chosen_state', 'train_rejected_state']:
        data[k] = data[k][:32]
    for k in ['test_chosen_blind', 'test_rejected_blind',
              'test_chosen_state', 'test_rejected_state']:
        data[k] = data[k][:16]
    EPOCHS = 1
    print("SMOKE MODE: 32 train / 16 test / 1 epoch", flush=True)


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
    print(f"\n{'='*60}", flush=True)
    print(f"  {DOMAIN} / {condition} / seed={SEED}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=1, dtype=torch.bfloat16,
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
    curve = []
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

            if args.eval_every and (step + 1) % (args.eval_every * GRAD_ACCUM) == 0:
                model.eval(); ec = et = 0
                with torch.no_grad():
                    for b in test_loader:
                        rc = model(input_ids=b['c_ids'].to(device), attention_mask=b['c_mask'].to(device)).logits.squeeze(-1)
                        rr = model(input_ids=b['r_ids'].to(device), attention_mask=b['r_mask'].to(device)).logits.squeeze(-1)
                        ec += (rc > rr).sum().item(); et += len(rc)
                curve.append({'opt_step': (step + 1) // GRAD_ACCUM, 'epoch': epoch, 'eval_acc': ec / et})
                print(f"    [curve] step {(step+1)//GRAD_ACCUM}: eval_acc={ec/et:.4f}", flush=True)
                model.train()

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
        print(f"  Epoch {epoch+1}: train_acc={n_correct/n_total:.4f}, eval_acc={eval_acc:.4f}, time={elapsed:.0f}s", flush=True)

        if eval_acc > best_acc:
            best_acc = eval_acc

    if args.save_adapter:
        adir = os.path.join(OUT_DIR, 'adapters')
        os.makedirs(adir, exist_ok=True)
        apath = os.path.join(adir, f"{DOMAIN}_{condition}_seed{SEED}")
        model.save_pretrained(apath)
        print(f"  Adapter saved: {apath}", flush=True)
    del model
    torch.cuda.empty_cache()
    print(f"  Best eval_acc: {best_acc:.4f}", flush=True)
    return best_acc, curve


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

blind_acc, blind_curve = train_and_eval('blind',
    data['train_chosen_blind'], data['train_rejected_blind'],
    data['test_chosen_blind'], data['test_rejected_blind'], tokenizer)

cond_acc, cond_curve = train_and_eval('state_conditioned',
    data['train_chosen_state'], data['train_rejected_state'],
    data['test_chosen_state'], data['test_rejected_state'], tokenizer)

lift = cond_acc - blind_acc

print(f"\n{'='*60}", flush=True)
print(f"RESULT: {DOMAIN} seed={SEED}", flush=True)
print(f"  Blind:  {blind_acc:.4f}", flush=True)
print(f"  PRM:    {cond_acc:.4f}", flush=True)
print(f"  Lift:   {lift:+.4f}", flush=True)
print(f"{'='*60}", flush=True)

result = {
    'domain': DOMAIN, 'seed': SEED, 'blind_curve': blind_curve, 'cond_curve': cond_curve,
    'blind_acc': blind_acc, 'cond_acc': cond_acc, 'lift': lift,
    'lr': LR, 'epochs': EPOCHS,
}
suffix = '_smoke' if args.smoke else ''
out_path = os.path.join(OUT_DIR, f'{DOMAIN}{args.tag}_seed{SEED}{suffix}.json')
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_path}", flush=True)
