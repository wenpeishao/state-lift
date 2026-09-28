"""
R6 local preprocessing: save GSM8K train+test problems to a .pt file
so the SSCC script doesn't need internet.
"""
import torch
import json
from datasets import load_dataset

ds_train = load_dataset('openai/gsm8k', 'main', split='train')
ds_test = load_dataset('openai/gsm8k', 'main', split='test')

def parse_problems(ds, n):
    problems = []
    for ex in list(ds)[:n]:
        ans = ex.get('answer', '')
        final = None
        for line in reversed(ans.split('\n')):
            if '####' in line:
                final = line.split('####')[-1].strip().replace(',', '')
                break
        if final:
            problems.append({'question': ex['question'], 'final_answer': final})
    return problems

train_problems = parse_problems(ds_train, 500)
test_problems = parse_problems(ds_test, 200)

print(f"Train: {len(train_problems)}, Test: {len(test_problems)}")

torch.save({'train': train_problems, 'test': test_problems},
           'data/r6_gsm8k_problems.pt')
print("Saved to r6_gsm8k_problems.pt")
