"""
R29: PRE-REGISTERED prospective SL -- CraigslistBargain (fresh domain, never measured).
Genuine label: sale price ratio (final price / listing price) per dialogue,
trajectory-level. Grouped CV. Prediction to be recorded BEFORE any training.
"""
import numpy as np, json
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import pandas as pd

SEED=42; D_PCA=16; N_MAX=12000
np.random.seed(SEED)
enc = SentenceTransformer('all-MiniLM-L6-v2')
p = hf_hub_download('stanfordnlp/craigslist_bargains', 'default/craigslist_bargains-train.parquet',
                    repo_type='dataset', revision='refs/convert/parquet')
df = pd.read_parquet(p)
ds = df.to_dict('records')
print(len(ds), 'dialogues, keys:', list(ds[0].keys()))
S,A,Y,G = [],[],[],[]
for di, ex in enumerate(ds):
    utts = list(ex.get('utterance', []))
    items = ex.get('items', {})
    try:
        price_listed = float(items['Price'][0])
    except Exception:
        continue
    if price_listed <= 0:
        continue
    da = ex.get('dialogue_acts', {})
    prices = [float(p) for p in list(da.get('price', [])) if p is not None and p > 0]
    if not prices:
        continue
    final_price = prices[-1]
    q = max(0.0, min(final_price / price_listed, 1.5))
    turns = [u for u in utts if isinstance(u, str) and u.strip()]
    if len(turns) < 3:
        continue
    for i in range(1, len(turns)):
        S.append("Negotiation: " + " ".join(t[:80] for t in turns[:i])[:300])
        A.append(turns[i][:200]); Y.append(q); G.append(di)
Y=np.array(Y,float); G=np.array(G)
print(f"{len(Y)} tuples from {len(set(G.tolist()))} dialogues")
if len(Y) < 500:
    raise SystemExit("too few tuples -- inspect schema")
if len(Y) > N_MAX:
    idx=np.random.default_rng(SEED).choice(len(Y),N_MAX,replace=False)
    S=[S[i] for i in idx]; A=[A[i] for i in idx]; Y=Y[idx]; G=G[idx]
es=enc.encode(S,batch_size=128,show_progress_bar=False); ea=enc.encode(A,batch_size=128,show_progress_bar=False)
ps=PCA(n_components=D_PCA,random_state=SEED).fit_transform(es)
pa=PCA(n_components=D_PCA,random_state=SEED).fit_transform(ea)
Xsa=np.hstack([ps,pa])
out={}
for mode in ['random','grouped']:
    if mode=='random':
        cv=KFold(5,shuffle=True,random_state=SEED)
        r2a=cross_val_score(Ridge(1.0),pa,Y,cv=cv,scoring='r2').mean()
        r2sa=cross_val_score(Ridge(1.0),Xsa,Y,cv=cv,scoring='r2').mean()
    else:
        cv=GroupKFold(5)
        r2a=cross_val_score(Ridge(1.0),pa,Y,cv=cv,groups=G,scoring='r2').mean()
        r2sa=cross_val_score(Ridge(1.0),Xsa,Y,cv=cv,groups=G,scoring='r2').mean()
    out[mode]={'r2_action':float(r2a),'r2_state_action':float(r2sa),'sl':float(r2sa-r2a)}
    print(f"{mode:8s}: R2(a)={r2a:+.4f} R2(s+a)={r2sa:+.4f} SL={r2sa-r2a:+.4f}")
out['n']=int(len(Y))
out['preregistration']='SL measured 2026-07-24 BEFORE any reward-model training on this domain; prediction per one-directional rule to be recorded in results file by aggregating script'
json.dump(out,open('results/r29_craigslist_prereg_sl.json','w'),indent=2)
print('saved r29')
