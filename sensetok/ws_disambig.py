"""Robust WSD cross-encoder trained on the 3.26M Sonnet-tagged occurrences.

GlossBERT-style: score([CLS] context [SEP] sense-gloss), softmax over a word's candidate
glosses, cross-entropy to the Sonnet label. One model generalises across all 19k words.
Held-out by OCCURRENCE; report accuracy overall and on BALANCED words (where the cheap
nearest-centroid propagator only hit 0.686).
"""
import json, re, collections, random
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
DEV="cuda:0" if torch.cuda.is_available() else "cpu"; MODEL="distilbert-base-uncased"

def load_data():
    meta=json.load(open("out/batches/tag_meta.json"))
    reqs=json.load(open("out/batches/tag_reqs.json"))
    tagged=json.load(open("out/tagged_senses.json"))
    inv=json.load(open("out/full_inventory.json"))
    def g(s):
        if isinstance(s,str): return s
        return s.get("gloss") or s.get("definition") or ""
    # map custom_id -> contexts
    byid={r["custom_id"]:r["user"] for r in reqs}
    data=[]  # (context, [glosses], gold)
    for cid,(w,ci,n) in meta.items():
        if cid not in byid or w not in tagged or w not in inv: continue
        glosses=[g(s) for s in inv[w]]
        if len(glosses)<2: continue
        lines=byid[cid].split("\n")
        ctxs=[re.sub(r'^\d+\.\s*','',ln) for ln in lines if re.match(r'^\d+\.',ln)]
        for k,c in enumerate(ctxs):
            gi=str(ci+k)
            if gi in tagged[w] and tagged[w][gi]<len(glosses):
                data.append((c, glosses, tagged[w][gi], w))
    return data

class XEnc(nn.Module):
    def __init__(self):
        super().__init__(); self.bert=AutoModel.from_pretrained(MODEL); self.head=nn.Linear(self.bert.config.dim,1)
    def forward(self, enc):
        h=self.bert(**enc).last_hidden_state[:,0]; return self.head(h).squeeze(-1)

def main():
    data=load_data()
    print(f"training examples: {len(data):,}", flush=True)
    rng=random.Random(0); rng.shuffle(data)
    nte=min(30000,len(data)//10); te=data[:nte]; tr=data[nte:]
    tok=AutoTokenizer.from_pretrained(MODEL); m=XEnc().to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-5)
    def batch_encode(items):
        pairs=[]; owner=[]; gold=[]
        for c,gl,s,w in items:
            for gi,gtext in enumerate(gl):
                pairs.append((c[:300], gtext)); owner.append(len(gold))
            gold.append(s)
        enc=tok([p[0] for p in pairs],[p[1] for p in pairs],truncation=True,max_length=96,
                padding=True,return_tensors="pt")
        return {k:v.to(DEV) for k,v in enc.items()}, owner, gold, [len(x[1]) for x in items]
    STEPS=3000; BS=16
    m.train()
    for step in range(STEPS):
        items=[tr[rng.randrange(len(tr))] for _ in range(BS)]
        enc,owner,gold,klens=batch_encode(items)
        sc=m(enc)
        loss=0; off=0
        for i,k in enumerate(klens):
            logits=sc[off:off+k]; loss=loss+nn.functional.cross_entropy(logits[None],torch.tensor([gold[i]],device=DEV)); off+=k
        loss=loss/BS
        opt.zero_grad(); loss.backward(); opt.step()
        if step%500==0: print(f"  step {step} loss {loss.item():.3f}", flush=True)
    # eval
    m.eval(); correct=collections.Counter(); tot=collections.Counter()
    # balanced = word whose gold senses are mixed in the test set
    wsense=collections.defaultdict(collections.Counter)
    for c,gl,s,w in te: wsense[w][s]+=1
    with torch.no_grad():
        for i in range(0,len(te),16):
            items=te[i:i+16]; enc,owner,gold,klens=batch_encode(items); sc=m(enc).cpu().numpy(); off=0
            for j,k in enumerate(klens):
                pred=int(np.argmax(sc[off:off+k])); off+=k
                c,gl,s,w=items[j]
                top=wsense[w].most_common(1)[0][1]/sum(wsense[w].values())
                bucket="balanced" if top<0.75 else "skewed"
                for b in (bucket,"all"):
                    tot[b]+=1; correct[b]+= (pred==s)
    print(f"\nWSD cross-encoder (trained on {len(tr):,} Sonnet labels):")
    for b in ("all","balanced","skewed"):
        if tot[b]: print(f"  {b:<10} acc {correct[b]/tot[b]:.3f}  (n={tot[b]:,})")
    print(f"\ncompare: nearest-centroid propagator was 0.879 all / 0.686 balanced")
    torch.save(m.state_dict(),"out/wsd_xenc.pt")

if __name__=="__main__":
    main()
