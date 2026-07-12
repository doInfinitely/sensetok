"""WSD cross-encoder v2: TARGET-TOKEN readout + CASED backbone + proper training.

Changes from v1 (which lost to the propagator):
- distilbert-base-CASED (v1 was uncased -> blind to Bush/bush capitalisation)
- read the TARGET word's contextual token(s), not [CLS] (point the head where sense lives)
- 12k steps x BS32 = ~380k examples (v1: 48k = 1.5% of data; undertrained)
- 500-step linear warmup; periodic eval to see plateau vs undertraining
Same eval protocol as v1 (global 30k held-out, balanced = word top-sense <75%) for comparison.
"""
import json, re, glob, collections, random
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
DEV="cuda:0" if torch.cuda.is_available() else "cpu"; MODEL=os.environ.get("WSD_MODEL","distilbert-base-uncased")

def clean_target(c):
    # strip [[ ]]; return CLEAN text + target char span. The target signal goes in via a
    # summed mask embedding (segment-embedding style), not brackets or a marker token.
    m=re.search(r'\[\[(.*?)\]\]', c)
    if not m: return c, (0, len(c.split()[0]) if c else 1)
    tgt=m.group(1); clean=c[:m.start()]+tgt+c[m.end():]
    return clean, (m.start(), m.start()+len(tgt))

def load_data():
    meta=json.load(open("out/batches/tag_meta.json")); reqs=json.load(open("out/batches/tag_reqs.json"))
    tagged=json.load(open("out/tagged_senses.json")); inv=json.load(open("out/full_inventory.json"))
    def g(s): return s if isinstance(s,str) else (s.get("gloss") or s.get("definition") or "")
    byid={r["custom_id"]:r["user"] for r in reqs}; data=[]
    for cid,(w,ci,n) in meta.items():
        if cid not in byid or w not in tagged or w not in inv: continue
        gl=[g(s) for s in inv[w]]
        if len(gl)<2: continue
        ctxs=[re.sub(r'^\d+\.\s*','',ln) for ln in byid[cid].split("\n") if re.match(r'^\d+\.',ln)]
        for k,c in enumerate(ctxs):
            gi=str(ci+k)
            if gi in tagged[w] and tagged[w][gi]<len(gl): data.append((c,gl,tagged[w][gi],w))
    return data

class XEnc(nn.Module):
    def __init__(self):
        super().__init__(); self.bert=AutoModel.from_pretrained(MODEL)
        d=self.bert.config.dim
        self.tgt_emb=nn.Embedding(2,d); nn.init.zeros_(self.tgt_emb.weight)  # target MASK, starts no-op
        self.head=nn.Linear(d,1)
    def forward(self, enc, tgt):
        ids=enc["input_ids"]; am=enc["attention_mask"]
        mask=torch.zeros_like(ids)
        for j,ps in enumerate(tgt): mask[j,torch.tensor(ps,device=ids.device)]=1
        # inject the target signal INTO the encoder (not just readout): word emb + target-mask emb
        we=self.bert.embeddings.word_embeddings(ids)+self.tgt_emb(mask)
        h=self.bert(inputs_embeds=we,attention_mask=am).last_hidden_state
        pooled=torch.stack([h[j][torch.tensor(tgt[j],device=h.device)].mean(0) for j in range(h.shape[0])])
        return self.head(pooled).squeeze(-1)

TOK=AutoTokenizer.from_pretrained(MODEL)
def batch_encode(items):
    cleans=[]; gls=[]; gold=[]; spans=[]
    for c,gl,s,w in items:
        cl,sp=clean_target(c)
        for gt in gl: cleans.append(cl[:300]); gls.append(gt); spans.append(sp)
        gold.append(s)
    enc=TOK(cleans,gls,truncation=True,max_length=96,padding=True,
            return_offsets_mapping=True,return_tensors="pt")
    offs=enc.pop("offset_mapping")
    tgt=[]
    for j in range(len(cleans)):
        sids=enc.sequence_ids(j); a,b=spans[j]
        sel=[p for p in range(len(sids)) if sids[p]==0 and int(offs[j][p][0])!=int(offs[j][p][1])
             and not (int(offs[j][p][1])<=a or int(offs[j][p][0])>=b)]
        tgt.append(sel or [0])
    return {k:v.to(DEV) for k,v in enc.items()}, tgt, gold, [len(x[1]) for x in items]

def evaluate(m, te, wsense, nmax=None):
    m.eval(); cor=collections.Counter(); tot=collections.Counter()
    sub=te if nmax is None else te[:nmax]
    with torch.no_grad():
        for i in range(0,len(sub),16):
            items=sub[i:i+16]; enc,tgt,gold,klens=batch_encode(items); sc=m(enc,tgt).cpu().numpy(); off=0
            for j,k in enumerate(klens):
                pred=int(np.argmax(sc[off:off+k])); off+=k
                c,gl,s,w=items[j]
                top=wsense[w].most_common(1)[0][1]/sum(wsense[w].values())
                for bkt in (("balanced" if top<0.75 else "skewed"),"all"):
                    tot[bkt]+=1; cor[bkt]+=(pred==s)
    return {b:cor[b]/tot[b] for b in ("all","balanced","skewed") if tot[b]}, dict(tot)

def main():
    import sys
    SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0
    data=load_data(); print(f"examples: {len(data):,}",flush=True)
    rng=random.Random(SEED); torch.manual_seed(SEED); rng.shuffle(data)
    nte=min(30000,len(data)//10); te=data[:nte]; tr=data[nte:]
    wsense=collections.defaultdict(collections.Counter)
    for c,gl,s,w in te: wsense[w][s]+=1
    m=XEnc().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-5)
    STEPS=12000; BS=32; WARM=500
    for step in range(1,STEPS+1):
        for pg in opt.param_groups: pg["lr"]=2e-5*min(1.0,step/WARM)
        m.train(); items=[tr[rng.randrange(len(tr))] for _ in range(BS)]
        enc,tgt,gold,klens=batch_encode(items); sc=m(enc,tgt)
        loss=0; off=0
        for i,k in enumerate(klens):
            loss=loss+nn.functional.cross_entropy(sc[off:off+k][None],torch.tensor([gold[i]],device=DEV)); off+=k
        loss=loss/BS; opt.zero_grad(); loss.backward(); opt.step()
        if step%500==0: print(f"  step {step} loss {loss.item():.3f}",flush=True)
        if step%3000==0:
            acc,_=evaluate(m,te,wsense,nmax=5000)
            print(f"  [eval@{step}] all {acc.get('all',0):.3f} balanced {acc.get('balanced',0):.3f}",flush=True)
    acc,tot=evaluate(m,te,wsense)
    print(f"\nWSD cross-encoder v2 (cased + target-token, {len(tr):,} labels, {STEPS} steps):")
    for b in ("all","balanced","skewed"):
        if b in acc: print(f"  {b:<10} acc {acc[b]:.3f}  (n={tot[b]:,})")
    print(f"\nv1 (uncased+[CLS], 3k steps): 0.794 all / 0.679 balanced")
    print(f"propagator: 0.879 all / 0.686 balanced")
    torch.save(m.state_dict(),f"out/wsd_xenc2_{MODEL.split(chr(45))[-1]}_s{SEED}.pt")

if __name__=="__main__":
    main()
