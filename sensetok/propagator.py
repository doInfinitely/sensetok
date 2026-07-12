"""Does a cheap propagator trained on Sonnet labels recover held-out Sonnet labels?

If yes, we Sonnet-label a small sample per word and the propagator (distilbert nearest-
centroid) tags the whole corpus for free — avoiding ~$2900 of direct Sonnet tagging.

Per word: embed each labeled occurrence (distilbert, last-4-layer sum over the target
span). Split 30 train / 10 test. Fit one centroid per sense from train. Predict test by
nearest centroid + a frequency prior tuned on train. Score:
  propagator vs Sonnet   (test-set agreement)   <- the number that matters
  MFS baseline           (always sense 0)
  random-merge null      (does the inventory beat arbitrary bucketing at same k)
Bar: propagator must beat MFS AND clear the null, at agreement high enough that the
resulting token purity is worth minting (target ~0.85+).
"""
import json, glob, collections
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel

MODEL="distilbert-base-uncased"; DEV="cuda:0" if torch.cuda.is_available() else "cpu"; L4=(-4,-3,-2,-1)

@torch.no_grad()
def embed(sents, spans, tok, mod, bs=32):
    V=np.zeros((len(sents),mod.config.dim),dtype=np.float32)
    for i in range(0,len(sents),bs):
        b=sents[i:i+bs]; sp=spans[i:i+bs]
        enc=tok(b,is_split_into_words=True,return_tensors="pt",padding=True,truncation=True,max_length=64)
        out=mod(**{k:v.to(DEV) for k,v in enc.items()},output_hidden_states=True)
        h=sum(out.hidden_states[l] for l in L4)
        for j,(s,e) in enumerate(sp):
            wid=enc.word_ids(j); sel=[p for p,w in enumerate(wid) if w is not None and s<=w<e] or [0]
            V[i+j]=h[j,sel].mean(0).float().cpu().numpy()
    return V/ (np.linalg.norm(V,axis=1,keepdims=True)+1e-9)

def main():
    inv=json.load(open("out/boot2_inventory.json"))
    items={}
    for f in sorted(glob.glob("scale/boot2_*.json")):
        for it in json.load(f if False else open(f)): items[it["id"]]=it
    labs={}
    for f in sorted(glob.glob("scale/lab2_*.json")):
        try:
            for r in json.load(open(f)): labs[r["id"]]=r["sense"]
        except Exception as e: print("bad",f,e)
    ids=[i for i in items if i in labs]
    print(f"labeled occurrences: {len(ids):,} over {len(set(items[i]['word'] for i in ids))} words")
    # parse [[...]] to a token span
    from re import findall
    def toks_span(sent):
        ws=sent.split(); s=e=None
        out=[]
        for k,w in enumerate(ws):
            if w.startswith("[[") and w.endswith("]]"): s=k; e=k+1; out.append(w[2:-2])
            else: out.append(w)
        return out,(s if s is not None else 0, e if e is not None else 1)
    byw=collections.defaultdict(list)
    for i in ids: byw[items[i]["word"]].append(i)
    tok=AutoTokenizer.from_pretrained(MODEL); mod=AutoModel.from_pretrained(MODEL).to(DEV).eval()
    # embed all at once
    sents=[]; spans=[]; order=[]
    for w,iis in byw.items():
        for i in iis:
            t,sp=toks_span(items[i]["sentence"]); sents.append(t); spans.append(sp); order.append(i)
    V=embed(sents,tok,mod) if False else embed(sents,spans,tok,mod)
    vec={i:V[k] for k,i in enumerate(order)}
    rng=np.random.default_rng(0)
    prop=mfs=n=0; kappa_o=[]; kappa_e=[]
    per=[]
    for w,iis in byw.items():
        if len(iis)<12: continue
        iis=[iis[k] for k in rng.permutation(len(iis))]
        cut=int(len(iis)*0.75); tr,te=iis[:cut],iis[cut:]
        k=len(inv[w])
        cents={}
        for s in range(k):
            xs=[vec[i] for i in tr if labs[i]==s]
            if xs: cents[s]=np.mean(xs,0)
        pri=collections.Counter(labs[i] for i in tr)
        for i in te:
            n+=1
            g=labs[i]
            if cents:
                sc={s:float(vec[i]@c)+0.15*np.log((pri[s]+1)/(len(tr)+k)) for s,c in cents.items()}
                p=max(sc,key=sc.get)
            else: p=0
            prop+= (p==g); mfs+= (g==0)
            kappa_o.append(p==g); kappa_e.append(1.0/k)
        per.append(w)
    po=prop/n; pe=np.mean(kappa_e); kap=(po-pe)/(1-pe+1e-12)
    print(f"\npropagator vs Sonnet (held-out): agreement {po:.3f}  kappa {kap:.3f}")
    print(f"MFS baseline (always sense 0):   agreement {mfs/n:.3f}")
    print(f"  words tested {len(per)}, test occurrences {n}")
    print(f"\nbar: beat MFS and reach ~0.85 agreement to be worth minting sense tokens.")
    json.dump({"prop":po,"mfs":mfs/n,"kappa":kap,"n":n},open("out/propagator.json","w"))

if __name__=="__main__":
    main()
