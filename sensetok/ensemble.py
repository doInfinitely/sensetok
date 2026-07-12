"""Ensemble: propagator for SKEWED words, cross-encoder v2 for BALANCED words.

Both scored on the SAME held-out set (seed-0 split, matching v2), routing decided by the
TRAIN sense distribution (no test-label leakage). Tests whether routing beats either alone.
"""
import os, json, collections, random
import numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE","1")
from transformers import AutoTokenizer, AutoModel
from sensetok.ws_disambig2 import load_data, XEnc, batch_encode, MODEL
DEV="cuda:0" if torch.cuda.is_available() else "cpu"; L4=(-4,-3,-2,-1)

@torch.no_grad()
def embed_prop(items, tok, mod, bs=32):
    # frozen distilbert, target-span mean-pool of last-4 layers (the propagator's features)
    from sensetok.ws_disambig2 import clean_target
    V=np.zeros((len(items),mod.config.dim),dtype=np.float32)
    for i in range(0,len(items),bs):
        b=items[i:i+bs]; texts=[]; spans=[]
        for c,gl,s,w in b:
            cl,sp=clean_target(c); texts.append(cl[:300]); spans.append(sp)
        enc=tok(texts,truncation=True,max_length=64,padding=True,return_offsets_mapping=True,return_tensors="pt")
        offs=enc.pop("offset_mapping")
        out=mod(**{k:v.to(DEV) for k,v in enc.items()},output_hidden_states=True)
        h=sum(out.hidden_states[l] for l in L4)
        for j,(a,bb) in enumerate(spans):
            sel=[p for p in range(offs.shape[1]) if int(offs[j][p][0])!=int(offs[j][p][1])
                 and not (int(offs[j][p][1])<=a or int(offs[j][p][0])>=bb)] or [0]
            V[i+j]=h[j,sel].mean(0).float().cpu().numpy()
    return V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9)

def main():
    data=load_data(); rng=random.Random(0); rng.shuffle(data)
    nte=min(30000,len(data)//10); te=data[:nte]; tr=data[nte:]
    # train sense distribution per word (for routing + propagator centroids)
    trbyw=collections.defaultdict(list)
    for it in tr: trbyw[it[3]].append(it)
    tok=AutoTokenizer.from_pretrained(MODEL); mod=AutoModel.from_pretrained(MODEL).to(DEV).eval()
    # cross-encoder predictions on te
    m=XEnc().to(DEV); m.load_state_dict(torch.load("out/wsd_xenc2_uncased_s0.pt",map_location=DEV)); m.eval()
    xe_pred={}
    with torch.no_grad():
        for i in range(0,len(te),16):
            items=te[i:i+16]; enc,tgt,gold,kl=batch_encode(items); sc=m(enc,tgt).cpu().numpy(); off=0
            for j,k in enumerate(kl): xe_pred[i+j]=int(np.argmax(sc[off:off+k])); off+=k
    # propagator: build centroids per test word from its TRAIN occurrences (cap 60), predict te
    words=set(it[3] for it in te)
    cents={}
    for w in words:
        tt=trbyw.get(w,[])[:60]
        if not tt: continue
        V=embed_prop(tt,tok,mod); cc=collections.defaultdict(list)
        for k,it in enumerate(tt): cc[it[2]].append(V[k])
        cents[w]={s:np.mean(v,0) for s,v in cc.items()}
    Vte=embed_prop(te,tok,mod)
    # route by train skew
    trdist={w:collections.Counter(it[2] for it in trbyw[w]) for w in words}
    def skewed(w):
        d=trdist.get(w); 
        return True if not d else (max(d.values())/sum(d.values()))>=0.75
    res=collections.Counter(); tot=collections.Counter()
    for i,(c,gl,s,w) in enumerate(te):
        # propagator pred
        pp=0
        if w in cents and cents[w]:
            pp=max(cents[w],key=lambda ss: float(Vte[i]@cents[w][ss]))
        xp=xe_pred[i]
        route= pp if skewed(w) else xp
        for name,pred in (("propagator",pp),("cross-enc",xp),("ensemble",route)):
            tot[name]+=1; res[name]+= (pred==s)
        b="skewed" if skewed(w) else "balanced"
        tot[(b,"ensemble")]+=1; res[(b,"ensemble")]+= (route==s)
    print("\nENSEMBLE (common held-out, routed by train skew):")
    for n in ("propagator","cross-enc","ensemble"):
        print(f"  {n:<12} {res[n]/tot[n]:.3f}  (n={tot[n]:,})")
    for b in ("skewed","balanced"):
        print(f"  ensemble[{b}] {res[(b,'ensemble')]/tot[(b,'ensemble')]:.3f}  (n={tot[(b,'ensemble')]:,})")

if __name__=="__main__":
    main()
