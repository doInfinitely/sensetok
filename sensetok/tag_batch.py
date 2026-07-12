"""Direct Sonnet sense-tagging of wikitext via the Batch API (thinking OFF, batched).

One request = one word's inventory + up to CHUNK occurrences -> a sense index per occurrence.
Requires out/scale_inventory.json (2k) + inv24 results merged into out/full_inventory.json.
"""
import json, collections, re, sys, os
from datasets import load_dataset
WORD=re.compile(r"[A-Za-z']+")
CHUNK=40

def gather(words_inv, cap):
    ds=load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    occ=collections.defaultdict(list); n=0
    want=set(words_inv)
    for r in ds:
        t=r["text"]
        if len(t)>200 and not t.lstrip().startswith("="):
            toks=WORD.findall(t)
            for i,w in enumerate(toks):
                lw=w.lower()
                if lw in want and len(occ[lw])<cap:
                    s=max(0,i-10);e=min(len(toks),i+10)
                    seg=toks[s:e]; seg[i-s]="[["+seg[i-s]+"]]"
                    occ[lw].append(" ".join(seg))
            n+=1
        if n>=200000: break
    return occ

def build_requests(inv, occ):
    reqs=[]; meta={}
    for w,ctxs in occ.items():
        senses=inv[w]
        def g(s):
            if isinstance(s,str): return s
            return (s.get("gloss") or s.get("definition") or s.get("sense") or str(s)) + \
                   (f" [{s.get('cue','')}]" if isinstance(s,dict) and s.get("cue") else "")
        slist="\n".join(f"{j}: {g(s)}" for j,s in enumerate(senses))
        for ci in range(0,len(ctxs),CHUNK):
            chunk=ctxs[ci:ci+CHUNK]
            cid=f"t_{len(reqs)}"
            meta[cid]=(w,ci,len(chunk))
            numbered="\n".join(f"{k}. {c}" for k,c in enumerate(chunk))
            reqs.append({"custom_id":cid,"model":"claude-sonnet-5","max_tokens":len(chunk)*6+80,
                "thinking_off":True,
                "system":(f"Word: '{w}'. Senses:\n{slist}\n\nFor each numbered context (target "
                          "in [[brackets]]), give the sense index. Output ONLY a JSON array of "
                          f"{len(chunk)} integers in order, e.g. [0,2,0,1]. Nothing else."),
                "user":numbered})
    return reqs, meta

if __name__=="__main__":
    cap=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
    inv={**json.load(open("out/scale_inventory.json"))}
    if os.path.exists("out/full_inventory.json"):
        inv.update(json.load(open("out/full_inventory.json")))
    inv={w:s for w,s in inv.items() if len(s)>=2}
    print(f"split words with inventory: {len(inv):,}")
    occ=gather(inv,cap)
    reqs,meta=build_requests(inv,occ)
    print(f"occurrences: {sum(len(v) for v in occ.values()):,}  requests: {len(reqs):,}")
    json.dump(meta,open("out/batches/tag_meta.json","w"))
    json.dump(reqs,open("out/batches/tag_reqs.json","w"))
    print("wrote tag requests (submit separately)")
