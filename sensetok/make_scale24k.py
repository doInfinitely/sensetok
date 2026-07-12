"""Full inventory-design payload for the top-24k polysemous types + a hard-word-weighted
tagging sample plan. One wikitext pass. Shards at ~100 words for inventory (fewer agents).
"""
import json, collections, re
from nltk.corpus import wordnet as wn
from sensetok.inventory import gloss_text
WORD = re.compile(r"[A-Za-z']+")

def main():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    cnt = collections.Counter(); n = 0
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="):
            cnt.update(w.lower() for w in WORD.findall(t) if w.isalpha()); n += 1
        if n >= 200000: break
    poly = [(w,c) for w,c in cnt.items() if c >= 10 and len(wn.synsets(w)) >= 2]
    poly.sort(key=lambda x:-x[1])
    # already have top-2k designed; do the NEXT ones (2k..24k) to avoid redoing
    have = set(json.load(open("out/scale_inventory.json")))
    todo = [(w,c) for w,c in poly[:24000] if w not in have]
    print(f"polysemous top-24k: {min(len(poly),24000):,}; already designed {len(have)&set(w for w,_ in poly[:24000]).__len__() if False else len(have)}; TODO {len(todo):,}")
    # contexts
    want = set(w for w,_ in todo); ctx = collections.defaultdict(list); n = 0
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="):
            toks = WORD.findall(t)
            for i,w in enumerate(toks):
                lw=w.lower()
                if lw in want and len(ctx[lw])<5:
                    s=max(0,i-12);e=min(len(toks),i+12);seg=toks[s:e];seg[i-s]="[[%s]]"%seg[i-s]
                    ctx[lw].append(" ".join(seg))
            n+=1
        if n>=200000: break
    payload=[{"word":w,"freq":c,"wordnet":[gloss_text(s)[:110] for s in wn.synsets(w)][:20],
              "contexts":ctx[w][:5]} for w,c in todo]
    PER=100; nsh=(len(payload)+PER-1)//PER
    for i in range(nsh):
        json.dump(payload[i*PER:(i+1)*PER], open(f"scale/inv24_{i:03d}.json","w"))
    print(f"inventory-design shards: {nsh} of ~{PER} words  (est ${len(payload)*1700/1e6*6:.0f} Sonnet)")

if __name__=="__main__":
    main()
