"""For each split word, sample N occurrences from the corpus -> Sonnet labels them ->
those labels train the cheap contextual disambiguator (1-NN + prior) that tags the
whole corpus. Spends Sonnet only on a bootstrap set, not every occurrence.
"""
import json, collections, re, sys
from datasets import load_dataset
WORD = re.compile(r"[A-Za-z']+")
N_PER = 30

def main():
    inv = json.load(open("out/scale_inventory.json"))
    split = {w: s for w, s in inv.items() if len(s) >= 2}
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    occ = collections.defaultdict(list); n = 0
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="):
            toks = WORD.findall(t)
            for i, w in enumerate(toks):
                lw = w.lower()
                if lw in split and len(occ[lw]) < N_PER:
                    s = max(0, i-12); e = min(len(toks), i+12)
                    seg = toks[s:e]; seg[i-s] = "[[%s]]" % seg[i-s]
                    occ[lw].append(" ".join(seg))
            n += 1
        if n >= 120000: break
    items, iid = [], 0
    for w, ctxs in occ.items():
        senses = [f"{j}: {s['gloss']}  [{s['cue']}]" for j,s in enumerate(split[w])]
        for c in ctxs:
            items.append({"id": iid, "word": w, "sentence": c, "senses": senses}); iid += 1
    S = 40
    for i in range(S):
        json.dump(items[i::S], open(f"scale/boot_{i:02d}.json","w"))
    print(f"bootstrap items: {len(items):,} over {len(occ):,} split words ({S} shards)")
    print(f"cost estimate: ~{len(items)*45/1e6:.1f}M in + {len(items)*2/1e6:.2f}M out "
          f"= ~${len(items)*45/1e6*3 + len(items)*2/1e6*15:.0f} Sonnet")

if __name__ == "__main__":
    main()
