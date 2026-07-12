"""Measurement-sized bootstrap: enough to MEASURE the propagator, not label the corpus.

If a cheap propagator trained on Sonnet labels recovers held-out Sonnet labels on 150
words, it works on all 1542 -> then the propagator tags the whole corpus for free (no more
Sonnet). So we only Sonnet-label a sample: 150 split words x 40 occurrences, 300/shard.
"""
import json, collections, re
from datasets import load_dataset
WORD = re.compile(r"[A-Za-z']+")
N_PER, N_WORDS, PER_SHARD = 40, 150, 300

def main():
    inv = json.load(open("out/scale_inventory.json"))
    split = {w: s for w, s in inv.items() if len(s) >= 2}
    # stratify: take every k-th split word by the assembled order (freq-descending-ish)
    words = list(split)
    sample = words[::max(1, len(words)//N_WORDS)][:N_WORDS]
    ds = load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train")
    occ = collections.defaultdict(list); n = 0
    want = set(sample)
    for r in ds:
        t = r["text"]
        if len(t) > 200 and not t.lstrip().startswith("="):
            toks = WORD.findall(t)
            for i, w in enumerate(toks):
                lw = w.lower()
                if lw in want and len(occ[lw]) < N_PER:
                    s = max(0, i-14); e = min(len(toks), i+14)
                    seg = toks[s:e]; seg[i-s] = "[[%s]]" % seg[i-s]
                    occ[lw].append(" ".join(seg))
            n += 1
        if n >= 150000: break
    items, iid = [], 0
    for w in sample:
        senses = [f"{j}: {s['gloss']}  [{s['cue']}]" for j,s in enumerate(split[w])]
        for c in occ[w]:
            items.append({"id": iid, "word": w, "sentence": c, "senses": senses}); iid += 1
    nsh = (len(items) + PER_SHARD - 1)//PER_SHARD
    for i in range(nsh):
        json.dump(items[i*PER_SHARD:(i+1)*PER_SHARD], open(f"scale/boot2_{i:02d}.json","w"))
    json.dump({w: split[w] for w in sample}, open("out/boot2_inventory.json","w"))
    print(f"sample: {len(sample)} words, {len(items):,} occurrences, {nsh} shards of ~{PER_SHARD}")
    print(f"mean occ/word: {len(items)/len(sample):.1f}")
    print(f"cost: ~{len(items)*50/1e6:.1f}M in + {len(items)*2/1e6:.2f}M out = "
          f"~${len(items)*50/1e6*3 + len(items)*2/1e6*15:.1f} Sonnet")

if __name__ == "__main__":
    main()
