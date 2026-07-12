"""Merge the Sonnet inventory shards, validate, and report coverage + the sense-token budget."""
import json, collections, glob, os

def main():
    inv = {}
    bad = []
    for f in sorted(glob.glob("scale/out_*.json")):
        try:
            for r in json.load(open(f)):
                inv[r["word"]] = r["senses"]
        except Exception as e:
            bad.append((f, str(e)[:60]))
    if bad:
        print("UNPARSEABLE SHARDS (need repair):")
        for f, e in bad: print(f"  {f}: {e}")
    ks = [len(v) for v in inv.values()]
    poly = {w: v for w, v in inv.items() if len(v) >= 2}
    print(f"\nwords with an inventory: {len(inv):,}")
    print(f"  mean senses/word {sum(ks)/len(ks):.2f}   "
          f"dist {dict(sorted(collections.Counter(ks).items()))}")
    print(f"  actually split (>=2 senses): {len(poly):,}   "
          f"kept monosemous: {len(inv)-len(poly):,}")
    # new sense-token count = sum of senses over split words + 1 per monosemous
    extra = sum(len(v)-1 for v in poly.values())
    print(f"  extra sense tokens minted: {extra:,} (each split word adds k-1 symbols)")
    json.dump(inv, open("out/scale_inventory.json","w"), indent=1)
    print(f"  wrote out/scale_inventory.json")

if __name__ == "__main__":
    main()
