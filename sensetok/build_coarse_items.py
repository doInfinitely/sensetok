"""Re-annotate the SAME 300 pilot items against the LLM-designed coarse inventory.

Same sentences, same targets, different sense lists. Two independent passes, then
kappa vs the random-merge null at matched k. Gold labels are unusable here (different
inventory), which is the point: for a tokenizer, consistency is what matters, not
agreement with a lexicographer.
"""
import json, collections

def main():
    inv = {}
    for i in range(4):
        for r in json.load(open(f"coarse/out_{i}.json")):
            inv[(r["word"], r["pos"])] = r["senses"]
    items = []
    for i in range(3): items += json.load(open(f"pilot/shard_{i}.json"))

    out, missing = [], 0
    for it in items:
        key = (it["target"], it["pos"])
        if key not in inv: missing += 1; continue
        ss = inv[key]
        out.append({"id": it["id"], "sentence": it["sentence"], "target": it["target"],
                    "pos": it["pos"],
                    "senses": [f"{j}: {s['gloss']}  [cue: {s['cue']}]" for j, s in enumerate(ss)]})
    ks = [len(inv[(i['target'], i['pos'])]) for i in items if (i['target'], i['pos']) in inv]
    print(f"items with a coarse inventory: {len(out)}/{len(items)}  (missing {missing})")
    print(f"mean coarse senses/word: {sum(ks)/len(ks):.2f}   "
          f"distribution: {dict(sorted(collections.Counter(ks).items()))}")
    print(f"(WordNet on the same items: 7.5)")
    for i in range(3):
        json.dump(out[i*100:(i+1)*100] if i<2 else out[200:], open(f"coarse/items_{i}.json","w"), indent=1)
        n = len(out[i*100:(i+1)*100]) if i<2 else len(out[200:])
        print(f"  coarse/items_{i}.json: {n} items")
    json.dump({f"{w}|{p}": s for (w,p), s in inv.items()}, open("out/coarse_inventory.json","w"), indent=1)
    print("\nexample:")
    ex = [o for o in out if o["target"] in ("be","work","have")][:1] or out[:1]
    print(json.dumps(ex[0], indent=1)[:600])

if __name__ == "__main__":
    main()
