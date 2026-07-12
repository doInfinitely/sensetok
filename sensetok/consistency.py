"""Test-retest agreement, not accuracy.

For a TOKENIZER, the annotator does not need to match WordNet -- it needs to assign the
same token to the same meaning every time. A self-consistent labeling defines a valid
sense inventory even if every lexicographer disagrees. (Same argument that settled the
morphology dictionary: consistency over etymology.)

Accuracy against SemCor gold measures agreement with SemCor's arbitrary sense-choice
conventions. The 1-NN model was FIT on those conventions; Haiku never saw them. So
accuracy systematically favours the fitted model and tells us nothing about token purity.

Token purity is bounded by test-retest agreement:
  two independent annotations of the SAME occurrence disagree p of the time
  => a sense token receives ~p impure instances.
"""
import json, collections
import numpy as np
from nltk.corpus import wordnet as wn


def load(prefix):
    d = {}
    for k in range(3):
        for r in json.load(open(f"pilot/{prefix}_{k}.json")): d[r["id"]] = r["sense"]
    return d


def main():
    a, b = load("ans"), load("rerun")
    gold = json.load(open("pilot/gold.json"))
    items = []
    for k in range(3): items += json.load(open(f"pilot/shard_{k}.json"))
    items = {it["id"]: it for it in items}

    ids = sorted(set(a) & set(b))
    print(f"items annotated twice: {len(ids)}")

    agree = collections.Counter(); n = collections.Counter()
    both_right = cons_wrong = 0
    for i in ids:
        nsen = len(items[i]["senses"])
        bucket = "2-3 senses" if nsen <= 3 else ("4-7 senses" if nsen <= 7 else "8+ senses")
        for bk in (bucket, "all"):
            n[bk] += 1; agree[bk] += (a[i] == b[i])
        if a[i] == b[i]:
            if a[i] == gold[i]: both_right += 1
            else: cons_wrong += 1

    print(f"\n{'subset':<13}{'n':>6}{'test-retest':>13}")
    print("-" * 33)
    for bk in ("all", "2-3 senses", "4-7 senses", "8+ senses"):
        if n[bk]: print(f"{bk:<13}{n[bk]:>6}{agree[bk]/n[bk]:>13.3f}")

    A = np.mean([a[i] == gold[i] for i in ids])
    B = np.mean([b[i] == gold[i] for i in ids])
    print(f"\naccuracy vs SemCor gold: run1 {A:.3f}  run2 {B:.3f}")
    print(f"of the {agree['all']} consistent items: {both_right} match gold, "
          f"{cons_wrong} consistently disagree with gold")
    print(f"  -> consistent-but-not-gold is a USABLE sense token; inconsistent is not")
    print(f"\nimplied token purity ceiling: {agree['all']/n['all']:.3f}")
    print(f"  (false morphological decomposition contaminates 1.4% of occurrences)")

    # per-word: which words could we safely split?
    byw = collections.defaultdict(lambda: [0, 0])
    for i in ids:
        w = items[i]["target"]
        byw[w][0] += (a[i] == b[i]); byw[w][1] += 1
    multi = {w: v for w, v in byw.items() if v[1] >= 3}
    if multi:
        r = sorted(multi.items(), key=lambda kv: kv[1][0] / kv[1][1])
        print(f"\nwords seen >=3 times, lowest agreement:")
        for w, (h, t) in r[:6]: print(f"   {w:<14}{h}/{t}")


if __name__ == "__main__":
    main()
