"""Score a sense inventory by CHANCE-CORRECTED test-retest agreement.

THE TRAP: raw agreement inflates as the inventory shrinks. With WordNet's 8.6 senses per
polysemous token, chance agreement is ~12%; with 4 senses it is 25%; with 2 it is 50%. A
coarse inventory looks more consistent even if it is arbitrary. So:

  metric = chance-corrected agreement,  kappa = (p_o - p_e) / (1 - p_e)
           with a PER-ITEM chance model: an annotator picking uniformly among that
           item's k candidate senses agrees with itself with probability 1/k, so
           p_e = mean_i (1 / k_i).

           Cohen's kappa proper is NOT usable here: 300 items spread over 264 words means
           ~1 annotation per word, so its marginal-frequency chance model is estimated
           from a single observation and collapses.
  null   = RANDOM MERGE of the SAME WordNet senses into the SAME number of groups

The null is free: we already have two independent Haiku passes over the same 300 items in
the WordNet inventory (pilot/ans_*.json, pilot/rerun_*.json). Map those labels through a
random partition into k groups and you get exactly "what agreement looks like when you
coarsen without thinking."

An inventory earns its keep only if kappa(LLM-coarse) > kappa(random-merge at same k).
"""
import json, collections
import numpy as np


def cohens_kappa(a, b, n_cat):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = 0.0
    for c in range(n_cat):
        pe += (a == c).mean() * (b == c).mean()
    return (po - pe) / (1 - pe + 1e-12), po


def random_merge_null(a, b, k, n_sense, rng):
    """partition n_sense WordNet senses into k groups at random; relabel both passes."""
    if n_sense <= k: return None
    g = rng.integers(0, k, n_sense)
    while len(set(g)) < k: g = rng.integers(0, k, n_sense)
    return [int(g[x]) for x in a], [int(g[x]) for x in b]


def main():
    from nltk.corpus import wordnet as wn
    items = []
    for i in range(3): items += json.load(open(f"pilot/shard_{i}.json"))
    items = {it["id"]: it for it in items}
    A, B = {}, {}
    for i in range(3):
        for r in json.load(open(f"pilot/ans_{i}.json")): A[r["id"]] = r["sense"]
        for r in json.load(open(f"pilot/rerun_{i}.json")): B[r["id"]] = r["sense"]
    ids = sorted(set(A) & set(B))

    # ---- WordNet inventory: kappa, per item pooled by treating senses as categories ----
    # pool across words by computing kappa per word then weighting by n items
    byw = collections.defaultdict(list)
    for i in ids: byw[(items[i]["target"], items[i]["pos"])].append(i)

    rng = np.random.default_rng(0)
    rows = []
    for (w, pos), its in byw.items():
        n_sense = len(items[its[0]]["senses"])
        a = [A[i] for i in its]; b = [B[i] for i in its]
        rows.append((w, pos, len(its), n_sense, a, b))

    def pooled(rows, relabel=None, k=None):
        """po = observed self-agreement; pe = mean_i 1/k_i (uniform choice among candidates)."""
        agree, chance = [], []
        for w, pos, n, ns, a, b in rows:
            cats = ns
            if relabel is not None:
                out = relabel(a, b, ns)
                if out is None:                    # fewer senses than k: nothing to merge
                    aa, bb, cats = a, b, ns
                else:
                    aa, bb, cats = out[0], out[1], k
            else:
                aa, bb = a, b
            for x, y in zip(aa, bb):
                agree.append(x == y); chance.append(1.0 / cats)
        po = float(np.mean(agree)); pe = float(np.mean(chance))
        return (po - pe) / (1 - pe + 1e-12), po, len(agree)

    kw, pow_, n = pooled(rows)
    ns_mean = np.mean([r[3] for r in rows])
    pe_w = float(np.mean([1.0/r[3] for r in rows for _ in r[4]]))
    print(f"WordNet inventory ({ns_mean:.1f} senses/word, {n} items in {len(rows)} words)")
    print(f"   raw agreement {pow_:.3f}   chance {pe_w:.3f}   kappa {kw:.3f}\n")

    print(f"{'k':<5}{'random-merge raw':>18}{'chance':>9}{'random-merge kappa':>21}{'sd':>8}")
    print("-" * 62)
    for k in (2, 3, 4, 5):
        ks, ps = [], []
        for rep in range(30):
            r = np.random.default_rng(rep)
            kk, pp, nn = pooled(rows, relabel=lambda a,b,ns: random_merge_null(a,b,k,ns,r), k=k)
            ks.append(kk); ps.append(pp)
        print(f"{k:<5}{np.mean(ps):>18.3f}{1.0/k:>9.3f}{np.mean(ks):>21.3f}{np.std(ks):>8.3f}")
    print("\nRaw agreement rises with coarseness for free. Even KAPPA can rise, because")
    print("merging ANY two confusable senses removes a confusion. So the bar for an")
    print("LLM-designed inventory is: beat the RANDOM MERGE at its own k.")
    json.dump({"wordnet_kappa": kw, "wordnet_raw": pow_}, open("out/kappa_baseline.json","w"))


if __name__ == "__main__":
    main()
