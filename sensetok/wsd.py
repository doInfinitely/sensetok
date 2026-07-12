"""1-NN disambiguation against sense centroids, vs the most-frequent-sense baseline.

MFS is WordNet's sense 1, whose ordering is itself derived from SemCor frequency.
It is a notoriously strong baseline (~65-70% F1) and every WSD number is meaningless
without it printed alongside.

Accuracy is reported on POLYSEMOUS instances separately. Monosemous words are free
and there are a lot of them; including them inflates the headline by ~10 points.
"""
import collections, json
import numpy as np, torch
from nltk.corpus import wordnet as wn

from sensetok.vectors import embed_glosses, MODEL, DEV, OUT
from sensetok.inventory import sense_key

def main():
    from transformers import AutoTokenizer, AutoModel
    C = np.load(f"{OUT}/sense_centroids.npy")
    keys = json.load(open(f"{OUT}/sense_keys.json"))
    kidx = {k: i for i, k in enumerate(keys)}
    d = torch.load(f"{OUT}/test.pt", weights_only=False)
    te, TV = d["test"], d["test_vecs"]
    TV = TV / (np.linalg.norm(TV, axis=1, keepdims=True) + 1e-9)

    # candidate sets, from the gold key's lemma+pos (standard all-words WSD setup)
    cands, gold, skip = [], [], 0
    for (_, _, _, k) in te:
        try:
            lem = wn.lemma_from_key(k); syn = lem.synset()
        except Exception:
            cands.append(None); gold.append(None); skip += 1; continue
        cs = wn.synsets(lem.name(), syn.pos())
        cands.append([sense_key(lem.name(), s) for s in cs])
        gold.append(k)

    need = sorted({k for cs in cands if cs for k in cs} - set(kidx))
    print(f"candidate keys needing gloss backoff: {len(need):,}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL).to(DEV).eval()
    G = embed_glosses(need, tok, mod)
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
    gidx = {k: i for i, k in enumerate(need)}

    def vec(k):
        return C[kidx[k]] if k in kidx else G[gidx[k]]

    n = collections.Counter(); hit = collections.Counter()
    cov_hit = cov_tot = 0
    for i, (cs, g) in enumerate(zip(cands, gold)):
        if not cs or g is None: continue
        poly = len(cs) >= 2
        bucket = "poly" if poly else "mono"
        n[bucket] += 1; n["all"] += 1
        # MFS = first synset for the lemma
        mfs = cs[0]
        if mfs == g: hit[f"mfs_{bucket}"] += 1; hit["mfs_all"] += 1
        # 1-NN
        sims = [float(TV[i] @ vec(k)) for k in cs]
        pred = cs[int(np.argmax(sims))]
        if pred == g: hit[f"knn_{bucket}"] += 1; hit["knn_all"] += 1
        cov_tot += len(cs); cov_hit += sum(k in kidx for k in cs)

    print(f"\nskipped (unparseable gold key): {skip:,}")
    print(f"candidate senses with a SemCor centroid: {cov_hit/max(cov_tot,1):.1%}\n")
    print(f"{'subset':<10}{'n':>9}{'MFS':>10}{'1-NN':>10}{'delta':>10}")
    print("-" * 49)
    for b in ("all", "poly", "mono"):
        if not n[b]: continue
        m, k = hit[f"mfs_{b}"] / n[b], hit[f"knn_{b}"] / n[b]
        print(f"{b:<10}{n[b]:>9,}{m:>10.3f}{k:>10.3f}{k-m:>+10.3f}")
    print("\npoly = >=2 candidate senses; the only subset where WSD does anything")

if __name__ == "__main__":
    main()
