"""Unsupervised sense induction, validated against SemCor gold senses.

The claim to test: clustering a word type's contextual vectors recovers its senses.
The null that must accompany it: a RANDOM partition into the same number of clusters
with the same cluster sizes. A k-way split of a frequent token can look structured for
free, and adjusted-Rand against gold is not obviously above chance until you check.

Second null: TOPIC. Contextual vectors of a word carry the document's topic as well as
the word's sense. `bank` in a finance article and `bank` in a geology article may cluster
by article, not by meaning. Proxy: cluster the vectors of MONOSEMOUS words, which have
exactly one sense, and see how much apparent structure appears anyway.

ARI is chance-corrected, so ARI(random) ~ 0 by construction; the informative comparison
is induced-vs-gold against monosemous-vs-<its own gold, a single class> and against the
size-matched permutation.
"""
import collections, json
import numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from nltk.corpus import wordnet as wn

from sensetok.vectors2 import nrm
from sensetok.vectors import OUT

MIN_N, KMAX = 30, 6
MIN_N_MONO = 15
rng = np.random.default_rng(0)


def choose_k(X, kmax=KMAX):
    """silhouette over k=2..kmax; k=1 (do not split) if no k beats a floor."""
    best_k, best_s = 1, -1.0
    for k in range(2, min(kmax, len(X) - 1) + 1):
        lab = KMeans(k, n_init=4, random_state=0).fit_predict(X)
        if len(set(lab)) < 2: continue
        s = silhouette_score(X, lab)
        if s > best_s: best_k, best_s = k, s
    return (best_k, best_s) if best_s >= 0.05 else (1, best_s)


def size_matched_null(labels, rng):
    p = rng.permutation(len(labels))
    return np.asarray(labels)[p]


def main():
    d = torch.load(f"{OUT}/sense_v2.pt", weights_only=False)
    insts = d["dev"] + d["test"]
    V = nrm(np.vstack([d["dev_vecs"], d["test_vecs"]]))

    by = collections.defaultdict(list)
    for i, (_, _, _, g) in enumerate(insts):
        try: lem = wn.lemma_from_key(g)
        except Exception: continue
        by[(lem.name().lower(), lem.synset().pos())].append((i, g))

    poly, mono = [], []
    for (w, pos), items in by.items():
        ngold = len({g for _, g in items})
        ncand = len(wn.synsets(w, pos))
        if ncand >= 2 and len(items) >= MIN_N:
            poly.append((w, pos, items, ngold))
        elif ncand == 1 and len(items) >= MIN_N_MONO:
            mono.append((w, pos, items, ngold))
    print(f"types with >={MIN_N} instances:  polysemous {len(poly)}   monosemous {len(mono)}")

    rows = []
    for w, pos, items, ngold in poly:
        idx = [i for i, _ in items]
        gold = [g for _, g in items]
        X = V[idx]
        k, sil = choose_k(X)
        lab = np.zeros(len(X), int) if k == 1 else KMeans(k, n_init=4, random_state=0).fit_predict(X)
        ari = adjusted_rand_score(gold, lab)
        ari_null = np.mean([adjusted_rand_score(gold, size_matched_null(lab, rng)) for _ in range(20)])
        rows.append((w, len(X), ngold, k, sil, ari, ari_null))

    mrows = []
    for w, pos, items, _ in mono:
        X = V[[i for i, _ in items]]
        k, sil = choose_k(X)
        mrows.append((w, len(X), k, sil))

    A = np.array([r[5] for r in rows]); N = np.array([r[6] for r in rows])
    K = np.array([r[3] for r in rows]); G = np.array([r[2] for r in rows])
    print(f"\nINDUCED vs GOLD (adjusted Rand, chance-corrected):")
    print(f"   induced          mean {A.mean():+.4f}   median {np.median(A):+.4f}")
    print(f"   size-matched null mean {N.mean():+.4f}")
    print(f"   types where induced > null: {(A>N).sum()}/{len(A)}")
    print(f"   spearman(k_chosen, n_gold_senses_present) = "
          f"{np.corrcoef(K.argsort().argsort(), G.argsort().argsort())[0,1]:+.3f}")
    print(f"   mean k chosen {K.mean():.2f}   mean gold senses present {G.mean():.2f}")

    sm = np.array([r[2] for r in mrows])          # r = (w, n, k, sil)
    ss = np.array([r[3] for r in mrows])
    sp = np.array([r[4] for r in rows])
    print(f"\nTOPIC CONTROL — monosemous types (exactly one WordNet sense):")
    print(f"   n types {len(mrows)}")
    print(f"   fraction split (k>1): {(sm>1).mean():.1%}   mean k {sm.mean():.2f}   "
          f"mean silhouette {ss.mean():.3f}")
    print(f"   polysemous:           {(K>1).mean():.1%}   mean k {K.mean():.2f}   "
          f"mean silhouette {sp.mean():.3f}")
    print("   if monosemous words split just as eagerly, the clusters are TOPIC, not sense")

    print(f"\n{'word':<14}{'n':>5}{'gold':>6}{'k':>4}{'sil':>7}{'ARI':>8}{'null':>8}")
    for r in sorted(rows, key=lambda r: -r[5])[:10]:
        print(f"  {r[0]:<12}{r[1]:>5}{r[2]:>6}{r[3]:>4}{r[4]:>7.3f}{r[5]:>8.3f}{r[6]:>8.3f}")
    print("  ...")
    for r in sorted(rows, key=lambda r: r[5])[:5]:
        print(f"  {r[0]:<12}{r[1]:>5}{r[2]:>6}{r[3]:>4}{r[4]:>7.3f}{r[5]:>8.3f}{r[6]:>8.3f}")
    json.dump([[r[0], int(r[1]), int(r[2]), int(r[3]), float(r[4]), float(r[5]), float(r[6])]
               for r in rows], open(f"{OUT}/induce.json", "w"), indent=1)


if __name__ == "__main__":
    main()
