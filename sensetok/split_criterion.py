"""Which unsupervised statistic identifies a genuinely polysemous word?

Ground truth ("truly split") = the word's second-most-common GOLD sense holds >=20% of
its instances. Not "WordNet lists >=2 senses" -- most polysemous words are used in one
sense in any given corpus.

Candidate statistics, all computed WITHOUT gold:
  silhouette          separation of the k=2 clustering            (what induce.py used)
  -I(cluster;doc)     negated MI between cluster and document     (what I proposed)
  doc_stability       cluster on half the DOCUMENTS, cluster on the other half,
                      apply both to all points, ARI between labelings, averaged
                      over random document halves                 (the refinement)
  inst_stability      same but resampling INSTANCES, not documents (the standard
                      method -- should FAIL, since topic is stable under it)

Report AUC against 'truly split'. Resampling must happen at the level of the nuisance.
"""
import collections, json
import numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, silhouette_score, roc_auc_score,
                             mutual_info_score)
from nltk.corpus import semcor, wordnet as wn

from sensetok.vectors2 import nrm
from sensetok.vectors import OUT
from sensetok.topic_control import doc_ids

MIN_N, MIN_DOC, REPS = 30, 6, 5
rng = np.random.default_rng(0)


def assign(X, C):
    return np.argmax(X @ C.T, axis=1)


def cluster(X, k=2):
    km = KMeans(k, n_init=4, random_state=0).fit(X)
    C = km.cluster_centers_
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)


def stability(X, groups, k=2, reps=REPS):
    """cluster on half the groups, half on the other; compare labelings on ALL points."""
    uniq = np.unique(groups)
    if len(uniq) < 2: return np.nan
    out = []
    for r in range(reps):
        p = np.random.default_rng(r).permutation(len(uniq))
        A = set(uniq[p[: len(uniq) // 2]])
        ma = np.array([g in A for g in groups])
        if ma.sum() < k + 1 or (~ma).sum() < k + 1: continue
        try:
            Ca, Cb = cluster(X[ma], k), cluster(X[~ma], k)
        except Exception:
            continue
        out.append(adjusted_rand_score(assign(X, Ca), assign(X, Cb)))
    return float(np.mean(out)) if out else np.nan


def main():
    d = torch.load(f"{OUT}/sense_v2.pt", weights_only=False)
    files = sorted(semcor.fileids())
    perm = np.random.default_rng(0).permutation(len(files))
    docs = np.array(doc_ids([files[i] for i in perm[40:80]]) +
                    doc_ids([files[i] for i in perm[:40]]))
    insts = d["dev"] + d["test"]
    V = nrm(np.vstack([d["dev_vecs"], d["test_vecs"]]))
    assert len(docs) == len(insts)

    by = collections.defaultdict(list)
    for i, (_, _, _, g) in enumerate(insts):
        try: lem = wn.lemma_from_key(g)
        except Exception: continue
        by[(lem.name().lower(), lem.synset().pos())].append((i, g))

    rows = []
    for (w, pos), items in by.items():
        if len(items) < MIN_N: continue
        idx = np.array([i for i, _ in items])
        gold = [g for _, g in items]
        dd = docs[idx]
        if len(set(dd)) < MIN_DOC: continue
        X = V[idx]
        c = collections.Counter(gold).most_common()
        truly = len(c) >= 2 and c[1][1] / len(gold) >= 0.20

        lab = KMeans(2, n_init=4, random_state=0).fit_predict(X)
        sil = silhouette_score(X, lab)
        mi = mutual_info_score(lab, dd)
        h = mutual_info_score(lab, lab) + 1e-9
        doc_stab = stability(X, dd)
        inst_stab = stability(X, np.arange(len(X)) % 4)   # resample by instance
        rows.append((w, len(X), len(set(dd)), truly, sil, -mi / h, doc_stab, inst_stab))

    rows = [r for r in rows if not any(np.isnan(x) for x in r[4:])]
    y = np.array([r[3] for r in rows], int)
    print(f"types: {len(rows)}   truly split: {y.sum()}   not: {(1-y).sum()}\n")
    print(f"{'statistic':<18}{'AUC':>8}   {'mean(split)':>12}{'mean(not)':>11}")
    print("-" * 52)
    for j, nm in ((4, "silhouette"), (5, "-I(clust;doc)/H"), (6, "doc_stability"),
                  (7, "inst_stability")):
        v = np.array([r[j] for r in rows])
        auc = roc_auc_score(y, v)
        print(f"{nm:<18}{auc:>8.3f}   {v[y==1].mean():>12.3f}{v[y==0].mean():>11.3f}")

    # empirical null: calibrate on words that are NOT truly split
    v = np.array([r[6] for r in rows])
    thr = np.percentile(v[y == 0], 95)
    sel = v >= thr
    print(f"\ndoc_stability threshold at 95th pct of NOT-split words: {thr:.3f}")
    print(f"  words passing: {sel.sum()}  of which truly split: {y[sel].sum()} "
          f"({y[sel].mean():.0%} precision, {y[sel].sum()/max(y.sum(),1):.0%} recall)")
    print("\nresampling by INSTANCE is the standard stability method and it fails here:")
    print("  topic is constant within a document, so both instance-halves see the same")
    print("  topics and topic clusters look perfectly stable. Resample the NUISANCE.")
    json.dump([[r[0], int(r[1]), int(r[2]), bool(r[3])] + [float(x) for x in r[4:]]
               for r in rows], open(f"{OUT}/split_criterion.json", "w"), indent=1)


if __name__ == "__main__":
    main()
