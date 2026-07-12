"""Do induced clusters track SENSE or DOCUMENT?

The monosemous control showed clusters form regardless of polysemy, with higher
silhouette than polysemous words. Silhouette measures separation, not meaning.

Decisive test: for each word, compare
    ARI(clusters, gold sense)   vs   ARI(clusters, document id)
A word whose clusters predict the document better than the sense is being split by
topic. Tokens minted from such clusters are document-dependent -- the exact vice sense
tokens exist to remove.

This is the control that invalidates 'purity w.r.t. its own inventory'. A random split
is also 100% pure w.r.t. itself.
"""
import collections, json
import numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from nltk.corpus import semcor, wordnet as wn

from sensetok.vectors2 import nrm
from sensetok.vectors import OUT, load_instances

MIN_N, KMAX = 30, 6


def doc_ids(fileids):
    """rebuild the per-instance file id; load_instances is deterministic."""
    out = []
    for fid in fileids:
        n0 = len(load_instances([fid]))
        out += [fid] * n0
    return out


def choose_k(X):
    best_k, best_s = 1, -1.0
    for k in range(2, min(KMAX, len(X) - 1) + 1):
        lab = KMeans(k, n_init=4, random_state=0).fit_predict(X)
        if len(set(lab)) < 2: continue
        s = silhouette_score(X, lab)
        if s > best_s: best_k, best_s = k, s
    return best_k if best_s >= 0.05 else 1


def main():
    d = torch.load(f"{OUT}/sense_v2.pt", weights_only=False)
    split = json.load(open(f"{OUT}/split.json")) if False else None
    files = sorted(semcor.fileids())
    perm = np.random.default_rng(0).permutation(len(files))
    test_f = [files[i] for i in perm[:40]]
    dev_f = [files[i] for i in perm[40:80]]

    docs = doc_ids(dev_f) + doc_ids(test_f)
    insts = d["dev"] + d["test"]
    assert len(docs) == len(insts), f"{len(docs)} != {len(insts)}"
    V = nrm(np.vstack([d["dev_vecs"], d["test_vecs"]]))

    by = collections.defaultdict(list)
    for i, (_, _, _, g) in enumerate(insts):
        try: lem = wn.lemma_from_key(g)
        except Exception: continue
        by[(lem.name().lower(), lem.synset().pos())].append((i, g))

    rows = []
    for (w, pos), items in by.items():
        if len(items) < MIN_N: continue
        ncand = len(wn.synsets(w, pos))
        idx = [i for i, _ in items]
        gold = [g for _, g in items]
        doc = [docs[i] for i in idx]
        if len(set(doc)) < 2: continue
        X = V[idx]
        k = choose_k(X)
        if k == 1: continue
        lab = KMeans(k, n_init=4, random_state=0).fit_predict(X)
        rows.append((w, ncand, len(X), k,
                     adjusted_rand_score(gold, lab),
                     adjusted_rand_score(doc, lab),
                     adjusted_rand_score(gold, doc)))

    poly = [r for r in rows if r[1] >= 2]
    mono = [r for r in rows if r[1] == 1]
    for tag, g in (("polysemous", poly), ("monosemous", mono)):
        if not g: continue
        S = np.array([r[4] for r in g]); D = np.array([r[5] for r in g])
        print(f"{tag:<12} n={len(g):<4}  ARI(clusters,sense) {S.mean():+.3f}   "
              f"ARI(clusters,doc) {D.mean():+.3f}   doc>sense in {(D>S).sum()}/{len(g)}")
    GD = np.array([r[6] for r in poly])
    print(f"\nconfound check: ARI(gold sense, document) = {GD.mean():+.3f}")
    print("  (senses are not uniformly distributed over documents, so some of the")
    print("   sense signal IS topic; this is the ceiling on how cleanly they separate)")

    print(f"\n{'word':<13}{'cand':>5}{'n':>5}{'k':>3}{'ARI sense':>11}{'ARI doc':>9}")
    for r in sorted(poly, key=lambda r: -(r[5] - r[4]))[:8]:
        print(f"  {r[0]:<11}{r[1]:>5}{r[2]:>5}{r[3]:>3}{r[4]:>11.3f}{r[5]:>9.3f}   <- topic")
    for r in sorted(poly, key=lambda r: -(r[4] - r[5]))[:5]:
        print(f"  {r[0]:<11}{r[1]:>5}{r[2]:>5}{r[3]:>3}{r[4]:>11.3f}{r[5]:>9.3f}   <- sense")
    json.dump([[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows],
              open(f"{OUT}/topic_control.json", "w"), indent=1)


if __name__ == "__main__":
    main()
