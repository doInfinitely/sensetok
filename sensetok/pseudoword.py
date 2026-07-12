"""Pseudoword evaluation (Schütze 1992): exact sense labels, no annotation, any scale.

SemCor could not answer the gating question -- every statistic sat at chance (AUC 0.52)
on 30-150 instances per word. Two problems: no power, and 'truly split' was defined by
gold senses that are themselves topic-correlated (ARI(sense,doc)=+0.117).

Pseudowords fix both:
  POSITIVE  merge two monosemous words A,B into one token. TWO senses by construction,
            labels known exactly, as many occurrences as the corpus provides.
  NEGATIVE  a single monosemous word. ONE sense by construction -- but its occurrences
            still span many articles with different topics. This is the hard negative
            the whole criterion has to survive.

If a statistic cannot separate POSITIVE from NEGATIVE here, it cannot gate sense
induction anywhere. Documents = wikitext articles.
"""
import collections, json, re
import numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, silhouette_score, roc_auc_score,
                             mutual_info_score)
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel

from sensetok.vectors import MODEL, DEV, OUT
from sensetok.vectors2 import L4, nrm

WORD = re.compile(r"[A-Za-z']+")
N_ART, MAX_OCC, MIN_OCC, MIN_ART, WIN = 30000, 200, 60, 8, 24
N_PAIRS = 120


def articles(n):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    arts, cur = [], []
    head = re.compile(r"^\s*=\s[^=].*=\s*$")
    for r in ds:
        t = r["text"]
        if head.match(t):
            if cur: arts.append(cur)
            cur = []
            if len(arts) >= n: break
        elif len(t) > 200:
            cur.append(t)
    return [a for a in arts if a]


@torch.no_grad()
def embed(spans, tok, mod, bs=64):
    V = np.zeros((len(spans), mod.config.dim), dtype=np.float32)
    for i in range(0, len(spans), bs):
        b = spans[i:i + bs]
        enc = tok([s[0] for s in b], is_split_into_words=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=64)
        out = mod(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True)
        h = sum(out.hidden_states[l] for l in L4)
        for j, (_, pos) in enumerate(b):
            wid = enc.word_ids(j)
            sel = [p for p, w in enumerate(wid) if w == pos] or [0]
            V[i + j] = h[j, sel].mean(0).float().cpu().numpy()
    return V


def assign(X, C): return np.argmax(X @ C.T, axis=1)
def km(X, k=2):
    C = KMeans(k, n_init=4, random_state=0).fit(X).cluster_centers_
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)


def stability(X, groups, reps=5):
    u = np.unique(groups)
    if len(u) < 2: return np.nan
    o = []
    for r in range(reps):
        p = np.random.default_rng(r).permutation(len(u))
        A = set(u[p[:len(u)//2]]); m = np.array([g in A for g in groups])
        if m.sum() < 3 or (~m).sum() < 3: continue
        try: o.append(adjusted_rand_score(assign(X, km(X[m])), assign(X, km(X[~m]))))
        except Exception: pass
    return float(np.mean(o)) if o else np.nan


def stats(X, doc):
    lab = KMeans(2, n_init=4, random_state=0).fit_predict(X)
    sil = silhouette_score(X, lab)
    mi = mutual_info_score(lab, doc) / (mutual_info_score(lab, lab) + 1e-9)
    return sil, -mi, stability(X, doc), stability(X, np.arange(len(X)) % 4)


def main():
    arts = articles(N_ART)
    print(f"articles: {len(arts):,}", flush=True)
    occ = collections.defaultdict(list)
    for ai, a in enumerate(arts):
        for para in a:
            toks = WORD.findall(para)
            for i, w in enumerate(toks):
                lw = w.lower()
                if len(lw) > 3 and lw.isalpha():
                    occ[lw].append((ai, toks, i))
    mono = [w for w, v in occ.items()
            if len(v) >= MIN_OCC and len({a for a, _, _ in v}) >= MIN_ART
            and len(wn.synsets(w)) == 1]
    print(f"monosemous candidates (1 wn synset, >={MIN_OCC} occ, >={MIN_ART} articles): {len(mono):,}")
    rng = np.random.default_rng(0)
    mono = [mono[i] for i in rng.permutation(len(mono))][:2 * N_PAIRS]

    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL).to(DEV).eval()

    data = {}
    for w in mono:
        v = occ[w]
        if len(v) > MAX_OCC: v = [v[i] for i in rng.permutation(len(v))[:MAX_OCC]]
        spans, docs = [], []
        for ai, toks, i in v:
            s = max(0, i - WIN); e = min(len(toks), i + WIN)
            spans.append((toks[s:e], i - s)); docs.append(ai)
        data[w] = (nrm(embed(spans, tok, mod)), np.array(docs))
        if len(data) % 40 == 0: print(f"  embedded {len(data)}/{len(mono)}", flush=True)

    rows = []
    for w in mono:                                   # NEGATIVE: one sense, many topics
        X, dc = data[w]
        s = stats(X, dc)
        if not any(np.isnan(x) for x in s): rows.append((w, 0, len(X), *s, np.nan))
    # HARD pairing: pair each word with its NEAREST NEIGHBOUR by mean vector, so the
    # two "senses" are semantically close and topically overlapping. Random pairs
    # (banana|door) are two unrelated, topic-disjoint words -- trivially separable, and
    # they manufacture a positive that is ALSO a topic split. That is what inflated the
    # first run's AUCs to 0.998 and inverted -I(clust;doc) to 0.087.
    M = nrm(np.stack([data[w][0].mean(0) for w in mono]))
    sim = M @ M.T; np.fill_diagonal(sim, -9)
    used, pairs = set(), []
    for i in np.argsort(-sim.max(1)):
        if mono[i] in used: continue
        j = int(np.argmax(np.where([m in used for m in mono], -9, sim[i])))
        if mono[j] in used or i == j: continue
        used.add(mono[i]); used.add(mono[j]); pairs.append((mono[i], mono[j]))
    print(f"hard pairs: {len(pairs)}  e.g. " +
          ", ".join(f"{a}|{b}" for a, b in pairs[:5]))
    for a, b in pairs:
        Xa, da = data[a]; Xb, db = data[b]
        X = np.vstack([Xa, Xb]); dc = np.concatenate([da, db])
        y = np.array([0] * len(Xa) + [1] * len(Xb))
        s = stats(X, dc)
        lab = KMeans(2, n_init=4, random_state=0).fit_predict(X)
        if not any(np.isnan(x) for x in s):
            rows.append((f"{a}|{b}", 1, len(X), *s, adjusted_rand_score(y, lab)))

    y = np.array([r[1] for r in rows])
    print(f"\npseudowords: {(y==1).sum()} positive (2 senses), {(y==0).sum()} negative (1 sense)")
    print(f"mean occurrences: {np.mean([r[2] for r in rows]):.0f}\n")
    print(f"{'statistic':<18}{'AUC':>8}{'mean(2-sense)':>15}{'mean(1-sense)':>15}")
    print("-" * 56)
    for j, nm in ((3, "silhouette"), (4, "-I(clust;doc)/H"), (5, "doc_stability"),
                  (6, "inst_stability")):
        v = np.array([r[j] for r in rows])
        print(f"{nm:<18}{roc_auc_score(y, v):>8.3f}{v[y==1].mean():>15.3f}{v[y==0].mean():>15.3f}")

    ari = np.array([r[7] for r in rows if r[1] == 1])
    print(f"\nARI(clusters, true pseudo-sense) on positives: {np.nanmean(ari):.3f}")
    print("  -> does induction RECOVER a known 2-sense mixture at realistic n?")
    print(f"\nSemCor reference (n=30-150, real senses): every AUC was 0.39-0.53")
    print(f"RANDOM-pair pseudowords (banana|door): silhouette AUC 0.998, ARI 0.993")
    print("If HARD pairs collapse toward the SemCor numbers, random pseudowords were")
    print("measuring 'are these two different words', not 'are these two senses'.")
    json.dump([[r[0], int(r[1]), int(r[2])] + [float(x) for x in r[3:]] for r in rows],
              open(f"{OUT}/pseudoword.json", "w"), indent=1)


if __name__ == "__main__":
    main()
