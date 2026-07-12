"""WSD with hypernym backoff + frequency prior. Tune on dev, report on test."""
import collections, json
import numpy as np, torch
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel

from sensetok.vectors2 import embed_texts, nrm
from sensetok.vectors import MODEL, DEV, OUT
from sensetok.inventory import sense_key, gloss_text


def build(keys, C, counts):
    kidx = {k: i for i, k in enumerate(keys)}
    tot = sum(counts.values())
    prior = {k: counts[k] / tot for k in keys}
    return kidx, prior, tot


def hypernym_vec(k, kidx, C):
    """mean centroid of the sibling senses under this sense's hypernyms."""
    try: syn = wn.lemma_from_key(k).synset()
    except Exception: return None
    sibs = []
    for h in syn.hypernyms() + syn.instance_hypernyms():
        for kid in h.hyponyms():
            for l in kid.lemmas():
                if l.key() in kidx: sibs.append(kidx[l.key()])
    if not sibs:
        for l in syn.lemmas():                     # own lemmas' other keys
            if l.key() in kidx: sibs.append(kidx[l.key()])
    return C[sibs].mean(0) if sibs else None


def evaluate(insts, IV, kidx, C, prior, back, beta):
    n = collections.Counter(); hit = collections.Counter()
    for i, (_, _, _, g) in enumerate(insts):
        try:
            lem = wn.lemma_from_key(g); syn = lem.synset()
        except Exception: continue
        cs = [sense_key(lem.name(), s) for s in wn.synsets(lem.name(), syn.pos())]
        if not cs: continue
        b = "poly" if len(cs) >= 2 else "mono"
        n[b] += 1; n["all"] += 1
        if cs[0] == g: hit[f"mfs_{b}"] += 1; hit["mfs_all"] += 1
        sc = []
        for k in cs:
            v = C[kidx[k]] if k in kidx else back.get(k)
            s = float(IV[i] @ v) if v is not None else -1.0
            sc.append(s + beta * np.log(prior.get(k, 1e-7)))
        p = cs[int(np.argmax(sc))]
        if p == g: hit[f"knn_{b}"] += 1; hit["knn_all"] += 1
    return n, hit


def main():
    d = torch.load(f"{OUT}/sense_v2.pt", weights_only=False)
    keys, C, counts = d["keys"], d["C"], d["counts"]
    kidx, prior, _ = build(keys, C, counts)
    DV, TE = nrm(d["dev_vecs"]), nrm(d["test_vecs"])

    need = set()
    for insts in (d["dev"], d["test"]):
        for (_, _, _, g) in insts:
            try: lem = wn.lemma_from_key(g); syn = lem.synset()
            except Exception: continue
            for s in wn.synsets(lem.name(), syn.pos()):
                k = sense_key(lem.name(), s)
                if k not in kidx: need.add(k)
    need = sorted(need)
    back = {}
    miss = []
    for k in need:
        v = hypernym_vec(k, kidx, C)
        if v is None: miss.append(k)
        else: back[k] = v / (np.linalg.norm(v) + 1e-9)
    print(f"backoff needed {len(need):,}: hypernym {len(back):,}, gloss {len(miss):,}")
    if miss:
        tok = AutoTokenizer.from_pretrained(MODEL)
        mod = AutoModel.from_pretrained(MODEL).to(DEV).eval()
        texts = []
        for k in miss:
            try: texts.append(gloss_text(wn.lemma_from_key(k).synset()))
            except Exception: texts.append("")
        G = nrm(embed_texts(texts, tok, mod))
        for k, v in zip(miss, G): back[k] = v

    print("\ntuning beta (frequency prior) on DEV, polysemous accuracy:")
    best, bb = -1, 0.0
    for beta in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
        n, h = evaluate(d["dev"], DV, kidx, C, prior, back, beta)
        acc = h["knn_poly"] / n["poly"]
        mfs = h["mfs_poly"] / n["poly"]
        print(f"   beta {beta:<5} 1-NN {acc:.4f}   (dev MFS {mfs:.4f})")
        if acc > best: best, bb = acc, beta
    print(f"\nchosen beta = {bb}\n")

    n, h = evaluate(d["test"], TE, kidx, C, prior, back, bb)
    print(f"TEST  {'subset':<8}{'n':>9}{'MFS':>10}{'1-NN':>10}{'delta':>10}")
    print("-" * 49)
    for b in ("all", "poly", "mono"):
        if not n[b]: continue
        m, k = h[f"mfs_{b}"] / n[b], h[f"knn_{b}"] / n[b]
        print(f"      {b:<8}{n[b]:>9,}{m:>10.3f}{k:>10.3f}{k-m:>+10.3f}")
    print(f"\nvectors.py (last layer, gloss backoff, no prior) got poly 0.662 vs MFS 0.687")
    json.dump({"beta": bb}, open(f"{OUT}/wsd_config.json", "w"))


if __name__ == "__main__":
    main()
