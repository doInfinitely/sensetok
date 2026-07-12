"""Sense vectors, take 2. Three fixes over vectors.py:

1. LAST-4-LAYER SUM, not the final layer. The final layer of an MLM is specialised
   for token prediction; the layers below carry more lexical semantics. (LMMS uses
   BERT-large's last 4.)
2. HYPERNYM BACKOFF, not gloss vectors, for the 33% of candidate senses SemCor never
   saw. `cos(instance, centroid)` and `cos(instance, gloss_sentence_vector)` are not
   commensurable -- different spaces on one scale -- and the gloss-only senses are
   systematically the rare ones, so the error is systematic, not noise. A missing
   sense instead inherits the mean centroid of its hypernym's other hyponyms, which
   lives in the same space by construction. Gloss only as last resort.
3. A SENSE-FREQUENCY PRIOR, tuned on a dev split. MFS wins because sense frequency is
   enormously skewed; a similarity-only model throws that away. score = cos + b*log p(s).

Dev split is carved from the TRAIN files, so the test files stay untouched.
"""
import collections, json
import numpy as np, torch
from nltk.corpus import semcor, wordnet as wn
from transformers import AutoTokenizer, AutoModel

from sensetok.vectors import load_instances, MODEL, DEV, OUT
from sensetok.inventory import gloss_text, sense_key

L4 = (-4, -3, -2, -1)


@torch.no_grad()
def embed_spans(insts, tok, mod, bs=32):
    V = np.zeros((len(insts), mod.config.dim), dtype=np.float32)
    for i in range(0, len(insts), bs):
        b = insts[i:i + bs]
        enc = tok([t for t, _, _, _ in b], is_split_into_words=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=256)
        out = mod(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True)
        h = sum(out.hidden_states[l] for l in L4)
        for j, (_, s, e, _) in enumerate(b):
            wid = enc.word_ids(j)
            sel = [p for p, w in enumerate(wid) if w is not None and s <= w < e] or [0]
            V[i + j] = h[j, sel].mean(0).float().cpu().numpy()
    return V


@torch.no_grad()
def embed_texts(texts, tok, mod, bs=64):
    V = np.zeros((len(texts), mod.config.dim), dtype=np.float32)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=64)
        out = mod(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True)
        h = sum(out.hidden_states[l] for l in L4)
        m = enc["attention_mask"].to(DEV).unsqueeze(-1)
        V[i:i + bs] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
    return V


def nrm(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def main():
    files = sorted(semcor.fileids())
    perm = np.random.default_rng(0).permutation(len(files))
    test_f = [files[i] for i in perm[:40]]
    dev_f = [files[i] for i in perm[40:80]]
    train_f = [files[i] for i in perm[80:]]
    print(f"files: {len(train_f)} train / {len(dev_f)} dev / {len(test_f)} test")

    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL).to(DEV).eval()
    tr, dv, te = (load_instances(f) for f in (train_f, dev_f, test_f))
    print(f"instances: {len(tr):,} / {len(dv):,} / {len(te):,}", flush=True)

    V = embed_spans(tr, tok, mod)
    by = collections.defaultdict(list)
    for i, (_, _, _, k) in enumerate(tr): by[k].append(i)
    keys = sorted(by)
    C = nrm(np.stack([V[by[k]].mean(0) for k in keys]))
    counts = {k: len(by[k]) for k in keys}
    print(f"centroids: {len(keys):,}", flush=True)

    torch.save({"keys": keys, "C": C, "counts": counts,
                "dev": dv, "dev_vecs": embed_spans(dv, tok, mod),
                "test": te, "test_vecs": embed_spans(te, tok, mod)},
               f"{OUT}/sense_v2.pt")
    print(f"wrote {OUT}/sense_v2.pt")


if __name__ == "__main__":
    main()
