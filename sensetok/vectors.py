"""Sense centroids from SemCor: average the contextual embedding of each annotated
instance of a sense (LMMS-lite, Loureiro & Jorge 2019).

Senses SemCor never saw get a backoff vector from their gloss. Roughly 60% of
WordNet senses never appear in SemCor, so the backoff is not a corner case.

Split by FILE. Splitting by instance leaks: the same document mentions the same
sense repeatedly in near-identical contexts, and a 1-NN classifier then reads its
own training data back.
"""
import collections, json, os, sys
import numpy as np, torch
from nltk.corpus import semcor, wordnet as wn
from transformers import AutoTokenizer, AutoModel

from sensetok.inventory import gloss_text

MODEL = "distilbert-base-uncased"
DEV = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
OUT = "out"


def load_instances(fileids):
    """-> list of (tokens, start, end, sense_key)"""
    out = []
    for fid in fileids:
        for sent in semcor.tagged_sents(fid, tag="sem"):
            toks, spans = [], []
            for chunk in sent:
                lab = chunk.label() if hasattr(chunk, "label") else None
                words = chunk.leaves() if hasattr(chunk, "leaves") else list(chunk)
                s = len(toks); toks += [str(w) for w in words]; e = len(toks)
                if lab is not None and hasattr(lab, "key"):
                    spans.append((s, e, lab.key()))
            for s, e, k in spans:
                out.append((toks, s, e, k))
    return out


@torch.no_grad()
def embed_spans(insts, tok, mod, bs=32):
    """mean-pool last hidden states over the wordpieces of the annotated span."""
    vecs = np.zeros((len(insts), mod.config.dim), dtype=np.float32)
    for i in range(0, len(insts), bs):
        batch = insts[i:i + bs]
        enc = tok([t for t, _, _, _ in batch], is_split_into_words=True,
                  return_tensors="pt", padding=True, truncation=True, max_length=256)
        h = mod(**{k: v.to(DEV) for k, v in enc.items()}).last_hidden_state
        for j, (_, s, e, _) in enumerate(batch):
            wids = enc.word_ids(j)
            sel = [p for p, w in enumerate(wids) if w is not None and s <= w < e]
            if not sel: sel = [0]
            vecs[i + j] = h[j, sel].mean(0).float().cpu().numpy()
    return vecs


@torch.no_grad()
def embed_glosses(keys, tok, mod, bs=64):
    texts = []
    for k in keys:
        try: syn = wn.lemma_from_key(k).synset()
        except Exception: texts.append(""); continue
        texts.append(gloss_text(syn))
    V = np.zeros((len(keys), mod.config.dim), dtype=np.float32)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=64)
        h = mod(**{k: v.to(DEV) for k, v in enc.items()}).last_hidden_state
        m = enc["attention_mask"].to(DEV).unsqueeze(-1)
        V[i:i + bs] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
    return V


def main():
    files = sorted(semcor.fileids())
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(files))
    test_f = [files[i] for i in perm[:40]]
    train_f = [files[i] for i in perm[40:]]
    print(f"semcor files: {len(train_f)} train / {len(test_f)} test (split by FILE)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL).to(DEV).eval()

    tr = load_instances(train_f)
    te = load_instances(test_f)
    print(f"annotated instances: {len(tr):,} train / {len(te):,} test", flush=True)

    V = embed_spans(tr, tok, mod)
    by = collections.defaultdict(list)
    for i, (_, _, _, k) in enumerate(tr): by[k].append(i)
    keys = sorted(by)
    C = np.stack([V[by[k]].mean(0) for k in keys])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    print(f"sense centroids from SemCor: {len(keys):,}", flush=True)

    np.save(f"{OUT}/sense_centroids.npy", C)
    json.dump(keys, open(f"{OUT}/sense_keys.json", "w"))
    json.dump({"train": train_f, "test": test_f}, open(f"{OUT}/split.json", "w"))
    # cache test instances for wsd.py
    torch.save({"test": te, "test_vecs": embed_spans(te, tok, mod)}, f"{OUT}/test.pt")
    print("wrote out/sense_centroids.npy, out/sense_keys.json, out/test.pt")

    tot = sum(len(wn.synsets(l)) for l in list(wn.all_lemma_names())[:1])  # touch wn
    covered = len(keys)
    allsenses = sum(len(s.lemmas()) for s in wn.all_synsets())
    print(f"\nWordNet word-sense pairs: {allsenses:,}")
    print(f"covered by SemCor:       {covered:,} ({covered/allsenses:.1%}) "
          f"-> the rest need gloss backoff")


if __name__ == "__main__":
    main()
