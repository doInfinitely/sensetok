"""Wiktionary sense inventories via the REST definition API.

Remy: "you could literally go through wiktionary yourself and mop up the senses."
The insight is that inventory construction is PER TYPE (24,405 polysemous wikitext words
above freq 10) while annotation is PER TOKEN (14.1M). Fixing the inventory is three
orders of magnitude cheaper than annotating against a broken one -- and the inventory is
what is failing (test-retest 0.863 on WordNet, 0.788 on its 8+ sense words).

WordNet: `hijacker` = "a holdup man who stops a vehicle and steals from it".
Wiktionary: `hijacker` = "someone who hijacks" | "hijackware".
"""
import json, re, sys, time, html
import urllib.request, urllib.error

API = "https://en.wiktionary.org/api/rest_v1/page/definition/"
TAG = re.compile(r"<[^>]+>")
POS_MAP = {"n": "Noun", "v": "Verb", "a": "Adjective", "s": "Adjective", "r": "Adverb"}


def clean(s):
    return re.sub(r"\s+", " ", html.unescape(TAG.sub("", s))).strip()


def fetch(word, retries=3):
    url = API + urllib.parse.quote(word.replace(" ", "_"))
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sensetok/0.1 (research)"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(1 + a)
        except Exception:
            time.sleep(1 + a)
    return None


def senses(word, pos):
    """-> list of {gloss, examples} for the requested POS."""
    d = fetch(word)
    if not d or "en" not in d: return []
    want = POS_MAP.get(pos)
    out = []
    for block in d["en"]:
        if want and block.get("partOfSpeech") != want: continue
        for x in block.get("definitions", []):
            g = clean(x.get("definition", ""))
            if not g or len(g) < 3: continue
            ex = [clean(e) for e in x.get("examples", [])][:2]
            out.append({"gloss": g, "examples": ex})
    return out


if __name__ == "__main__":
    import collections
    from nltk.corpus import wordnet as wn
    items = []
    for k in range(3): items += json.load(open("pilot/shard_%d.json" % k))
    targets = sorted({(it["target"], it["pos"]) for it in items})
    print(f"distinct (target, pos) in pilot: {len(targets)}", flush=True)

    out, rows = {}, []
    import os
    cache = json.load(open("out/wikt_senses.json")) if os.path.exists("out/wikt_senses.json") else {}
    for i, (w, p) in enumerate(targets):
        key = f"{w}|{p}"
        s = cache[key] if key in cache else senses(w, p)
        nw = len(wn.synsets(w, p if p != "s" else "a"))
        out[f"{w}|{p}"] = s
        rows.append((w, p, nw, len(s)))
        if (i + 1) % 40 == 0: print(f"  fetched {i+1}/{len(targets)}", flush=True)
        time.sleep(0.02)
    json.dump(out, open("out/wikt_senses.json", "w"), indent=1)

    got = [r for r in rows if r[3] > 0]
    import numpy as np
    W = np.array([r[2] for r in got]); K = np.array([r[3] for r in got])
    print(f"\nfetched senses for {len(got)}/{len(rows)} targets")
    print(f"  mean senses  WordNet {W.mean():.2f}   Wiktionary {K.mean():.2f}")
    print(f"  median       WordNet {np.median(W):.0f}      Wiktionary {np.median(K):.0f}")
    print(f"  Wiktionary coarser on {(K<W).sum()}/{len(got)} words, finer on {(K>W).sum()}")
    print(f"\nthe words that broke test-retest agreement:")
    print(f"  {'word':<10}{'pos':>4}{'WordNet':>9}{'Wiktionary':>12}")
    for w, p, nw, nk in rows:
        if w in ("be", "have", "had", "is", "work", "make", "give", "early", "wash"):
            print(f"  {w:<10}{p:>4}{nw:>9}{nk:>12}")
    print("\nexample — 'hijacker':")
    for s in out.get("hijacker|n", [])[:3]: print("   -", s["gloss"][:70])
