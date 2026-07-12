"""Sense inventories and how polysemous English actually is."""
import collections
from nltk.corpus import wordnet as wn

POS_MAP = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "r": wn.ADV, "s": wn.ADJ}


def senses(lemma, pos=None):
    """Candidate synsets for a lemma, in WordNet's frequency-ish order (sense 1 first)."""
    return wn.synsets(lemma, pos) if pos else wn.synsets(lemma)


def sense_key(lemma, syn):
    for l in syn.lemmas():
        if l.name().lower() == lemma.lower():
            return l.key()
    return syn.name()


def supersense(syn):
    """45 coarse classes: noun.animal, verb.motion, ..."""
    return syn.lexname()


def gloss_text(syn):
    ex = syn.examples()[:2]
    return syn.definition() + (" ; " + " ; ".join(ex) if ex else "")


def polysemy_report(words):
    rows = collections.Counter()
    ss = collections.Counter()
    for w in words:
        s = senses(w)
        if not s: continue
        rows[min(len(s), 10)] += 1
        ss[min(len({supersense(x) for x in s}), 10)] += 1
    return rows, ss
