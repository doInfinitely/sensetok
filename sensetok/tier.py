"""Which annotator tier to buy. NOT 'LLM vs random' -- that was a foregone conclusion.

Remy: 'the question is which version of you to use, not whether random beats LLM.'

Two decision axes, both on the coarse (2.54-sense) inventory, same 300 items:
  SELF-CONSISTENCY   test-retest within a tier (hA vs hB, sA vs sB). Can this model apply
                     the inventory stably? This bounds token PURITY directly.
  CROSS-TIER         does the cheap model agree with the expensive one? (Haiku vs Sonnet)
                     Disagreement = places the cheap tier gets it wrong that the dear one
                     doesn't. Localise it: is it the frequent function words?

Cost of annotating full wikitext-103 (~64M polysemous tokens, batched by type):
  Haiku  ~$355     Sonnet  ~$1.0k     Opus  ~$5k+
Decision: cheapest tier whose self-consistency clears the bar AND whose disagreement with
the next tier up is not concentrated on the tokens that carry the mass.

Reported as agreement AND chance-corrected kappa (per-item 1/k_i model), because the
coarse inventory has few senses and raw agreement is inflated.
"""
import json, collections
import numpy as np


def load(path):
    return {r["id"]: r["sense"] for r in json.load(open(path))}


def agreement(x, y, items):
    ids = [i for i in x if i in y]
    ag = np.array([x[i] == y[i] for i in ids])
    ks = np.array([len(items[i]["senses"]) for i in ids])
    po = ag.mean(); pe = np.mean(1.0 / ks)
    return po, (po - pe) / (1 - pe + 1e-12), len(ids)


def main():
    items = {}
    for i in range(3):
        for it in json.load(open(f"coarse/items_{i}.json")): items[it["id"]] = it
    P = {k: load(f"coarse/{k}.json") for k in ("hA", "hB", "sA", "sB")}

    print("SELF-CONSISTENCY (test-retest on the coarse inventory)")
    print(f"  {'tier':<10}{'agreement':>11}{'kappa':>9}")
    for tier, (a, b) in (("Haiku", ("hA","hB")), ("Sonnet", ("sA","sB"))):
        po, k, n = agreement(P[a], P[b], items)
        print(f"  {tier:<10}{po:>11.3f}{k:>9.3f}")
    print(f"  reference: WordNet fine inventory, Haiku test-retest kappa 0.824 raw 0.863")
    print(f"  random-merge null at k~2.5: kappa ~0.83  <- coarse must beat THIS, not 0.824")

    print("\nCROSS-TIER (do the tiers agree?)")
    for lab, a, b in (("Haiku vs Sonnet (pass A)","hA","sA"),
                      ("Haiku-A vs Sonnet-B","hA","sB")):
        po, k, n = agreement(P[a], P[b], items)
        print(f"  {lab:<26}{po:>8.3f} agree   kappa {k:.3f}")

    # majority vote per tier, then where do the tiers differ?
    def majority(a, b):
        return {i: (P[a][i] if P[a][i]==P[b][i] else P[a][i]) for i in P[a] if i in P[b]}
    hm, sm = majority("hA","hB"), majority("sA","sB")
    diff = [i for i in hm if i in sm and hm[i]!=sm[i]]
    print(f"\n  items where Haiku-consensus != Sonnet-consensus: {len(diff)}/{len(hm)}")

    # is disagreement concentrated on frequent / high-sense words?
    byw = collections.Counter(items[i]["target"] for i in diff)
    print("  most-disputed words:", ", ".join(f"{w}({c})" for w,c in byw.most_common(8)))
    # flakiness: within-tier disagreement per word
    print("\n  words where HAIKU flips between its own two passes:")
    hflip = collections.Counter(items[i]["target"] for i in P["hA"]
                                if i in P["hB"] and P["hA"][i]!=P["hB"][i])
    sflip = collections.Counter(items[i]["target"] for i in P["sA"]
                                if i in P["sB"] and P["sA"][i]!=P["sB"][i])
    print("   haiku flips:", ", ".join(f"{w}({c})" for w,c in hflip.most_common(8)))
    print("   sonnet flips:", ", ".join(f"{w}({c})" for w,c in sflip.most_common(8)))
    print(f"   haiku total flips {sum(hflip.values())}, sonnet {sum(sflip.values())}")
    json.dump({"hflip":dict(hflip),"sflip":dict(sflip),"crosstier_diff":len(diff)},
              open("out/tier.json","w"), indent=1)


if __name__ == "__main__":
    main()
