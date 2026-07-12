# sensetok — what the first day established

All numbers: distilbert-base-uncased, SemCor split by FILE (272 train / 40 dev / 40 test),
accuracy on **polysemous** instances only, MFS baseline printed beside everything.

## WSD works, but the frequency prior does most of the work

| variant | poly acc | vs MFS (0.687) |
|---|---|---|
| 1-NN, last layer, gloss backoff, no prior | 0.662 | −0.025 |
| + last-4-layer sum, hypernym backoff | 0.661 | −0.026 |
| + sense-frequency prior (β=0.02) | **0.711** | +0.023 |
| coarse (45 supersenses), same model | **0.804** | +0.027 (MFS 0.777) |

Similarity alone **loses to picking sense 1**. The learned part contributes ~+0.02;
the prior contributes ~+0.06. Any claim that "contextual embeddings disambiguate
words" has to be stated against MFS or it is not a claim.

Design error found and fixed: `cos(instance, centroid)` and `cos(instance, gloss_vector)`
are not commensurable — different spaces on one scale — and gloss-backed senses are
systematically the rare ones, so the error is systematic. Backoff now inherits the mean
centroid of hypernym siblings, which lives in the same space by construction.

## The central risk: tagging noise dwarfs the problem it solves

29% of polysemous instances get the WRONG fine sense; 20% wrong at supersense level.

For comparison, in the lexicon project a *false morphological decomposition* corrupts
1.4% of word occurrences, and that was serious enough to rebuild the dictionary over.

Worse in kind: a false decomposition is a deterministic, invertible re-spelling
(`station → <lex:state> <op:suf.ion>`); the word is recoverable and the cost is bounded
at 0.0034 bits/char. A WSD error puts two genuinely different meanings under one token,
*inconsistently across contexts*. That is the exact pathology sense tokens exist to remove.

WordNet granularity is the culprit: `give` has 45 senses (we tag it at 0.33), `make` 51
(0.60), `find` 18 (0.47). Reliability by lemma, on dev:

    acc >= 0.95:  256 lemmas covering 21.9% of polysemous instances
    acc >= 0.80:  469 lemmas covering 43.2%
    acc >= 0.70:  564 lemmas covering 64.0%

Vocabulary blowup over the top 29,182 words: coarse 2.38x, fine 3.66x.

## Induced senses: real signal, but "separable" is the wrong criterion

RETRACTED, stated here because I wrote it down before testing it: *"an induced inventory
has no error rate — the clusters ARE the definition, so purity is 100% w.r.t. itself."*
True and vacuous. A **random** split is also 100% pure w.r.t. itself. Purity against your
own inventory is a tautology; the question is whether the inventory tracks meaning.

`induce.py`, `topic_control.py`, on SemCor dev+test (47k instances, gold senses):

| | ARI(clusters, sense) | ARI(clusters, document) |
|---|---|---|
| polysemous types (n=196) | **+0.169** | +0.101 |
| monosemous types (n=12) | +0.000 | +0.102 |
| size-matched permutation null | +0.0006 | — |

Monosemous words have zero sense signal by construction and the **same** topic signal.
So clustering finds ~0.10 ARI of document structure unconditionally, and polysemous words
carry a real sense component on top. Induction works — but silhouette cannot see the
difference: monosemous types split 100% of the time with **higher** mean silhouette
(0.189) than polysemous ones (0.169). Separability is not sensehood.

40% of polysemous types (79/196) are topic-dominated: `daily` (ARI_doc 1.000, ARI_sense
0.000), `patient`, `hour`, `century`. Clean sense splits: `too` (1.000), `little` (0.902),
`so` (0.793), `add` (0.745).

**⇒ The unsupervised split criterion is: cluster, then REJECT the split if the clusters
predict the DOCUMENT.** Document id is free at training time; no annotation needed; it
targets exactly the failure mode. Silhouette does not.

Limit: `ARI(gold sense, document) = +0.117`. Senses genuinely correlate with documents
(*bank* really does live in finance articles), so this filter discards some true splits
along with the topic ones. Precision/recall dial, not a clean separator.

Plan:
1. `tag.py` — induce on a corpus with document ids; split a type only if ARI(clusters,doc)
   is low and cluster count is stable under bootstrap.
2. Score with the z*-upper bound and the random-split null (see README).
3. Train static embeddings on the unambiguous vocabulary; check `bank#river` is near
   `river` and `apple#company` near `microsoft`.

## Reproduce

    .venv/bin/python -m sensetok.vectors2   # sense centroids  -> out/sense_v2.pt
    .venv/bin/python -m sensetok.wsd2       # tune beta on dev, report test


## LLM annotation (Haiku pilot, 300 SemCor polysemous instances)

Remy: *"why not just annotate all the polysemous words in wikitext, how hard could that
be?"* — Scale: 61.8% of wikitext tokens are polysemous (8.6 candidate senses per token).
Batched by word TYPE (glosses once, then many contexts), ~569M input tokens for the
200k-paragraph subset: **~$355 on Haiku with the batch API**, ~$1,600 for full
wikitext-103. Affordable. The clustering detour was unnecessary.

| subset | n | MFS | 1-NN | Haiku |
|---|---|---|---|---|
| all | 300 | 0.733 | 0.760 | 0.720 |
| 2-3 senses | 96 | 0.885 | 0.885 | 0.760 |
| 4-7 senses | 100 | 0.740 | 0.770 | **0.820** |
| 8+ senses | 104 | 0.587 | 0.635 | 0.587 |
| coarse (supersense) | 236 | 0.822 | 0.860 | 0.847 |

Haiku loses on the EASY cases and wins on the real ones. Inspection of disagreements:
`hijacker` gold *"a holdup man who stops a vehicle and steals from it"* vs Haiku
*"someone who uses force to take over a vehicle"* — Haiku is right. Likewise `wash`,
`manufacturer`. These are WordNet granularity artifacts; human ITA on fine-grained
WordNet is ~67-80%, so 0.72 is near the ceiling of what "gold" means. And the 1-NN was
FIT on SemCor, so accuracy-vs-gold systematically favours it.

**⇒ Accuracy is the wrong metric. Measure TEST-RETEST AGREEMENT.**
A tokenizer needs the annotator to be self-consistent, not lexicographically correct.
A consistent labeling defines a valid inventory even if every lexicographer disagrees.
(Third time this principle has appeared: consistency > etymology in the morphology
dictionary; consistency > correctness here.)

| subset | n | test-retest |
|---|---|---|
| all | 300 | **0.863** |
| 2-3 senses | 96 | 0.896 |
| 4-7 senses | 100 | 0.910 |
| 8+ senses | 104 | 0.788 |

Of 259 self-consistent items: 199 match gold, **60 consistently disagree** — usable
tokens, not errors. Only the 41 inconsistent items are unusable.

**Token purity ceiling 0.863 => 13.7% of sense tokens carry the wrong meaning.**
Ten times the 1.4% contamination from false morphological decomposition. With WordNet's
inventory, sense tokens are NOT worth minting, and a better annotator cannot fix it.

Worst agreement: `be` 3/5, `have` 2/3, `had`, `is` — the highest-frequency verbs, which
WordNet splits 13-20 ways. The frequent words carry the probability mass AND the
impossible distinctions. Same structure as the morphology result.

**Next (cheap, decisive):** have the LLM PROPOSE a small inventory per word from sampled
occurrences, then gate on test-retest agreement >= 0.95. Does a coarse LLM-defined
inventory take the frequent verbs from 0.79 to ~0.97? If `be`/`have` stay unstable under
any inventory, they have no discrete senses to tokenize — split the rest.


## Annotator TIER decides it, not the inventory (2026-07-10)

Remy: *"the question is which version of you to use, not whether random beats LLM."*
Correct. The random-merge null was answering a foregone question. The decision axis is the
model tier, and it has a dollar figure: Haiku ~$355 for full wikitext-103, Sonnet ~$1k.

Coarse inventory: LLM merged WordNet + Wiktionary senses down to **2.54 senses/word** (from
7.5), using real corpus contexts. Then the SAME 300 pilot items, two independent passes per
tier, scored by chance-corrected self-consistency (per-item 1/k model).

| tier | test-retest kappa | vs random-merge bar (0.854) | own-pass flips /300 |
|---|---|---|---|
| Haiku | 0.802 | **−0.052 (below null)** | 31 |
| Sonnet | 0.943 | **+0.088 (clears it)** | 9 |

**The bar is the random-merge-to-same-k null, not WordNet's 0.824** — coarsening buys
consistency for free (merging any two confusable senses removes a confusion), so a coarse
inventory must beat an *arbitrary* merge of the same granularity, which sits at kappa 0.854.

- **Haiku is below the null.** It applies its own designed inventory LESS consistently than
  a random bucketing of the same size. Its sense tokens are not mintable — 20% would carry
  the wrong sense, worse than coin-flip merging. $355 buys an unusable annotation.
- **Sonnet clears the null by +0.088** (kappa 0.943, 9 flips vs Haiku's 31). Sense
  tokenization IS mintable at the Sonnet tier, ~$1k for full wikitext-103.

This overturns the earlier "29% tagging error kills sense tokens" verdict: that was Haiku +
WordNet-fine granularity. Fix BOTH the model (Haiku→Sonnet) and the inventory
(7.5→2.5 senses) and it works.

Residual caveat: cross-tier agreement is only kappa 0.783, and the Haiku/Sonnet disputes
cluster on the highest-frequency function words (`be`, `believed`) — the ones carrying the
most tokens. So even Sonnet has instability exactly where it is most expensive. 300 items,
single null bootstrap.

The anchoring worry (80% sense-0) was unfounded: SemCor says the top sense covers 72% of
polysemous instances and the top two cover 89%, and all four passes disambiguate at
20-25% — so the skew is the language, not lazy annotation.
