# sensetok — word-sense-disambiguated tokenization

One token per *sense*, not per *word*. `bank#river` and `bank#money` are different
symbols, so the embedding of a token is context-free by construction.

Motivation, from the lexicon project: we found `<lex:state>` silently carrying the
statistics of *state* and *station*, and Remy's objection was that this makes the
lexeme's meaning context-dependent when the whole design premise is that it is a
context-free atom the operators act on. Polysemy does the same thing to every
frequent word in English, false decomposition or not. `state` (condition / polity /
to say) was already three words wearing one token.

This is orthogonal to the operator language. It composes with it: sense-tag first,
then decompose morphology over the sense-tagged lexemes.

## The scoring trap (read before running anything)

Sense tagging is **not** a deterministic function of the surface text. The sense is a
latent variable `z`. So a sense-token model defines `P(text) = Σ_z P(text, z)`, and
comparing its `P(text, z*)` against a word model's `P(text)` compares unlike things —
it charges the sense model for predicting information the text never contained.

Two valid moves:

1. **Fixed-`z*` upper bound.** For any particular `z*`, `-log P(text) ≤ -log P(text, z*)`.
   So scoring with the WSD-assigned senses is *conservative*: if the sense model wins
   under `z*`, it wins under marginalization too. Cheap, rigorous, use this first.
2. **Marginalize** over the senses of the observed word at each position. Exact
   marginalization is exponential in sequence length; beam/importance-sample if the
   `z*` bound turns out to be close.

Never report `-log P(text, z*)` as if it were `-log P(text)`.

## Controls that must exist before any claim

- **Random-split null.** Split each polysemous word into `k` tokens *at random*, with
  the same `k` and the same token-frequency profile as the WSD split. Any benefit that
  survives WSD but not random assignment is sense; anything both produce is vocabulary
  size. (Under `z*` scoring a random split should be *penalised* — it adds entropy the
  context cannot predict — so this null also validates the scoring harness.)
- **Frequency-matched null.** Sense tokens are rarer than their parent word. Confirm any
  win is not just "rarer tokens have sharper contexts."
- **No `<unk>` cheat.** Sense splitting multiplies types (~3-4x). Under a fixed vocab
  cap the tail collapses to `<unk>`, which *discards the word* and flatters bits/char.
  Keep an escape hatch; score `-inf` if any scored window contains `<unk>`.
- **bits/char over the ORIGINAL characters**, counting only characters inside the
  scored window.

## Sense inventory: three arms, and my prior

| arm | source | senses | note |
|---|---|---|---|
| `fine` | WordNet synsets | ~206k word-sense pairs | notoriously too fine-grained |
| `coarse` | WordNet supersense (lexname) | 45 | high WSD accuracy, may be too blunt |
| `induced` | cluster contextual vectors per type | chosen per word | corpus-fitted; catches `apple`→Inc. |

Prior CONFIRMED (see RESULTS.md): **`induced` beats `fine`.** Same argument that settled the morphology dictionary
— for an LM, a split earns its keep if it pools statistics usefully, not if it matches a
lexicographer. WordNet also has no sense for *Apple the company*, and Remy asked for
exactly that case on day one.

## Pipeline

    inventory.py   sense inventories + polysemy stats
    vectors.py     SemCor instance embeddings -> sense centroids (LMMS-lite),
                   gloss/hypernym backoff for senses SemCor never saw
    wsd.py         1-NN disambiguator + most-frequent-sense baseline; eval on held-out
                   SemCor files (split by FILE, never by instance)
    induce.py      unsupervised sense induction; k per type by silhouette
    tag.py         tag a corpus -> sense-token stream
    score.py       the harness above: z*-bound, marginalisation, the nulls
    embed.py       train static embeddings on the unambiguous vocabulary

Evaluate WSD only on **polysemous** instances; monosemous ones are free and inflate
accuracy. Report the MFS baseline beside every number (it is ~65-70% and hard to beat).

Uses the lexicon venv (`.venv` symlink). Runs on GPU 1.
