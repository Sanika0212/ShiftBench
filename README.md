# ShiftBench

A grounding checker is two things: a score, and a threshold on that score. The score gets
all the attention. The threshold is where the guarantee lives, and it is fitted on whatever
labelled data exists and then used on data that looks nothing like it.

This benchmark measures what that costs. For every pair of domains it calibrates a detector
to catch 90% of unsupported claims in one domain, applies that threshold in another, and
reports what it actually caught. Then it asks the question a team actually faces: how many
labels from the new domain does it take to get the promise back.

Detectors are ranked by whether the promise survives the move, not by how well they separate
in the domain they were tuned on.

## The measurement

1. **Calibrate** on the source domain: pick the threshold whose recall on unsupported claims
   is the promised 0.90.
2. **Deploy** on the target domain: apply that same threshold, measure achieved recall with a
   bootstrap interval, and record the false alarm rate it cost.
3. **Compare four thresholds**, because the repair matters as much as the damage.
   1. `split`, plain conformal on the source. What a team would ship.
   2. `weighted`, reweighting source claims towards the target with a density ratio. Needs no
      target labels, and has a ceiling that is worth stating in advance: importance
      weighting needs the two domains to overlap, and a domain classifier on these features
      separates them perfectly as soon as they differ much at all. When it does, every
      source claim gets near enough the same weight and the weighted threshold is the plain
      one. Measured while building this, on synthetic domains: the threshold moved 0.096
      where the domains overlapped and 0.017 where they did not, which is the wrong way
      round for a shift repair. So every weighted row carries the classifier's separability
      and the spread of its weights, and a row whose spread is near zero is reported as
      having done nothing rather than as a repair that did not help.
   3. `mondrian`, a separate threshold per group inside the source. The group has to mean
      the same thing on both sides, which rules out the obvious choice. `group_id` keeps one
      document's claims together so a split never puts a document on both sides, and it is
      specific to its domain twice over: legal carries 591 groups across 600 claims, so no
      group holds the 25 unsupported claims a threshold needs, and no target domain contains
      a single one of the source's groups. Fitted on it, every claim took the pooled
      fallback and the method was `split` under another name, identical on all 20
      cross-domain pairs and on 4 of the 5 same-domain ones, with nothing in the output
      saying so. The groups are quantiles of evidence length instead: label-free, present in
      every domain, and coarse enough to fit on. Every mondrian row carries the share of
      target claims that got their own group's threshold, so a run that falls back says so.
   4. `oracle`, fitted on the target's own labels. The ceiling nobody has.
4. **Price the repair**: label k target claims, recalibrate on them, measure on the rest, over
   many draws. The curve says what k buys. The budget a detector is credited with is the
   smallest one that lands within tolerance on 90% of draws in every target domain. Taking
   the smallest over all domains instead credited the easiest one: word-overlap was reported
   as recovering at 50 labels because one target of five crossed the bar there while the
   other four sat between 0.78 and 0.84, and no target cleared it at 100 or 200.
5. **Ask whether anything warned first**: PSI and MMD on the score distributions need no
   labels at all, so if either correlates with the recall actually lost, it is an early
   warning worth acting on. Both are close to direction-blind, coming out nearly the same
   for a pair and for its reverse while the loss does not: an easy source deployed on a
   hard target loses recall, and the reverse over-recalls. So the correlation is reported
   among the pairs where the promise actually breaks, against the absolute loss, and
   against the signed loss over every pair, which is the weakest of the three for a reason
   that is arithmetic rather than empirical. A high reading is a reason to recalibrate and
   measure, not a reading of which way the promise moved.

A domain measured against itself uses a held-out half, so the diagonal of the matrix is an
honest number rather than a tautology.

## The domains

A row is a claim, the evidence a reader should check it against, and whether the evidence
supports it. Everything else about the original dataset is dropped.

| key | source | what a claim is | licence |
| --- | --- | --- | --- |
| scientific | SciFact (bigbio mirror) | a claim against the abstract a reviewer cited | CC BY-NC 2.0 |
| medical | MedHallu (PubMedQA) | an answer sentence against the PubMed context | see dataset card |
| general | RAGTruth | an answer sentence against the retrieved passages | MIT |
| legal | ContractNLI | a hypothesis about a contract against its spans | CC BY-NC-SA 4.0 |
| financial | Fin-Fact | a financial claim against a fact checker's evidence | Apache 2.0 |

Fin-Fact ships its evidence as records of a sentence and the URLs a fact checker archived
for it, and only the sentences are used. The URLs leaked: a refuted claim carries more
archived sources, so the count of URLs alone separates unsupported from supported at AUROC
0.570, and the embedding detector scored 0.580 on this domain while they sat in the evidence
against 0.523 without them. A benchmark about whether evidence supports a claim cannot score
a detector for counting the citations around it.

Three of the five are not redistributed here. SciFact and ContractNLI carry non-commercial
licences, and MedHallu's terms have to be read off its dataset card, so this repository ships
loaders and nothing else. `shiftbench fetch` downloads into `data/`, which is gitignored. No
source text is redistributed here.

Four of the five keep their rows on the Hub's automatic Parquet branch rather than on `main`,
because they ship a loading script and `datasets` 3.x no longer runs scripts. The loaders ask
for that revision by name. SciFact is taken from the bigbio mirror for the same reason.

## The detectors

| name | what it is | cost |
| --- | --- | --- |
| word-overlap | share of claim content words present in the evidence | free |
| embedding-similarity | cosine similarity of sentence embeddings | cheap |
| nli | an NLI model's entailment probability | medium |
| minicheck | a model trained for exactly this judgement | medium |
| judge | a local instruction model asked supported or not | slow |

The judge is listed and is not in the default set. It answers one word, so its score takes
two values, and a threshold at a promised recall needs a score with a distribution: every
quantile of a two-valued score is 0 or 1. Reading the token logprobs would give it one.
Until then it cannot take part in the comparison, which is worth stating because asking a
model directly is what a team tends to reach for first.

Which output of a cross-encoder means "supported" is read from the model's own
`config.json`, not assumed from the shape of the output. It was assumed once, as the last
output, and `cross-encoder/nli-deberta-v3-base` orders its labels contradiction,
entailment, neutral, so that read neutral as support.

`scripts/export_models.py` rebuilds the ONNX detectors from the Hub and reports the largest
difference between each export and PyTorch, so an export can be checked before its scores
are trusted.

The cross-encoders run int8 quantized, and precision is uniform across domains because
otherwise it confounds exactly what is being measured: a threshold fitted on int8 scores and
used on full precision ones would shift for a reason that has nothing to do with domain.
Measured on real pairs rather than the export's own sample, int8 minicheck sits 0.010 from
the full precision export on average, 0.08 at worst, correlates with it at Spearman 0.998,
and flips no decision at a 0.5 cut. The full precision export is kept beside the int8 one
so the cost of quantizing stays measurable, and `meta.json` in each export directory
carries the measured difference from PyTorch.

Measured by `scripts/export_models.py` on its own fixed pairs, on the output each detector
actually reads:

| export | full precision against PyTorch | int8 against PyTorch |
| --- | --- | --- |
| embedding | 1.19e-07 | not quantized |
| nli | 7.28e-11 | 4.25e-03 |
| minicheck | 4.17e-07 | 1.37e-02 |

Both int8 figures sit under the 0.02 tolerance this benchmark reports recall against, and
MiniCheck uses most of it. The full precision exports are faithful to PyTorch to seven
digits or better, so the int8 column is the cost of quantizing and not of exporting.

### Why none of these scores are imported

A sibling project, Evidence Gate, had already scored SciFact and MedHallu with detectors
carrying these same four names, on the same claims, and had kept its per-item scores.
Reusing them looked free: the same checkpoints on the same pairs, and hours of laptop heat
saved. A script here did exactly that, for seven of the twenty domain-detector cells.

Recomputing those cells locally and correlating against the imported score, by Spearman:

| detector | scientific | medical |
| --- | --- | --- |
| word-overlap | 0.57 | 0.64 |
| embedding-similarity | 0.74 | 0.08 |
| nli | 0.93 | not imported |
| minicheck | 0.95 | 0.64 |

None of it was the same measurement, and each row fails for its own reason.

The two cheap detectors were never the same computation: that project's lexical checker is
another library's, and its embedding detector is bge-small-en-v1.5 against all-MiniLM-L6-v2
here, which is why the medical column reaches 0.08.

The two cross-encoders are where the assumption looked safest and still broke. The
checkpoint, the precision, the 512 token limit, the pair order and the tokenizer output are
all identical, and running both projects' export files through the same code on the same
pairs gives the same scores to four decimal places and a Spearman of 1.0000, so the exports
are not the difference either. The difference is that the number stored over there is not a
forward pass. It is the output of a grounding pipeline: the answer is split into claims and
the claim scores are combined with a minimum, long passages are checked in windows of 250
words and the windows combined with a maximum, and a short answer is checked together with
its question. On MedHallu 42.6 percent of rows decompose into two or more claims, which is
why every medical row above is low. On SciFact 99.8 percent are a single claim, but 34
percent of the abstracts run past 250 words and are therefore windowed, which is the
residual gap at 0.93 and 0.95. The medical pairs were not even the same pairs: the claim
there is the answer alone, against each passage separately, and here it is the question and
answer together against the passages joined.

The name matched. The checkpoint matched. The export matched bit for bit. The number still
meant something else, and no amount of checking ids and labels at the boundary would have
caught it, because every id and every label agreed. A benchmark whose whole claim is that
one number moves when the domain changes cannot afford a second thing moving with it, so
every score here is computed by this repository and the import script is gone.

## Running it

```
make install      # uv sync
make fetch        # the five domains into data/, capped per domain
make score        # chunked, resumable, duty cycled
make analyse      # the matrices, budget curves, drift statistics, leaderboard
make serve        # the leaderboard at 127.0.0.1:8090
make check        # lint, types, tests, SQL
```

Scoring is the only expensive step and it is built to be interrupted. Each chunk is appended
to a per-claim cache, so stopping costs at most one chunk and starting again skips everything
already scored.

### On the machine doing the work

This was developed on a fanless 16 GB laptop that rebooted three times under an earlier
project's load. The panic log named the cause, and it was not heat: memory exhaustion, swap
thrash, and the kernel's watchdog giving up on a display server that had not checked in for
121 seconds.

So `src/shiftbench/thermal.py` bounds the run. It caps the thread pools, idles between chunks
so the chip is not at full tilt, and pauses when the kernel itself reports memory trouble,
reading `kern.memorystatus_level` and `kern.memorystatus_vm_pressure_level` rather than
counting free pages. Free pages are a bad signal on macOS: the figure sits near a quarter of
memory whatever is happening, because the system compresses rather than wastes, and a gate
built on it never opens. Background scheduling is available and off by default: measured, it
pinned the work to the efficiency cores and cost about ten times the wall clock, which is
more total heat for the same work.

<!-- RESULTS:start -->
## Results

Every threshold promises 90% recall on unsupported claims, fitted on one domain and used on another. 5 domains give 20 ordered pairs, over 3,000 claims in all.

| detector | promise kept | mean gap | worst gap | false alarms | labels to recover |
| --- | --- | --- | --- | --- | --- |
| nli | 55% | -0.067 | -0.437 | 0.62 | none up to 400 |
| word-overlap | 55% | -0.021 | -0.263 | 0.78 | none up to 400 |
| minicheck | 50% | -0.150 | -0.687 | 0.54 | none up to 400 |
| embedding-similarity | 50% | -0.072 | -0.503 | 0.73 | none up to 400 |

"Promise kept" is the share of ordered pairs whose achieved recall came within 0.02 of what was promised. The gap is achieved minus promised, so it is good near zero and a large positive one means the detector flags more than it was asked to. That is why detectors keeping as many promises are ranked by what the promise cost in false alarms rather than by the gap.

"Labels to recover" is the smallest budget at which every target domain lands within tolerance on 90% of draws. Every target, not the easiest one: a budget that recovers the promise on one domain of five has not recovered the promise.

No budget up to 400 cleared that bar for any of the 4 detectors. Those cells name the budget they were measured against rather than sitting empty, because an empty cell reads the same whether the measurement failed or was never taken. The share of draws the hardest target managed, at the budget that suited it best: nli at 76%, word-overlap at 78%, minicheck at 74%, embedding-similarity at 74%.

**embedding-similarity**, recall achieved when the threshold is fitted on the row's domain and used on the column's, against a promise of 90%. The diagonal is a held-out half of the same domain, so it is a measurement and not a tautology.

| fitted on | financial | general | legal | medical | scientific |
| --- | --- | --- | --- | --- | --- |
| financial | 0.91 | 0.71 | 0.94 | 0.57 | 0.94 |
| general | 1.00 | 0.92 | 1.00 | 0.95 | 1.00 |
| legal | 0.85 | 0.66 | 0.93 | 0.45 | 0.92 |
| medical | 1.00 | 0.84 | 1.00 | 0.89 | 0.99 |
| scientific | 0.82 | 0.64 | 0.87 | 0.40 | 0.90 |

**minicheck**, recall achieved when the threshold is fitted on the row's domain and used on the column's, against a promise of 90%. The diagonal is a held-out half of the same domain, so it is a measurement and not a tautology.

| fitted on | financial | general | legal | medical | scientific |
| --- | --- | --- | --- | --- | --- |
| financial | 0.88 | 1.00 | 1.00 | 1.00 | 0.99 |
| general | 0.62 | 0.91 | 0.99 | 0.98 | 0.98 |
| legal | 0.33 | 0.21 | 0.89 | 0.62 | 0.86 |
| medical | 0.47 | 0.58 | 0.97 | 0.91 | 0.95 |
| scientific | 0.39 | 0.34 | 0.93 | 0.78 | 0.97 |

**nli**, recall achieved when the threshold is fitted on the row's domain and used on the column's, against a promise of 90%. The diagonal is a held-out half of the same domain, so it is a measurement and not a tautology.

| fitted on | financial | general | legal | medical | scientific |
| --- | --- | --- | --- | --- | --- |
| financial | 0.91 | 0.85 | 0.98 | 1.00 | 0.96 |
| general | 0.96 | 0.86 | 0.99 | 1.00 | 0.99 |
| legal | 0.62 | 0.46 | 0.93 | 0.89 | 0.78 |
| medical | 0.65 | 0.49 | 0.92 | 0.95 | 0.80 |
| scientific | 0.77 | 0.64 | 0.95 | 0.95 | 0.89 |

**word-overlap**, recall achieved when the threshold is fitted on the row's domain and used on the column's, against a promise of 90%. The diagonal is a held-out half of the same domain, so it is a measurement and not a tautology.

| fitted on | financial | general | legal | medical | scientific |
| --- | --- | --- | --- | --- | --- |
| financial | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| general | 0.66 | 0.93 | 0.96 | 0.92 | 0.86 |
| legal | 0.64 | 0.88 | 0.99 | 0.88 | 0.84 |
| medical | 0.64 | 0.89 | 0.96 | 0.90 | 0.85 |
| scientific | 0.74 | 0.93 | 0.98 | 0.96 | 0.95 |

### What the repairs are worth

The same pairs under each way of choosing the threshold. `oracle` is fitted on the target's own labels and is the ceiling nobody has; `split` is what a team ships.

| method | mean recall | within tolerance | mean false alarms |
| --- | --- | --- | --- |
| split | 0.822 | 52% | 0.67 |
| weighted | 0.835 | 60% | 0.67 |
| mondrian | 0.844 | 57% | 0.69 |
| oracle | 0.912 | 100% | 0.71 |

The mondrian row carries the same kind of diagnostic. Its groups are quantiles of evidence length, so a group means the same thing in the target as in the source, and 100% of target claims were scored against their own group's threshold rather than the pooled fallback. It agrees exactly with split on 18% of pairs.

The weighted row comes with diagnostics, because there are two different reasons it could sit near the split row. The domain classifier separated the two domains perfectly on 10% of pairs, and the mean spread of its importance weights was 3.06.
So the weights did vary, and the reweighting still moved mean recall by +0.012. That is a real negative result rather than a degenerate estimator: the density ratio had something to work with and reweighting the source towards the target did not recover the recall that the shift cost.

### What it costs to get the promise back

Label k claims from the new domain, refit the threshold on them, measure on the rest, over many draws. Averaged over domains and detectors, against a promise of 90%.

| labels | mean recall | spread | share of draws within tolerance | false alarms |
| --- | --- | --- | --- | --- |
| 25 | 0.928 | 0.075 | 79% | 0.78 |
| 50 | 0.925 | 0.056 | 80% | 0.76 |
| 100 | 0.915 | 0.046 | 78% | 0.73 |
| 200 | 0.910 | 0.036 | 78% | 0.72 |
| 400 | 0.913 | 0.033 | 78% | 0.72 |

The column that moves is the spread, from 0.075 at 25 labels to 0.033 at 400. Mean recall does not climb with the budget and is not supposed to: refitting on the target's own claims puts the threshold in the right place at every budget, and what more labels buy is a threshold that lands there reliably rather than one that happens to. A small sample gives a noisy threshold, and a noisy threshold that errs towards flagging looks like good recall on average while being wrong in both directions on any given draw.

### Did anything warn before the labels arrived

PSI and MMD on the score distributions need no target labels, so they are the only warning available in advance. Spearman correlation against the recall the shipped threshold lost, over 80 ordered pairs.

| statistic | against signed loss | against absolute loss | among pairs that broke |
| --- | --- | --- | --- |
| PSI | -0.08 | +0.44 | +0.54 |
| MMD | +0.00 | +0.46 | +0.56 |

The last column is the one to read, over the 40 pairs where the promise actually broke. The first is near zero or negative, and that is arithmetic rather than a result: a score-distribution statistic is close to direction-blind, coming out nearly the same for a pair and for its reverse, while the loss does not. Shipping a threshold from an easy domain to a hard one loses recall and the reverse over-recalls, which pins the loss at its floor however large the drift, so correlating over both directions mixes a relationship with its mirror image. A high reading means recalibrate and measure, not that the promise moved in a particular direction.
<!-- RESULTS:end -->

## The warehouse and the site

`infra/bigquery/schema.sql` types the four results tables, and `src/shiftbench/site.py` is a
FastAPI app that renders the leaderboard from the parquet files. Neither has been applied:
there is no GCP project behind this repository and nothing here has been deployed.

## Licence

Apache 2.0. See `LICENSE` and `NOTICE`.
