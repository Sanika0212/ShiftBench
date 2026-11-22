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
   3. `mondrian`, a separate threshold per group inside the source.
   4. `oracle`, fitted on the target's own labels. The ceiling nobody has.
4. **Price the repair**: label k target claims, recalibrate on them, measure on the rest, over
   many draws. The curve says what k buys.
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

Three of the five are not redistributed here. SciFact and ContractNLI carry non-commercial
licences, and MedHallu's terms have to be read off its dataset card, so this repository ships
loaders and nothing else. `shiftbench fetch` downloads into `data/`, which is gitignored. No
source text is redistributed here.

Four of the five keep their rows on the Hub's automatic Parquet branch rather than on `main`,
because they ship a loading script and `datasets` 3.x no longer runs scripts. The loaders ask
for that revision by name. SciFact is taken from the bigbio mirror for the same reason, and
that mirror is also the one a sibling project scored, so the claim ids line up with the
cached scores `scripts/import_evidence_gate.py` reuses.

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

The cross-encoders run int8 quantized, and that is a decision the measurement forces rather
than a performance preference. `scripts/import_evidence_gate.py` reuses per-item scores a
sibling project already computed for the scientific and medical domains, and those are int8.
Precision has to be uniform across domains or it confounds exactly what is being measured:
int8 moves MiniCheck's probabilities by up to 0.0104 and the NLI model's by up to 0.0063,
against a tolerance of 0.02 on recall. A threshold fitted on int8 scores and used on full
precision ones would shift for a reason that has nothing to do with domain. The full
precision export is kept beside the int8 one so the cost of quantizing stays measurable.

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

## Results

Not yet measured. `make score` then `make analyse` produces `results/`, and the numbers go
here when they exist rather than before.

## The warehouse and the site

`infra/bigquery/schema.sql` types the four results tables, and `src/shiftbench/site.py` is a
FastAPI app that renders the leaderboard from the parquet files. Neither has been applied:
there is no GCP project behind this repository and nothing here has been deployed.

## Licence

Apache 2.0. See `LICENSE` and `NOTICE`.
