-- The four results tables, as BigQuery DDL.
--
-- `shiftbench analyse` writes results/{pairs,budgets,drift,leaderboard}.parquet. These are
-- the same tables, typed, for the case where the leaderboard is served from a warehouse
-- rather than from the parquet files on disk.
--
-- Loading them is one command per table, for example:
--
--   bq load --source_format=PARQUET shiftbench.pairs results/pairs.parquet
--
-- Nothing here has been applied. There is no GCP project behind this repository, so these
-- statements are validated by `make sql-check` (sqlfluff) and nothing else.

CREATE SCHEMA IF NOT EXISTS shiftbench OPTIONS (location = 'US');

-- One row per detector, source domain, target domain and threshold method: the promise
-- that was made on the source and what the target actually delivered.
CREATE TABLE IF NOT EXISTS shiftbench.pairs (
  detector STRING NOT NULL,
  source STRING NOT NULL,
  target STRING NOT NULL,
  method STRING NOT NULL,       -- split, weighted, mondrian, oracle
  threshold FLOAT64 NOT NULL,
  promised FLOAT64 NOT NULL,
  achieved FLOAT64 NOT NULL,
  achieved_low FLOAT64,         -- bootstrap interval; null for mondrian
  achieved_high FLOAT64,
  false_alarms FLOAT64 NOT NULL,
  gap FLOAT64 NOT NULL,         -- achieved minus promised, so negative is a broken promise
  n_source INT64 NOT NULL,
  n_target INT64 NOT NULL,
  -- Set on the weighted method only, NULL elsewhere. weight_spread near zero means the
  -- reweighting changed nothing and that row is the split threshold under another name;
  -- domain_separability at 1.0 means the domains do not overlap, so the weights are an
  -- extrapolation rather than an estimate.
  weight_spread FLOAT64,
  domain_separability FLOAT64,
  -- Set on the mondrian method only, NULL elsewhere: the share of target claims whose own
  -- group had a threshold rather than the pooled fallback. 0.0 means the method never ran,
  -- which is what happened while the groups were document ids no target domain contains.
  group_coverage FLOAT64
)
CLUSTER BY detector, source, target;

-- What the repair costs: label k claims in the target, recalibrate on them, measure on the
-- rest, repeated over draws.
CREATE TABLE IF NOT EXISTS shiftbench.budgets (
  detector STRING NOT NULL,
  target STRING NOT NULL,
  labels INT64 NOT NULL,
  recall FLOAT64 NOT NULL,
  recall_sd FLOAT64 NOT NULL,
  within_tolerance FLOAT64 NOT NULL,
  false_alarm_rate FLOAT64 NOT NULL,
  draws FLOAT64 NOT NULL
)
CLUSTER BY detector, target;

-- Label-free statistics on the score distributions, which is all a team has before the
-- target's labels arrive.
CREATE TABLE IF NOT EXISTS shiftbench.drift (
  detector STRING NOT NULL,
  source STRING NOT NULL,
  target STRING NOT NULL,
  psi FLOAT64 NOT NULL,
  mmd FLOAT64 NOT NULL,
  mmd_p_value FLOAT64 NOT NULL
)
CLUSTER BY detector, source, target;

-- The leaderboard: detectors ranked by whether the promise survives a change of domain.
CREATE TABLE IF NOT EXISTS shiftbench.leaderboard (
  detector STRING NOT NULL,
  pairs INT64 NOT NULL,
  mean_gap FLOAT64 NOT NULL,
  worst_gap FLOAT64 NOT NULL,
  promise_kept FLOAT64,
  mean_false_alarms FLOAT64 NOT NULL,
  labels_to_recover FLOAT64    -- null where no budget reached the tolerance
);
