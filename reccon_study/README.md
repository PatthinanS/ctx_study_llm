# `reccon_study/` — RECCON-IE cause-annotation analysis

Analysis-only package: joins RECCON's human emotion-cause annotations onto this
repo's IEMOCAP CSV, and evaluates C1/C2's context selections against the
resulting gold causes. It reads existing `outputs/*/preds.jsonl` and
`data/iemocap/iemocap_merged_all.csv`; it performs no inference and no
training, and never touches `src/run.py`.

## The reduced-gold-standard limitation

RECCON-IE's own transcript is a **filtered subset** of IEMOCAP: across the 16
dialogues it annotates, it has 665 turns vs. 1036 in this repo's CSV. It
dropped turns with no annotator-majority emotion label (`xxx`) plus rare
classes. RECCON's human annotators never saw the dropped turns, so they never
had the option to mark one as a cause.

This means: **every recall/coverage number this package reports is a lower
bound, not ground truth.** A selector that picks a dropped turn with genuine
causal content is indistinguishable, from RECCON's annotations alone, from one
that picked something irrelevant — naive recall would penalize both equally.
This is IR's "incomplete judgments" problem (Buckley, C. and Voorhees, E.M.,
2004, "Retrieval evaluation with incomplete information," *SIGIR*).

**Mitigation: three-way classification + condensed-list evaluation.** Every
turn a selector picks is classified as `gold_cause` (RECCON marked it as a
cause of the target), `visible_noncause` (RECCON saw it and didn't mark it as
a cause), or `invisible` (RECCON never saw it — dropped from its transcript,
unjudged). All of this package's precision/recall/F1 metrics are computed only
over the `gold_cause` + `visible_noncause` ("scoreable") subset — `invisible`
picks are dropped entirely rather than counted as errors. This is condensed-
list evaluation (Sakai, T. and Kando, N., 2008, "On information retrieval
metrics designed for evaluation with incomplete relevance assessments,"
*Information Retrieval* 11(5), 447–470), the standard treatment for exactly
this situation. `leakage.py`'s invisible-selection rate (with its bootstrap CI
against a random-pool-composition baseline) is the complementary check: it
tells you how often a selector reaches into the unjudged region at all, so you
can judge how much the missing-recall problem might actually be costing you.

There is **no ranked bpref** here. An earlier design used a bpref upper/lower
bound bracket (from the same Buckley & Voorhees 2004 paper), but with no rank
order surviving into `preds.jsonl` (`src/run.py` always stores
`selected_indices` pre-sorted chronologically ascending, regardless of
strategy), that bracket's upper bound reduces algebraically to plain recall
and its lower bound is that same recall times a constant — it carries no
information beyond what the condensed-list metric and the invisible-selection
rate already report separately. Buckley & Voorhees (2004) is still cited above
for the incomplete-judgments *problem* it names; Sakai & Kando (2008) is the
metric actually used here.

## Design decisions worth knowing before reading the code

- **`align.py` reuses `src.data.load_iemocap`/`reconstruct_dialogues`**,
  rather than an independent CSV load and sort. `src/run.py`'s C1/C2 pool
  construction groups and sorts the CSV the exact same way. Pandas'
  `sort_values` isn't stable by default; an independently-reimplemented sort
  could tie-break identically-timestamped rows differently, silently
  misaligning this package's `csv_pos` numbering against the pool positions a
  real run actually used. Sharing the function eliminates that risk instead
  of hoping ties don't occur.
- **Pool reconstruction is always CSV-derived**, in `leakage.py`/
  `selection_eval.py`, never arithmetic on a preds.jsonl record's own stored
  fields (except as a cross-check for C2). A target's pool is every prior turn
  of its dialogue — this is recomputed fresh from the CSV via a
  `utterance_id -> (session, dialog, idx_in_dialogue)` index, which is what
  lets C1 records (which store neither `selected_indices` nor `pool_size` at
  all) and C2 records (which store both) share the same code path.
- **Eligibility is centralized** in `eligibility.py`, consumed by both
  `leakage.py` and `selection_eval.py`. Every RECCON-aligned target is
  classified into exactly one of five buckets before any metric touches it:
  `not_in_preds` (no record for it in this run), `no_cause` (RECCON resolved
  no usable cause position at all — including a target whose only cause
  evidence is an unresolved reference or a latent-only ("b") marker, since
  neither gives a position to check against being in-pool), `no_in_pool_cause`
  (RECCON resolved a cause position, but every one is self-referential —
  unreachable, since the pool is strictly prior turns), `forced_selection`
  (`pool_size <= k`, so every
  strategy including random selects the entire pool — recall is trivially
  perfect and uninformative), and `scoreable` (none of the above). Only
  `scoreable` targets are scored. The `not_in_preds`/`no_cause`/
  `no_in_pool_cause`/`forced_selection`/`scoreable` funnel is printed first by
  both `leakage.py` and `selection_eval.py`, before any rate or percentage —
  the point is that no metric in this package's output should ever be read
  without knowing how many targets it's actually based on.
- **All aggregation is macro**: precision/recall/F1 (and the invisible rate)
  are computed per target, then averaged across targets, matching the
  standard IR convention (average per-query, then across queries) and this
  repo's own nan-on-empty convention (`src/metrics.py`'s `per_class_f1`/
  `macro_f1`). Every macro metric carries a 95% percentile bootstrap CI
  (10,000 resamples, seeded, resampled at the **target** level — a target's
  several selected turns aren't independent draws, so resampling individual
  selections would understate the true uncertainty).

## Scope: C0-C2 only; C1 is the recency floor

No IEMOCAP training happens anywhere in this repo, so no prompting-only
condition has a train/val/test contamination issue on any session — all 16
RECCON-annotated dialogues are valid input regardless of session. The
`--sessions` filter exists purely for a **future** fine-tuned condition (C3,
in the sibling PLM repo `ContextStudy_NewResearch`), which will need
`--sessions 4,5` when it exists; this package does no runtime contamination
detection today, it's just documented here.

C0 has no context and therefore nothing to select — it's out of scope for
this package entirely (it stays a pure downstream comparison point via
`src/score.py`). C1 records don't carry `selected_indices` (only C2's
`process_one_c2ab`/`process_one_c2c` do — `process_one`, C0/C1's path, doesn't),
so C1's selection is *derived*: the last `n_sel = min(k, pool_size)` pool
indices, i.e. plain recency. This makes C1 double as the recency floor for
`selection_eval.py` — pass a C1 run directory alongside C2 run directories in
`--runs` to get it scored identically and compared directly, rather than each
C2 run separately reconstructing a synthetic "last-k" baseline internally.

## Known limitations

`src/run.py` always chronologically re-sorts `selected_indices` before
writing them to `preds.jsonl`, for every C2 strategy — this destroys whatever
preference order a retriever's similarity ranking or an LLM judge's picks
originally had. A future `selected_indices_ranked` field on the C2 record
shapes could preserve that order for rank-aware analysis later. It isn't
needed by anything in this package (no ranked metric is used here — see
"no ranked bpref" above), so it's not implemented in this pass, and
`src/run.py` is left untouched.

## Commands

```bash
# 1. Build the alignment (place RECCON's iemocap_test.json at data/reccon/ first --
#    data/ is already gitignored, same manual-copy convention as the IEMOCAP CSV).
python -m reccon_study.align --reccon data/reccon/iemocap_test.json --csv data/iemocap/iemocap_merged_all.csv

# 2. Leakage: how often does each run's selector reach into unjudged (invisible) turns?
python -m reccon_study.leakage --runs outputs/c1_llama31 outputs/c2_random_llama31_test outputs/c2_sim_llama31_test outputs/c2_llm_llama31_test [--seed 42]

# 3. Selection quality against gold causes, C1 included as the recency floor.
python -m reccon_study.selection_eval --runs outputs/c1_llama31 outputs/c2_random_llama31_test outputs/c2_sim_llama31_test outputs/c2_llm_llama31_test [--sessions 4,5]
```

All three write JSON to `outputs/reccon/` and print a human-readable report to
stdout.

## Output artifacts

- **`outputs/reccon/reccon_ie_aligned.json`** — the per-dialogue turn
  alignment: RECCON turn ↔ CSV `csv_pos`/`utterance_id`, resolved cause links
  (`cause_csv_pos`/`cause_utterance_ids`), unresolved/latent cause-evidence
  counts, RECCON's cause-type tags, and each dialogue's `visible_csv_pos`/
  `invisible_csv_pos` sets (which CSV positions RECCON could and couldn't
  see). Everything downstream is built from this file.
- **`outputs/reccon/leakage.json`** — per run: the eligibility funnel, pooled
  three-way classification counts/rates of every selected turn, and the
  macro-averaged observed-invisible-rate vs. the random-pool-composition
  baseline rate, reported as a paired bootstrap difference with its CI (rather
  than two bare point rates) so "this selector's bias is smaller than chance"
  is a claim with an interval behind it, not an eyeball comparison.
- **`outputs/reccon/selection_eval.json`** — per run: the eligibility funnel,
  scoreable-subset precision/recall/F1 with bootstrap CIs (the headline
  condensed-list metric), the visible-pool counterfactual variant, and
  breakdowns by RECCON cause type and by cause-distance bucket (`<=k` vs.
  `>k`, the long-range-cause hypothesis this whole study exists to test), each
  block carrying its own `n` — Session 5 alone has only ~92-101 scoreable
  targets depending on scope, small enough that every report flags it rather
  than presenting a bare percentage.

## Testing

```bash
pytest                          # from repo root -- covers tests/ and reccon_study/tests/
pytest reccon_study/tests/      # scoped to this package only
```

The real-data regression tests (`reccon_study/tests/test_regression_real_data.py`)
skip automatically unless both `data/iemocap/iemocap_merged_all.csv` and
`data/reccon/iemocap_test.json` are present, mirroring
`tests/test_data.py`'s Session5-split-parity skip pattern.
