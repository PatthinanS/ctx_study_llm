"""Centralised per-target eligibility classification, shared by leakage.py and
selection_eval.py.

Every RECCON-aligned target utterance is classified into exactly one of five
buckets before any selection-quality metric touches it. Two of the five
(`no_in_pool_cause`, `forced_selection`) don't exist in a naive scoring setup
and silently inflate every downstream metric if left implicit:

- a target whose only *resolved* gold cause positions are self-referential
  (the cause turn IS the target) can never have its recall reach 1 no matter
  how good the selector is, since the pool is strictly prior turns -- scoring
  it anyway just adds noise pulling every selector's recall down by the same
  fixed, uninformative amount. (A target with only unresolved-reference or
  latent-only ("b") cause evidence and no resolved position at all gives us
  nothing to check against being in-pool -- it lands in `no_cause` instead,
  indistinguishable from having no annotation.)
- a target whose pool is so small that every strategy selects the entire pool
  (`pool_size <= k`) has trivially perfect recall for ANY selector, including
  random -- scoring it makes every selector look identically good and dilutes
  genuine differences between them.

Both are therefore excluded from `"scoreable"`, the only bucket any metric is
computed over.
"""
from __future__ import annotations

BUCKETS = ["not_in_preds", "no_cause", "no_in_pool_cause", "forced_selection", "scoreable"]


def has_annotation(target_align: dict) -> bool:
    """True iff RECCON resolved at least one cause POSITION for this target
    (cause_csv_pos non-empty). An unresolved integer reference or a latent-
    only ("b") marker with no resolved position gives us nothing to check
    against being in-pool or not -- it's indistinguishable from no
    annotation at all for scoring purposes, so it lands in "no_cause", not
    the separate "no_in_pool_cause" bucket (which is reserved for a target
    with a resolved-but-structurally-unreachable cause)."""
    return bool(target_align["cause_csv_pos"])


def in_pool_cause_positions(target_align: dict) -> list[int]:
    """Gold cause positions strictly before the target's own csv_pos -- the
    only ones a prior-turns-only pool could ever contain. Excludes self-
    referential causes (cause_csv_pos == target's own csv_pos) and any cause
    at or after the target (cause_csv_pos > csv_pos)."""
    target_pos = target_align["csv_pos"]
    return [c for c in target_align["cause_csv_pos"] if c < target_pos]


def in_pool_R(target_align: dict) -> int:
    """R: count of in-pool gold causes for this target."""
    return len(in_pool_cause_positions(target_align))


def target_cause_bucket(target_align: dict) -> str | None:
    """"no_cause" | "no_in_pool_cause" | None (has >=1 usable in-pool cause)."""
    if not has_annotation(target_align):
        return "no_cause"
    if not in_pool_cause_positions(target_align):
        return "no_in_pool_cause"
    return None


def is_forced_selection(pool_size: int, k: int) -> bool:
    """True iff every pool turn gets selected regardless of strategy
    (pool_size <= n_sel, where n_sel = min(k, pool_size) -- equivalently
    pool_size <= k), making recall trivially perfect for any selector."""
    n_sel = min(k, pool_size)
    return pool_size <= n_sel


def classify_eligibility(target_align: dict, in_preds: bool, pool_size: int, k: int) -> str:
    """Exactly one of BUCKETS, checked in order:
      1. "not_in_preds"      -- no record for this utterance_id in the run
      2. "no_cause"          -- no cause annotation at all
      3. "no_in_pool_cause"  -- annotated, but no usable in-pool cause
      4. "forced_selection"  -- pool_size <= k (recall trivially perfect)
      5. "scoreable"         -- none of the above
    """
    if not in_preds:
        return "not_in_preds"
    bucket = target_cause_bucket(target_align)
    if bucket is not None:
        return bucket
    if is_forced_selection(pool_size, k):
        return "forced_selection"
    return "scoreable"


def compute_funnel(
    targets: list[dict], in_preds_fn, pool_size_fn, k: int
) -> dict[str, int]:
    """Runs `classify_eligibility` over every target. `in_preds_fn`/
    `pool_size_fn` are callables `target_align -> bool`/`-> int`. Returns
    {"aligned": N, "not_in_preds": N, "no_cause": N, "no_in_pool_cause": N,
    "forced_selection": N, "scoreable": N} -- the five bucket counts always
    sum to "aligned" == len(targets)."""
    counts = {b: 0 for b in BUCKETS}
    for target_align in targets:
        bucket = classify_eligibility(
            target_align, in_preds_fn(target_align), pool_size_fn(target_align), k
        )
        counts[bucket] += 1
    counts["aligned"] = len(targets)
    return counts
