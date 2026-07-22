from reccon_study.eligibility import (
    BUCKETS,
    classify_eligibility,
    compute_funnel,
    has_annotation,
    in_pool_cause_positions,
    in_pool_R,
    is_forced_selection,
    target_cause_bucket,
)


def _target(csv_pos=10, cause_csv_pos=None, cause_unresolved=0, has_latent_marker=False):
    return {
        "utterance_id": f"u{csv_pos}",
        "csv_pos": csv_pos,
        "cause_csv_pos": cause_csv_pos or [],
        "cause_unresolved": cause_unresolved,
        "has_latent_marker": has_latent_marker,
    }


def test_bucket_not_in_preds_takes_priority_over_everything():
    t = _target(cause_csv_pos=[4])
    assert classify_eligibility(t, in_preds=False, pool_size=10, k=4) == "not_in_preds"


def test_bucket_no_cause_when_no_annotation_at_all():
    t = _target()
    assert has_annotation(t) is False
    assert target_cause_bucket(t) == "no_cause"
    assert classify_eligibility(t, in_preds=True, pool_size=10, k=4) == "no_cause"


def test_bucket_no_in_pool_cause_self_referential():
    t = _target(csv_pos=10, cause_csv_pos=[10])  # cause == self
    assert has_annotation(t) is True
    assert in_pool_cause_positions(t) == []
    assert target_cause_bucket(t) == "no_in_pool_cause"
    assert classify_eligibility(t, in_preds=True, pool_size=10, k=4) == "no_in_pool_cause"


def test_bucket_no_in_pool_cause_latent_only():
    t = _target(cause_csv_pos=[], has_latent_marker=True)
    assert target_cause_bucket(t) == "no_in_pool_cause"


def test_bucket_no_in_pool_cause_unresolved_only():
    t = _target(cause_csv_pos=[], cause_unresolved=1)
    assert target_cause_bucket(t) == "no_in_pool_cause"


def test_bucket_forced_selection_when_pool_le_k():
    t = _target(csv_pos=3, cause_csv_pos=[1])  # valid in-pool cause
    assert is_forced_selection(pool_size=3, k=4) is True
    assert classify_eligibility(t, in_preds=True, pool_size=3, k=4) == "forced_selection"


def test_bucket_scoreable():
    t = _target(csv_pos=10, cause_csv_pos=[4])
    assert is_forced_selection(pool_size=10, k=4) is False
    assert classify_eligibility(t, in_preds=True, pool_size=10, k=4) == "scoreable"


def test_pool_size_zero_is_forced_selection():
    # first utterance of a dialogue: pool_size=0 -> vacuously forced (nothing selectable)
    assert is_forced_selection(pool_size=0, k=4) is True


def test_in_pool_r_excludes_self_cause():
    t = _target(csv_pos=10, cause_csv_pos=[10, 4])  # self + one genuine prior cause
    assert in_pool_R(t) == 1


def test_in_pool_r_excludes_future_cause():
    t = _target(csv_pos=10, cause_csv_pos=[10, 15])  # self + a cause "after" the target
    assert in_pool_R(t) == 0


def test_compute_funnel_sums_to_aligned():
    targets = [
        _target(csv_pos=10, cause_csv_pos=[4]),        # scoreable (pool 10, k 4)
        _target(csv_pos=1, cause_csv_pos=[]),           # no_cause
        _target(csv_pos=5, cause_csv_pos=[5]),           # no_in_pool_cause (self)
        _target(csv_pos=2, cause_csv_pos=[1]),           # forced_selection (pool 2 <= k 4)
    ]
    pool_sizes = {targets[0]["utterance_id"]: 10, targets[1]["utterance_id"]: 10,
                  targets[2]["utterance_id"]: 10, targets[3]["utterance_id"]: 2}
    funnel = compute_funnel(
        targets,
        in_preds_fn=lambda t: True,
        pool_size_fn=lambda t: pool_sizes[t["utterance_id"]],
        k=4,
    )
    assert funnel["aligned"] == 4
    assert funnel["scoreable"] == 1
    assert funnel["no_cause"] == 1
    assert funnel["no_in_pool_cause"] == 1
    assert funnel["forced_selection"] == 1
    assert funnel["not_in_preds"] == 0
    assert sum(funnel[b] for b in BUCKETS) == funnel["aligned"]


def test_compute_funnel_not_in_preds():
    targets = [_target(csv_pos=10, cause_csv_pos=[4])]
    funnel = compute_funnel(targets, in_preds_fn=lambda t: False, pool_size_fn=lambda t: 10, k=4)
    assert funnel["not_in_preds"] == 1
    assert funnel["scoreable"] == 0
