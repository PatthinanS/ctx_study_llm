import re

from reccon_study.align import (
    DIALOG_RE,
    align_dialogue,
    align_one,
    containment_match,
    norm,
)


def test_norm_strips_non_alnum_lowercases():
    assert norm("Hello, World!  It's ME.") == "helloworlditsme"


def test_containment_match_accepts_short_fragment_via_length_guard():
    # "beatingaroundthebush" (20 chars) inside a 58-char string: ratio ~0.34
    # fails the plain 0.6 guard, but length >= 8 accepts it.
    short = norm("beating around the bush")
    long_ = norm("Stop beating around the bush and just tell me exactly what happened")
    assert len(short) / len(long_) <= 0.6
    assert containment_match(short, long_)
    assert containment_match(long_, short)


def test_containment_match_still_rejects_below_both_guards():
    short = norm("hi")  # contained, but < 8 chars and ratio far below 0.6
    long_ = norm("This is a very long sentence containing the word hi somewhere")
    assert short in long_
    assert not containment_match(short, long_)


def test_containment_match_rejects_no_overlap():
    assert not containment_match(norm("completely different"), norm("unrelated text entirely"))


def test_align_dialogue_matches_exact_and_skips_dropped_turns(synthetic_csv_rows, synthetic_reccon_turns):
    positions = align_dialogue(synthetic_reccon_turns, synthetic_csv_rows)
    assert positions == [0, 1, 3, 4, 6, 7]


def test_align_one_builds_visible_and_invisible_csv_pos(synthetic_csv_rows, synthetic_reccon_turns):
    record, unmatched = align_one("Session1", synthetic_reccon_turns, synthetic_csv_rows)
    assert record["matched"] == 6
    assert record["csv_n_turns"] == 8
    assert record["visible_csv_pos"] == [0, 1, 3, 4, 6, 7]
    assert record["invisible_csv_pos"] == [2, 5]
    assert set(record["visible_csv_pos"]) | set(record["invisible_csv_pos"]) == set(range(8))
    assert set(record["visible_csv_pos"]).isdisjoint(record["invisible_csv_pos"])
    assert unmatched == []


def test_cause_evidence_integer_resolves_to_csv_pos(fake_alignment):
    utterances = fake_alignment["Ses01F_impro01"]["utterances"]
    turn4 = next(u for u in utterances if u["reccon_turn"] == 4)
    assert turn4["csv_pos"] == 4
    assert turn4["cause_csv_pos"] == [3]
    assert turn4["cause_utterance_ids"] == ["Ses01F_impro01_M003"]
    assert turn4["cause_unresolved"] == 0
    assert turn4["max_cause_distance_csv"] == 1


def test_cause_evidence_pointing_at_missing_turn_increments_unresolved(fake_alignment):
    utterances = fake_alignment["Ses01F_impro01"]["utterances"]
    turn5 = next(u for u in utterances if u["reccon_turn"] == 5)
    assert turn5["cause_csv_pos"] == []
    assert turn5["cause_unresolved"] == 1
    assert turn5["has_latent_marker"] is False
    assert turn5["max_cause_distance_csv"] == -1


def test_cause_evidence_non_integer_sets_latent_marker_not_unresolved(fake_alignment):
    utterances = fake_alignment["Ses01F_impro01"]["utterances"]
    turn6 = next(u for u in utterances if u["reccon_turn"] == 6)
    assert turn6["has_latent_marker"] is True
    assert turn6["cause_unresolved"] == 0
    assert turn6["cause_csv_pos"] == []


def test_dialog_re_captures_ab_suffix():
    m = DIALOG_RE.search("dia_Ses03M_impro08b_utt_5")
    assert m is not None
    assert m.group(1) == "Ses03M_impro08b"


def test_dialog_re_captures_plain_dialog_name():
    m = re.search(DIALOG_RE, "dia_Ses01F_impro01_utt_0")
    assert m.group(1) == "Ses01F_impro01"
