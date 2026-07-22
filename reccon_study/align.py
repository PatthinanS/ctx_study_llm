"""Align RECCON-IE annotations to the ctx_study IEMOCAP CSV.

RECCON-IE's own transcript is a REDUCED version of IEMOCAP (665 turns vs 1036
across the 16 dialogues it annotates -- it drops turns with no annotator-
majority emotion label plus rare classes). RECCON turn indices are therefore
NOT usable as ctx_study CSV positions directly. RECCON turns are a monotone
subsequence of the CSV rows for a given dialogue, so alignment is done by
forward-cursor normalised-text matching (never backwards, so repeated lines
can't cross-match).

Deliberately reuses `src.data.load_iemocap`/`reconstruct_dialogues` rather
than an independent CSV load+sort: `src/run.py`'s C1/C2 pool construction
groups/sorts the CSV the exact same way (group by (session,dialog), sort by
start_time) via those same functions. Re-deriving that ordering independently
here (e.g. via the stdlib csv module + `list.sort`) risks a different tie-
break order under identical start_time values than pandas' non-stable
`sort_values` produces -- silently misaligning this module's `csv_pos`
numbering against the pool positions `leakage.py`/`selection_eval.py` later
reconstruct from the same CSV. Sharing the function eliminates that risk
instead of hoping ties don't occur.

Output: outputs/reccon/reccon_ie_aligned.json
  {
    "<dialog>": {
      "session": str,                # "SessionN", from the CSV row itself
      "csv_n_turns": int,
      "reccon_n_turns": int,
      "matched": int,
      "visible_csv_pos": [int],      # csv_pos values RECCON could see
      "invisible_csv_pos": [int],    # csv_pos values RECCON never saw
      "utterances": [
        {
          "reccon_turn": int,
          "utterance_id": str,
          "csv_pos": int,
          "speaker": str,
          "emotion_reccon": str,
          "emotion_csv": str,
          "cause_csv_pos": [int],
          "cause_utterance_ids": [str],
          "cause_unresolved": int,
          "has_latent_marker": bool,
          "types": [str],
          "max_cause_distance_csv": int   # -1 if no cause links
        }, ...
      ]
    }, ...
  }
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from typing import Any

import pandas as pd

from reccon_study.cli import ALL_SESSIONS, write_report
from src.data import load_iemocap, reconstruct_dialogues

DIALOG_RE = re.compile(r"(Ses\d\d[FM]_[a-z]+\d+[ab]?(?:_\d+)?)")

DEFAULT_RECCON = "data/reccon/iemocap_test.json"
DEFAULT_CSV = "data/iemocap/iemocap_merged_all.csv"

RECENCY_KS = [1, 2, 3, 4, 5, 8, 10, 15]


def norm(text: str) -> str:
    """Aggressive normalisation for matching: lowercase, alnum only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def containment_match(a: str, b: str) -> bool:
    """a, b already normalised. True iff one contains the other AND (the
    length-ratio guard passes OR the shorter string is >=8 chars).

    The plain ratio guard (`min(len)/max(len) > 0.6`) rejects legitimate
    cases where a RECCON turn's text is a short fragment fully contained in a
    much longer CSV utterance (e.g. a RECCON turn reading "beating around the
    bush." matching within a longer CSV line). The length-only branch
    recovers those without opening the door to trivial garbage matches on
    tiny substrings (hence the >=8 char floor).
    """
    if not a or not b:
        return False
    if not (a in b or b in a):
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    ratio_ok = len(shorter) / len(longer) > 0.6
    length_ok = len(shorter) >= 8
    return ratio_ok or length_ok


def align_dialogue(reccon_turns: list[dict], csv_rows: list[dict]) -> list[int | None]:
    """Monotone-subsequence alignment of RECCON turns onto CSV rows.

    Forward-only cursor: exact normalised-text match first (scanned forward
    from the cursor), then `containment_match` as a fallback. Returns a list
    of csv_rows positions (or None if unmatched), parallel to reccon_turns.
    """
    csv_norm = [norm(r["text"]) for r in csv_rows]
    out: list[int | None] = []
    cursor = 0
    for t in reccon_turns:
        target = norm(t["utterance"])
        hit = None
        for j in range(cursor, len(csv_norm)):
            if csv_norm[j] == target:
                hit = j
                break
        if hit is None:
            for j in range(cursor, len(csv_norm)):
                if containment_match(csv_norm[j], target):
                    hit = j
                    break
        out.append(hit)
        if hit is not None:
            cursor = hit + 1
    return out


def build_dialog_index(dialogues: dict[tuple[str, Any], list[dict]]) -> dict[str, tuple[str, Any]]:
    """dialog name -> (session, dialog) key. A flat map is safe here: no
    dialog name collides across sessions in the real CSV (verified directly:
    151 unique dialog names, 0 spanning more than one session)."""
    return {dkey[1]: dkey for dkey in dialogues}


def _translate_cause_evidence(
    evidence: list, turn_to_pos: dict[int, int]
) -> tuple[list[int], int, bool]:
    """Integer entries -> resolved csv_pos via turn_to_pos (unresolved += 1 if
    the referenced RECCON turn itself failed to align). Non-integer entries
    (e.g. "b") -> has_latent_marker=True, not counted as unresolved."""
    cause_pos: list[int] = []
    unresolved = 0
    latent = False
    for e in evidence:
        if isinstance(e, int):
            if e in turn_to_pos:
                cause_pos.append(turn_to_pos[e])
            else:
                unresolved += 1
        else:
            latent = True
    return sorted(set(cause_pos)), unresolved, latent


def align_one(
    session: str, reccon_turns: list[dict], csv_rows: list[dict]
) -> tuple[dict, list[dict]]:
    """Builds one dialogue's output record. Returns (record, unmatched) where
    unmatched is a list of {"reccon_turn", "text_prefix"} for turns that
    failed to align even after the containment fallback."""
    positions = align_dialogue(reccon_turns, csv_rows)
    turn_to_pos = {t["turn"]: p for t, p in zip(reccon_turns, positions) if p is not None}

    utterances = []
    unmatched = []
    for t, p in zip(reccon_turns, positions):
        if p is None:
            unmatched.append({"reccon_turn": t["turn"], "text_prefix": t["utterance"][:40]})
            continue
        row = csv_rows[p]
        evidence = t.get("expanded emotion cause evidence") or []
        cause_pos, unresolved, latent = _translate_cause_evidence(evidence, turn_to_pos)
        cause_uids = [csv_rows[c]["utterance_id"] for c in cause_pos]
        dists = [p - c for c in cause_pos]
        utterances.append({
            "reccon_turn": t["turn"],
            "utterance_id": row["utterance_id"],
            "csv_pos": p,
            "speaker": row["speaker"],
            "emotion_reccon": t.get("emotion"),
            "emotion_csv": row["emotion"],
            "cause_csv_pos": cause_pos,
            "cause_utterance_ids": cause_uids,
            "cause_unresolved": unresolved,
            "has_latent_marker": latent,
            "types": t.get("type", []),
            "max_cause_distance_csv": max(dists) if dists else -1,
        })

    visible = sorted({u["csv_pos"] for u in utterances})
    invisible = sorted(set(range(len(csv_rows))) - set(visible))

    record = {
        "session": session,
        "csv_n_turns": len(csv_rows),
        "reccon_n_turns": len(reccon_turns),
        "matched": len(utterances),
        "visible_csv_pos": visible,
        "invisible_csv_pos": invisible,
        "utterances": utterances,
    }
    return record, unmatched


def compute_dropped_turn_emotion_dist(
    result: dict, dialogues: dict[tuple[str, Any], list[dict]], dialog_index: dict[str, tuple[str, Any]]
) -> dict[str, int]:
    """CSV `emotion` value counts among CSV rows in the aligned dialogues that
    were NOT matched to any RECCON turn (the `invisible_csv_pos` rows)."""
    dist: dict[str, int] = defaultdict(int)
    for dialog, record in result.items():
        csv_rows = dialogues[dialog_index[dialog]]
        for pos in record["invisible_csv_pos"]:
            emo = csv_rows[pos]["emotion"]
            key = "nan" if pd.isna(emo) else str(emo)
            dist[key] += 1
    return dict(dist)


def build_alignment(
    reccon_path: str, csv_path: str, sessions: list[str] | None = None
) -> tuple[dict, dict]:
    """Top-level: loads both inputs, aligns every RECCON-annotated dialogue,
    returns (result, stats). `result` is exactly the JSON schema written to
    disk; `stats` is aggregate report data (not written) for `print_report`.
    """
    if sessions is None:
        sessions = ALL_SESSIONS
    with open(reccon_path, encoding="utf-8") as f:
        reccon = json.load(f)
    df = load_iemocap(csv_path)
    dialogues = reconstruct_dialogues(df, sessions=sessions)
    dialog_index = build_dialog_index(dialogues)

    result: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    tot_utt = tot_matched = tot_ann = tot_cause_links = tot_unresolved = 0
    label_agree = label_disagree = 0
    disagree_examples: list[dict] = []

    for key, val in reccon.items():
        m = DIALOG_RE.search(key)
        if m is None:
            failures.append((key, "dialog name not parseable from RECCON key"))
            continue
        dialog = m.group(1)
        dkey = dialog_index.get(dialog)
        if dkey is None:
            failures.append((dialog, "dialog not present in CSV"))
            continue
        session, _ = dkey
        csv_rows = dialogues[dkey]
        reccon_turns = val[0]

        record, unmatched = align_one(session, reccon_turns, csv_rows)
        result[dialog] = record

        tot_utt += len(reccon_turns)
        tot_matched += record["matched"]
        for u in unmatched:
            failures.append((dialog, f"turn {u['reccon_turn']}: {u['text_prefix']!r}"))
        for u in record["utterances"]:
            if u["cause_csv_pos"] or u["cause_unresolved"] or u["has_latent_marker"]:
                tot_ann += 1
            tot_cause_links += len(u["cause_csv_pos"])
            tot_unresolved += u["cause_unresolved"]
            if u["emotion_reccon"] == u["emotion_csv"]:
                label_agree += 1
            else:
                label_disagree += 1
                if len(disagree_examples) < 15:
                    disagree_examples.append({
                        "dialog": dialog, "utterance_id": u["utterance_id"],
                        "emotion_reccon": u["emotion_reccon"], "emotion_csv": u["emotion_csv"],
                    })

    per_session: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for dialog, record in result.items():
        s = record["session"]
        per_session[s][0] += 1
        per_session[s][1] += len(record["utterances"])
        per_session[s][2] += sum(1 for u in record["utterances"] if u["cause_csv_pos"])

    dists: list[int] = []
    fullcov: list[int] = []
    for record in result.values():
        for u in record["utterances"]:
            if not u["cause_csv_pos"]:
                continue
            ds = [u["csv_pos"] - c for c in u["cause_csv_pos"] if u["csv_pos"] - c > 0]
            if not ds:
                continue
            dists.extend(ds)
            fullcov.append(max(ds))

    stats = {
        "n_dialogues": len(result),
        "tot_utt": tot_utt,
        "tot_matched": tot_matched,
        "tot_ann": tot_ann,
        "tot_cause_links": tot_cause_links,
        "tot_unresolved": tot_unresolved,
        "label_agree": label_agree,
        "label_disagree": label_disagree,
        "disagree_examples": disagree_examples,
        "dropped_emotion_dist": compute_dropped_turn_emotion_dist(result, dialogues, dialog_index),
        "per_session": {s: {"n_dialogues": v[0], "n_utts": v[1], "n_cause_annotated": v[2]}
                         for s, v in sorted(per_session.items())},
        "recency_dists": dists,
        "recency_fullcov": fullcov,
        "failures": failures,
    }
    return result, stats


def print_report(result: dict, stats: dict) -> None:
    print("=" * 60)
    print(f"dialogues aligned  : {stats['n_dialogues']}")
    print(f"RECCON utterances  : {stats['tot_utt']}")
    rate = 100 * stats["tot_matched"] / stats["tot_utt"] if stats["tot_utt"] else 0.0
    print(f"matched to CSV     : {stats['tot_matched']} ({rate:.1f}%)")
    print(f"with cause annot.  : {stats['tot_ann']}")
    print(f"cause links mapped : {stats['tot_cause_links']}  (unresolved: {stats['tot_unresolved']})")
    n_labelled = stats["label_agree"] + stats["label_disagree"]
    agree_rate = 100 * stats["label_agree"] / n_labelled if n_labelled else 0.0
    print(f"label agreement    : {agree_rate:.1f}% ({stats['label_agree']}/{n_labelled})")
    if stats["disagree_examples"]:
        print("  disagreement examples (up to 15):")
        for ex in stats["disagree_examples"]:
            print(f"    {ex['dialog']} {ex['utterance_id']}: "
                  f"reccon={ex['emotion_reccon']!r} csv={ex['emotion_csv']!r}")
    print()
    print("dropped-turn emotion distribution (CSV rows RECCON never saw):")
    for emo, count in sorted(stats["dropped_emotion_dist"].items(), key=lambda kv: -kv[1]):
        print(f"  {emo:>6}: {count}")
    print()
    print("per session:")
    for s, v in stats["per_session"].items():
        print(f"  {s}: {v['n_dialogues']} dialogues, {v['n_utts']} aligned utts, "
              f"{v['n_cause_annotated']} with mapped causes")
    print()
    dists, fullcov = stats["recency_dists"], stats["recency_fullcov"]
    if dists:
        print("cause distance in CSV turn space (self-causes excluded):")
        for k in RECENCY_KS:
            a = 100 * sum(1 for x in dists if x <= k) / len(dists)
            b = 100 * sum(1 for x in fullcov if x <= k) / len(fullcov)
            print(f"  k={k:>2}: {a:5.1f}% of cause links | {b:5.1f}% of utts fully covered")
        print(f"  max distance: {max(dists)}   n_links={len(dists)}  n_utts={len(fullcov)}")
    print("=" * 60)
    if stats["failures"]:
        print(f"\nunmatched ({len(stats['failures'])}), first 15:")
        for f in stats["failures"][:15]:
            print("  ", f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reccon", default=DEFAULT_RECCON)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()

    result, stats = build_alignment(args.reccon, args.csv)
    print_report(result, stats)
    path = write_report("reccon_ie_aligned", result)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
