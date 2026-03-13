#!/usr/bin/env python3
"""
friends_annotation_pipeline.py

Deterministic speaker-annotation pipeline for Friends transcript TSV files.

Supported commands:
    python friends_annotation_pipeline.py process-season --input-zip s1.zip --output-dir out/ --season-label s1
    python friends_annotation_pipeline.py process-all --input-dir raw/ --output-dir processed/
    python friends_annotation_pipeline.py global-qa --enhanced-zips processed/s1_enhanced_outputs.zip processed/s2_enhanced_outputs.zip --output-dir final/

Input TSV columns expected:
    scene_id, sentence_id, utterance, speaker, utterance_ct, speaker_ct

Output schema:
    scene_id
    sentence_id
    utterance
    speaker
    speaker_confidence
    speaker_method
    alignment_score
    row_type
    filled_from_missing
    matched_to_ct
    scene_speaker_set
    prev_speaker_scene
    next_speaker_scene
    review_flag
    review_reason
    notes
    utterance_ct
    speaker_ct
    speaker_original
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


FINAL_COLUMNS = [
    "scene_id",
    "sentence_id",
    "utterance",
    "speaker",
    "speaker_confidence",
    "speaker_method",
    "alignment_score",
    "row_type",
    "filled_from_missing",
    "matched_to_ct",
    "scene_speaker_set",
    "prev_speaker_scene",
    "next_speaker_scene",
    "review_flag",
    "review_reason",
    "notes",
    "utterance_ct",
    "speaker_ct",
    "speaker_original",
]


@dataclass
class PipelineConfig:
    song_markers: list[str]
    continuation_prefixes: list[str]
    high_bridge_score: float
    ct_match_score: float
    direction_score: float
    name_address_score: float
    short_turn_score: float
    scene_context_score: float
    long_ambiguous_score: float
    weak_ambiguous_score: float
    scene_majority_score: float

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        data = yaml.safe_load(path.read_text())
        return cls(
            song_markers=data["rules"]["song_markers"],
            continuation_prefixes=data["rules"]["continuation_prefixes"],
            high_bridge_score=float(data["scores"]["high_bridge"]),
            ct_match_score=float(data["scores"]["ct_match"]),
            direction_score=float(data["scores"]["direction"]),
            name_address_score=float(data["scores"]["name_address"]),
            short_turn_score=float(data["scores"]["short_turn"]),
            scene_context_score=float(data["scores"]["scene_context"]),
            long_ambiguous_score=float(data["scores"]["long_ambiguous"]),
            weak_ambiguous_score=float(data["scores"]["weak_ambiguous"]),
            scene_majority_score=float(data["scores"]["scene_majority"]),
        )


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", clean_str(text)))


def normalize_text(text: str) -> str:
    t = clean_str(text).lower()
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", t)
    t = t.replace("’", "'").replace("`", "'")
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def row_type_from_text(text: str, cfg: PipelineConfig) -> str:
    t = clean_str(text)
    tl = t.lower()
    if t == "":
        return "empty"
    if re.fullmatch(r"\[.*\]|\(.*\)", t):
        return "direction"
    if any(marker in tl for marker in cfg.song_markers):
        return "song"
    if re.fullmatch(r"(la+|ha+|o+h+|ah+|uh+|um+|hmm+)[.!?]*", tl):
        return "reaction"
    return "dialogue"


def nearest_known_speaker(df: pd.DataFrame, idx: int, direction: str) -> tuple[str, int | None]:
    scene = df.at[idx, "scene_id"]
    step = -1 if direction == "prev" else 1
    j = idx + step
    while 0 <= j < len(df) and df.at[j, "scene_id"] == scene:
        sp = clean_str(df.at[j, "speaker"])
        if sp:
            return sp, j
        j += step
    return "", None


def name_mentions(text: str, candidates: Iterable[str]) -> set[str]:
    tl = " " + normalize_text(text) + " "
    hits: set[str] = set()
    for c in candidates:
        cs = clean_str(c)
        if cs and (" " + cs.lower() + " ") in tl:
            hits.add(cs)
    return hits


def infer_speaker(df: pd.DataFrame, idx: int, cfg: PipelineConfig) -> tuple[str, str, str, float, bool, str, str]:
    text = clean_str(df.at[idx, "utterance"])
    rtype = row_type_from_text(text, cfg)
    prev_sp, _ = nearest_known_speaker(df, idx, "prev")
    next_sp, _ = nearest_known_speaker(df, idx, "next")
    scene = df.at[idx, "scene_id"]
    scene_candidates = [
        s for s in df.loc[df["scene_id"] == scene, "speaker"].astype(str).str.strip().tolist() if s
    ]
    scene_candidates = list(dict.fromkeys(scene_candidates))
    wc = word_count(text)
    mentions = name_mentions(text, scene_candidates)
    norm = normalize_text(text)

    if rtype == "song":
        return "Song", "non_dialogue_rule", "high", cfg.ct_match_score - 0.02, False, "", \
            "Marked as song lyric rather than scene dialogue."
    if rtype == "direction":
        return "", "direction", "high", cfg.direction_score, False, "", \
            "Stage direction or sound cue kept without speaker."

    if prev_sp and next_sp and prev_sp == next_sp:
        return prev_sp, "scene_context_bridge", "high", cfg.high_bridge_score, False, "", \
            "Same speaker appears on both sides within the same scene."

    if prev_sp and next_sp and prev_sp != next_sp:
        if prev_sp in mentions and next_sp not in mentions:
            return next_sp, "name_address_rule", "medium", cfg.name_address_score, False, "", \
                "Utterance names the previous speaker, so the next speaker is more likely."
        if next_sp in mentions and prev_sp not in mentions:
            return prev_sp, "name_address_rule", "medium", cfg.name_address_score, False, "", \
                "Utterance names the next speaker, so the previous speaker is more likely."

    if wc <= 2 and prev_sp and next_sp and prev_sp != next_sp:
        return next_sp, "short_turn_alternation", "medium", cfg.short_turn_score, True, \
            "short_utterance_between_two_speakers", \
            "Short utterance inferred from local turn alternation."

    if prev_sp and any(norm.startswith(x) for x in cfg.continuation_prefixes):
        return prev_sp, "scene_context_inference", "medium", cfg.scene_context_score + 0.03, False, "", \
            "Continuation-style opener favors previous speaker within the same scene."

    if prev_sp and not next_sp:
        return prev_sp, "scene_context_inference", "medium", cfg.scene_context_score, wc <= 2, \
            "scene_end_short_utterance" if wc <= 2 else "", \
            "Filled from previous same-scene context."

    if next_sp and not prev_sp:
        return next_sp, "scene_context_inference", "medium", cfg.scene_context_score, wc <= 2, \
            "scene_start_short_utterance" if wc <= 2 else "", \
            "Filled from next same-scene context."

    if prev_sp and next_sp:
        if wc >= 5:
            return prev_sp, "scene_context_inference", "medium", cfg.long_ambiguous_score, True, \
                "two_sided_context_ambiguous", \
                "No direct transcript match; longer line defaulted to previous speaker in same scene."
        return next_sp, "scene_context_inference", "low", cfg.weak_ambiguous_score, True, \
            "two_sided_context_ambiguous", \
            "No direct transcript match; weak local context only."

    speaker_counts = (
        df.loc[(df["scene_id"] == scene) & (df["speaker"].astype(str).str.strip() != ""), "speaker"]
        .astype(str)
        .value_counts()
    )
    if len(speaker_counts) > 0:
        cand = speaker_counts.index[0]
        return cand, "scene_majority_fallback", "low", cfg.scene_majority_score, True, \
            "scene_majority_fallback", \
            "Fallback to most common known speaker in the scene."

    return "", "unresolved", "low", 0.0, True, "no_scene_candidates", \
        "No reliable inference available."


def recompute_scene_context(df: pd.DataFrame) -> pd.DataFrame:
    scene_speaker_set = (
        df.groupby("scene_id")["speaker"]
        .apply(lambda s: " | ".join(dict.fromkeys([x for x in s.astype(str).str.strip().tolist() if x])))
        .to_dict()
    )

    prev_scene, next_scene = [], []
    for i in range(len(df)):
        scene = df.at[i, "scene_id"]
        prev_sp = ""
        next_sp = ""
        if i > 0 and df.at[i - 1, "scene_id"] == scene:
            prev_sp = clean_str(df.at[i - 1, "speaker"])
        if i < len(df) - 1 and df.at[i + 1, "scene_id"] == scene:
            next_sp = clean_str(df.at[i + 1, "speaker"])
        prev_scene.append(prev_sp)
        next_scene.append(next_sp)

    df["scene_speaker_set"] = df["scene_id"].map(scene_speaker_set)
    df["prev_speaker_scene"] = prev_scene
    df["next_speaker_scene"] = next_scene
    return df


def ensure_required_input_columns(df: pd.DataFrame, filename: str) -> None:
    required = ["scene_id", "sentence_id", "utterance", "speaker", "utterance_ct", "speaker_ct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename}: missing required columns: {missing}")


def process_tsv(path: Path, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, sep="\t")
    ensure_required_input_columns(df, path.name)

    df["speaker"] = df["speaker"].apply(clean_str)
    df["speaker_ct"] = df["speaker_ct"].apply(clean_str)
    df["utterance"] = df["utterance"].apply(clean_str)
    df["utterance_ct"] = df["utterance_ct"].apply(clean_str)

    original_missing = (df["speaker"] == "")
    df["speaker_original"] = df["speaker"]
    df["row_type"] = df["utterance"].apply(lambda x: row_type_from_text(x, cfg))
    df["matched_to_ct"] = df["speaker_ct"] != ""
    df["filled_from_missing"] = False
    df["speaker_method"] = ""
    df["speaker_confidence"] = ""
    df["alignment_score"] = 0.0
    df["review_flag"] = False
    df["review_reason"] = ""
    df["notes"] = ""

    direct_mask = (df["speaker"] == "") & (df["speaker_ct"] != "")
    df.loc[direct_mask, "speaker"] = df.loc[direct_mask, "speaker_ct"]
    df.loc[direct_mask, "filled_from_missing"] = True
    df.loc[direct_mask, "speaker_method"] = "community_transcript_match"
    df.loc[direct_mask, "speaker_confidence"] = "high"
    df.loc[direct_mask, "alignment_score"] = cfg.ct_match_score
    df.loc[direct_mask, "notes"] = "Speaker copied from available community-transcript mapping."

    supported_mask = (df["speaker"] != "") & (df["speaker_ct"] != "") & (df["speaker"] == df["speaker_ct"])
    df.loc[supported_mask & (df["speaker_method"] == ""), "speaker_method"] = "community_transcript_match"
    df.loc[supported_mask & (df["speaker_confidence"] == ""), "speaker_confidence"] = "high"
    df.loc[supported_mask & (df["alignment_score"] == 0), "alignment_score"] = cfg.ct_match_score
    df.loc[supported_mask & (df["notes"] == ""), "notes"] = "Speaker agrees with available community-transcript mapping."

    for _ in range(3):
        unresolved = df.index[(df["speaker"] == "") & (df["row_type"] != "direction")].tolist()
        if not unresolved:
            break
        changed = False
        for idx in unresolved:
            sp, method, conf, score, review, reason, note = infer_speaker(df, idx, cfg)
            if sp != "":
                df.at[idx, "speaker"] = sp
                df.at[idx, "filled_from_missing"] = True
                df.at[idx, "speaker_method"] = method
                df.at[idx, "speaker_confidence"] = conf
                df.at[idx, "alignment_score"] = score
                df.at[idx, "review_flag"] = bool(review)
                df.at[idx, "review_reason"] = reason
                df.at[idx, "notes"] = note
                changed = True
        if not changed:
            break

    direction_mask = (df["row_type"] == "direction") & (df["speaker_method"] == "")
    df.loc[direction_mask, "speaker_method"] = "direction"
    df.loc[direction_mask, "speaker_confidence"] = "high"
    df.loc[direction_mask, "alignment_score"] = cfg.direction_score
    df.loc[direction_mask, "notes"] = "Stage direction or sound cue kept without speaker."

    unresolved_mask = (df["speaker"] == "") & (df["speaker_method"] == "")
    df.loc[unresolved_mask, "speaker_method"] = "unresolved"
    df.loc[unresolved_mask, "speaker_confidence"] = "low"
    df.loc[unresolved_mask, "alignment_score"] = 0.0
    df.loc[unresolved_mask, "review_flag"] = True
    df.loc[unresolved_mask, "review_reason"] = "unresolved_no_reliable_inference"
    df.loc[unresolved_mask, "notes"] = "No reliable automatic assignment available."

    df = recompute_scene_context(df)

    low_mask = df["filled_from_missing"] & (df["speaker_confidence"] == "low")
    df.loc[low_mask, "review_flag"] = True
    df.loc[low_mask & (df["review_reason"] == ""), "review_reason"] = "low_confidence_fill"

    medium_mask = (
        df["filled_from_missing"]
        & (df["speaker_confidence"] == "medium")
        & df["speaker_method"].isin(["short_turn_alternation", "scene_context_inference"])
    )
    df.loc[medium_mask & (df["review_reason"] == ""), "review_flag"] = True
    df.loc[medium_mask & (df["review_reason"] == ""), "review_reason"] = "medium_confidence_context_fill"

    df = df[FINAL_COLUMNS]

    summary = {
        "file": path.name,
        "rows": len(df),
        "original_missing_speaker_rows": int(original_missing.sum()),
        "filled_rows": int(df["filled_from_missing"].sum()),
        "review_rows": int(df["review_flag"].sum()),
        "unresolved_rows": int((df["speaker"] == "").sum()),
        "high_confidence_rows": int((df["speaker_confidence"] == "high").sum()),
        "medium_confidence_rows": int((df["speaker_confidence"] == "medium").sum()),
        "low_confidence_rows": int((df["speaker_confidence"] == "low").sum()),
    }
    return df, summary


def zip_dir(zip_path: Path, files: list[Path], arcname_fn=None) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = arcname_fn(f) if arcname_fn else f.name
            zf.write(f, arcname=arcname)


def process_season(input_zip: Path, output_dir: Path, season_label: str, cfg: PipelineConfig) -> None:
    season_work = output_dir / season_label
    input_dir = season_work / "input"
    enhanced_dir = season_work / "enhanced"
    review_dir = season_work / "review"
    for d in [season_work, input_dir, enhanced_dir, review_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_zip, "r") as zf:
        zf.extractall(input_dir)

    tsv_files = sorted(input_dir.rglob("*.tsv"))
    if not tsv_files:
        raise ValueError(f"No TSV files found in {input_zip}")

    summary_rows = []
    for tsv in tsv_files:
        df, summary = process_tsv(tsv, cfg)
        enhanced_path = enhanced_dir / tsv.name.replace(".tsv", "_enhanced.tsv")
        review_path = review_dir / tsv.name.replace(".tsv", "_review.tsv")
        df.to_csv(enhanced_path, sep="\t", index=False)
        df[df["review_flag"]].to_csv(review_path, sep="\t", index=False)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values("file")
    summary_path = season_work / f"{season_label}_annotation_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    enhanced_zip = output_dir / f"{season_label}_enhanced_outputs.zip"
    review_zip = output_dir / f"{season_label}_review_outputs.zip"
    zip_dir(enhanced_zip, [summary_path] + sorted(enhanced_dir.glob("*.tsv")))
    zip_dir(review_zip, sorted(review_dir.glob("*.tsv")))

    print(f"[OK] {season_label}: {enhanced_zip}")
    print(f"[OK] {season_label}: {review_zip}")
    print(f"[OK] {season_label}: {summary_path}")


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    for c in FINAL_COLUMNS:
        if c not in df.columns:
            if c in {"filled_from_missing", "matched_to_ct", "review_flag"}:
                df[c] = False
            elif c == "alignment_score":
                df[c] = 0.0
            else:
                df[c] = ""
    return df[FINAL_COLUMNS].copy()


def append_note(old: str, new: str) -> str:
    old = clean_str(old)
    if not old:
        return new
    if new in old:
        return old
    return old + " | " + new


def scene_neighbors(df: pd.DataFrame, idx: int) -> tuple[str, int | None, str, int | None]:
    scene = df.at[idx, "scene_id"]
    prev_sp = next_sp = ""
    prev_idx = next_idx = None

    j = idx - 1
    while j >= 0 and df.at[j, "scene_id"] == scene:
        sp = clean_str(df.at[j, "speaker"])
        if sp:
            prev_sp, prev_idx = sp, j
            break
        j -= 1

    j = idx + 1
    while j < len(df) and df.at[j, "scene_id"] == scene:
        sp = clean_str(df.at[j, "speaker"])
        if sp:
            next_sp, next_idx = sp, j
            break
        j += 1

    return prev_sp, prev_idx, next_sp, next_idx


def apply_change(df: pd.DataFrame, idx: int, new_speaker: str, method: str, confidence: str, score: float,
                 reason: str, season: str, fname: str, qa_changes: list[dict]) -> None:
    old_speaker = clean_str(df.at[idx, "speaker"])
    if old_speaker == new_speaker:
        return
    qa_changes.append({
        "season": season,
        "file": fname,
        "scene_id": df.at[idx, "scene_id"],
        "sentence_id": df.at[idx, "sentence_id"],
        "utterance": clean_str(df.at[idx, "utterance"]),
        "old_speaker": old_speaker,
        "new_speaker": new_speaker,
        "old_method": clean_str(df.at[idx, "speaker_method"]),
        "new_method": method,
        "qa_reason": reason,
        "speaker_ct": clean_str(df.at[idx, "speaker_ct"]),
    })
    df.at[idx, "speaker"] = new_speaker
    df.at[idx, "speaker_method"] = method
    df.at[idx, "speaker_confidence"] = confidence
    df.at[idx, "alignment_score"] = max(float(df.at[idx, "alignment_score"]), score)
    if confidence == "high":
        df.at[idx, "review_flag"] = False
        df.at[idx, "review_reason"] = ""
    df.at[idx, "notes"] = append_note(df.at[idx, "notes"], f"QA correction: {reason}")


def global_qa(enhanced_zips: list[Path], output_dir: Path) -> None:
    qa_work = output_dir / "global_qa_work"
    input_dir = qa_work / "input"
    final_dir = qa_work / "final_cleaned"
    report_dir = qa_work / "reports"
    for d in [qa_work, input_dir, final_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    season_files: list[tuple[str, Path]] = []
    for zpath in enhanced_zips:
        label = zpath.stem.replace("_enhanced_outputs", "")
        season_dir = input_dir / label
        season_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(season_dir)
        for f in sorted(season_dir.glob("*.tsv")):
            if "summary" in f.name.lower():
                continue
            season_files.append((label, f))

    qa_changes: list[dict] = []
    summary_rows: list[dict] = []

    for season_label, fpath in season_files:
        df = pd.read_csv(fpath, sep="\t")
        df = ensure_schema(df)

        for c in ["utterance", "speaker", "speaker_confidence", "speaker_method", "row_type",
                  "scene_speaker_set", "prev_speaker_scene", "next_speaker_scene",
                  "review_reason", "notes", "utterance_ct", "speaker_ct", "speaker_original"]:
            df[c] = df[c].apply(clean_str)
        for c in ["filled_from_missing", "matched_to_ct", "review_flag"]:
            df[c] = df[c].fillna(False).astype(bool)
        df["alignment_score"] = pd.to_numeric(df["alignment_score"], errors="coerce").fillna(0.0)
        empty_rt = df["row_type"] == ""
        df.loc[empty_rt, "row_type"] = df.loc[empty_rt, "utterance"].apply(
            lambda x: row_type_from_text(x, PipelineConfig.from_yaml(DEFAULT_CONFIG_PATH))
        )

        orig_review = int(df["review_flag"].sum())
        ct_override_count = 0
        bridge_fix_count = 0
        anomaly_fix_count = 0
        review_fix_count = 0

        # QA 1: speaker_ct overrides where weak/conflicted
        for idx in df.index:
            sp = clean_str(df.at[idx, "speaker"])
            sp_ct = clean_str(df.at[idx, "speaker_ct"])
            if sp and sp_ct and sp != sp_ct:
                if clean_str(df.at[idx, "speaker_method"]) != "community_transcript_match" or \
                   df.at[idx, "review_flag"] or clean_str(df.at[idx, "speaker_confidence"]) in {"low", "medium"}:
                    apply_change(
                        df, idx, sp_ct, "qa_speaker_ct_override", "high", 1.0,
                        "speaker_ct mismatch override", season_label, fpath.name, qa_changes
                    )
                    ct_override_count += 1

        # QA 2: bridge corrections
        for idx in df.index:
            sp = clean_str(df.at[idx, "speaker"])
            if not sp:
                continue
            prev_sp, _, next_sp, _ = scene_neighbors(df, idx)
            if prev_sp and next_sp and prev_sp == next_sp and sp != prev_sp:
                weak = clean_str(df.at[idx, "speaker_confidence"]) in {"low", "medium"} or df.at[idx, "review_flag"]
                shortish = word_count(df.at[idx, "utterance"]) <= 4
                no_ct = clean_str(df.at[idx, "speaker_ct"]) == ""
                if weak and shortish and no_ct:
                    apply_change(
                        df, idx, prev_sp, "qa_scene_bridge_override", "high", 0.9,
                        "surrounded by same speaker within scene", season_label, fpath.name, qa_changes
                    )
                    bridge_fix_count += 1

        # QA 3: scene anomalies
        for scene_id, sdf in df.groupby("scene_id", sort=False):
            scene_idx = sdf.index.tolist()
            scene_counts = sdf["speaker"].apply(clean_str).value_counts()
            ct_scene_speakers = {clean_str(x) for x in sdf["speaker_ct"].tolist() if clean_str(x)}
            for idx in scene_idx:
                sp = clean_str(df.at[idx, "speaker"])
                if not sp:
                    continue
                if scene_counts.get(sp, 0) != 1:
                    continue
                if clean_str(df.at[idx, "speaker_confidence"]) not in {"low", "medium"} and not df.at[idx, "review_flag"]:
                    continue
                prev_sp, _, next_sp, _ = scene_neighbors(df, idx)
                if prev_sp and next_sp and prev_sp == next_sp and prev_sp != sp and clean_str(df.at[idx, "speaker_ct"]) == "":
                    apply_change(
                        df, idx, prev_sp, "qa_scene_anomaly_override", "high", 0.9,
                        "one-off scene speaker anomaly corrected by local scene pattern",
                        season_label, fpath.name, qa_changes
                    )
                    anomaly_fix_count += 1
                    continue
                if ct_scene_speakers and sp not in ct_scene_speakers and prev_sp in ct_scene_speakers and next_sp == prev_sp and prev_sp:
                    apply_change(
                        df, idx, prev_sp, "qa_scene_anomaly_override", "high", 0.9,
                        "speaker absent from scene CT support and overridden by local scene evidence",
                        season_label, fpath.name, qa_changes
                    )
                    anomaly_fix_count += 1

        # QA 4: resolve review flags where evidence is now clear
        for idx in df.index:
            if not df.at[idx, "review_flag"]:
                continue
            sp = clean_str(df.at[idx, "speaker"])
            sp_ct = clean_str(df.at[idx, "speaker_ct"])
            wc = word_count(df.at[idx, "utterance"])
            prev_sp, _, next_sp, _ = scene_neighbors(df, idx)

            if sp and prev_sp and next_sp and prev_sp == next_sp == sp:
                df.at[idx, "review_flag"] = False
                df.at[idx, "review_reason"] = ""
                df.at[idx, "speaker_confidence"] = "high"
                df.at[idx, "speaker_method"] = "qa_review_resolved_bridge"
                df.at[idx, "alignment_score"] = max(float(df.at[idx, "alignment_score"]), 0.9)
                df.at[idx, "notes"] = append_note(df.at[idx, "notes"], "QA resolved prior review by same-speaker bridge.")
                review_fix_count += 1
                continue

            if sp and sp_ct and sp == sp_ct:
                df.at[idx, "review_flag"] = False
                df.at[idx, "review_reason"] = ""
                df.at[idx, "speaker_confidence"] = "high"
                df.at[idx, "speaker_method"] = "qa_review_resolved_ct_support"
                df.at[idx, "alignment_score"] = max(float(df.at[idx, "alignment_score"]), 1.0)
                df.at[idx, "notes"] = append_note(df.at[idx, "notes"], "QA resolved prior review with community-transcript support.")
                review_fix_count += 1
                continue

            if sp and not sp_ct and wc <= 2 and prev_sp and next_sp and prev_sp == next_sp == sp:
                df.at[idx, "review_flag"] = False
                df.at[idx, "review_reason"] = ""
                df.at[idx, "speaker_confidence"] = "high"
                df.at[idx, "speaker_method"] = "qa_review_resolved_short_bridge"
                df.at[idx, "alignment_score"] = max(float(df.at[idx, "alignment_score"]), 0.88)
                df.at[idx, "notes"] = append_note(df.at[idx, "notes"], "QA resolved short review row by local bridge pattern.")
                review_fix_count += 1

        df = recompute_scene_context(df)
        df = df[FINAL_COLUMNS]

        out_name = fpath.name.replace("_enhanced.tsv", "_final_cleaned.tsv")
        df.to_csv(final_dir / out_name, sep="\t", index=False)

        summary_rows.append({
            "season": season_label,
            "file": fpath.name,
            "rows": len(df),
            "review_rows_before_qa": orig_review,
            "review_rows_after_qa": int(df["review_flag"].sum()),
            "speaker_ct_overrides": ct_override_count,
            "scene_bridge_fixes": bridge_fix_count,
            "scene_anomaly_fixes": anomaly_fix_count,
            "review_rows_resolved": review_fix_count,
            "remaining_unresolved_rows": int((df["speaker"].astype(str).str.strip() == "").sum()),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["season", "file"])
    summary_path = report_dir / "global_qa_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    changes_df = pd.DataFrame(qa_changes)
    changes_path = report_dir / "global_qa_changes.tsv"
    changes_df.to_csv(changes_path, sep="\t", index=False)

    rollup = pd.DataFrame([{
        "files_processed": len(summary_df),
        "total_rows": int(summary_df["rows"].sum()),
        "review_rows_before_qa": int(summary_df["review_rows_before_qa"].sum()),
        "review_rows_after_qa": int(summary_df["review_rows_after_qa"].sum()),
        "speaker_ct_overrides": int(summary_df["speaker_ct_overrides"].sum()),
        "scene_bridge_fixes": int(summary_df["scene_bridge_fixes"].sum()),
        "scene_anomaly_fixes": int(summary_df["scene_anomaly_fixes"].sum()),
        "review_rows_resolved": int(summary_df["review_rows_resolved"].sum()),
        "remaining_unresolved_rows": int(summary_df["remaining_unresolved_rows"].sum()),
    }])
    rollup_path = report_dir / "global_qa_rollup.tsv"
    rollup.to_csv(rollup_path, sep="\t", index=False)

    final_zip = output_dir / "friends_final_cleaned_dataset.zip"
    reports_zip = output_dir / "friends_global_qa_reports.zip"
    zip_dir(final_zip, sorted(final_dir.glob("*.tsv")))
    zip_dir(reports_zip, sorted(report_dir.glob("*.tsv")))

    print(f"[OK] final cleaned dataset: {final_zip}")
    print(f"[OK] qa reports: {reports_zip}")
    print(f"[OK] qa rollup: {rollup_path}")


def process_all(input_dir: Path, output_dir: Path, cfg: PipelineConfig) -> None:
    season_zips = sorted(input_dir.glob("*.zip"))
    if not season_zips:
        raise ValueError(f"No zip files found in {input_dir}")
    for z in season_zips:
        season_label = z.stem
        process_season(z, output_dir, season_label, cfg)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("pipeline_config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Friends speaker-annotation pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to pipeline_config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("process-season", help="Process one season zip")
    p1.add_argument("--input-zip", type=Path, required=True)
    p1.add_argument("--output-dir", type=Path, required=True)
    p1.add_argument("--season-label", type=str, required=True)

    p2 = sub.add_parser("process-all", help="Process all season zips in a directory")
    p2.add_argument("--input-dir", type=Path, required=True)
    p2.add_argument("--output-dir", type=Path, required=True)

    p3 = sub.add_parser("global-qa", help="Run QA on enhanced output zips")
    p3.add_argument("--enhanced-zips", type=Path, nargs="+", required=True)
    p3.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    cfg = PipelineConfig.from_yaml(args.config)

    if args.command == "process-season":
        process_season(args.input_zip, args.output_dir, args.season_label, cfg)
    elif args.command == "process-all":
        process_all(args.input_dir, args.output_dir, cfg)
    elif args.command == "global-qa":
        global_qa(args.enhanced_zips, args.output_dir)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
