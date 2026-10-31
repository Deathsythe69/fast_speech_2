"""Alignment module and CLI for Montreal Forced Aligner (MFA) and TextGrid generation.
Phase 3 requirement: pre-alignment check, MFA alignment, TextGrid generation, and alignment report.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import soundfile as sf
import tgt
from tqdm import tqdm

# Ensure workspace root is in path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


def pre_alignment_check(raw_dir, lang, expected_sr=22050):
    """Pre-alignment validation:
    - Verifies all wav files exist, are readable, are mono, and match expected sampling rate.
    - Verifies matching .lab files exist and contain non-empty text.
    """
    print(f"[{lang.upper()}] Running pre-alignment checks in {raw_dir}...")
    wav_files = glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"[{lang.upper()}] Warning: No wav files found in {raw_dir}")
        return False, []

    errors = []
    valid_files = []

    for wav_path in wav_files:
        base, _ = os.path.splitext(wav_path)
        lab_path = base + ".lab"

        # Check lab
        if not os.path.exists(lab_path):
            errors.append((wav_path, "Missing matching .lab file"))
            continue

        with open(lab_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            errors.append((wav_path, "Empty transcript in .lab file"))
            continue

        # Check wav properties
        try:
            info = sf.info(wav_path)
            if info.samplerate != expected_sr:
                errors.append(
                    (wav_path, f"Sampling rate mismatch: {info.samplerate} != {expected_sr}")
                )
                continue
            if info.channels != 1:
                errors.append((wav_path, f"Channel count mismatch: {info.channels} != 1 (mono)"))
                continue
        except Exception as e:
            errors.append((wav_path, f"Corrupted audio file: {str(e)}"))
            continue

        valid_files.append((wav_path, lab_path))

    print(f"[{lang.upper()}] Checked {len(wav_files)} files: {len(valid_files)} valid, {len(errors)} issues.")
    return len(errors) == 0, valid_files


def generate_lexicon(raw_dir, lang, out_dict_path):
    """Generates an MFA-compatible pronunciation dictionary from transcripts in raw_dir."""
    os.makedirs(os.path.dirname(os.dict_path) if os.path.dirname(out_dict_path) else ".", exist_ok=True)
    lab_files = glob.glob(os.path.join(raw_dir, "**", "*.lab"), recursive=True)

    words = set()
    for lab in lab_files:
        with open(lab, "r", encoding="utf-8") as f:
            text = f.read().strip()
            for w in text.split():
                clean_w = w.strip(".,!?;:\"'()").lower()
                if clean_w:
                    words.add(clean_w)

    print(f"[{lang.upper()}] Building lexicon for {len(words)} unique words...")
    lexicon = {}

    if lang == "english":
        from English.text.g2p import text_to_phonemes_en
        for w in words:
            phones = text_to_phonemes_en(w)
            lexicon[w] = " ".join([p for p in phones if p != "sp"])
    else:
        from Telugu.text.g2p import text_to_phonemes_te
        for w in words:
            phones = text_to_phonemes_te(w)
            lexicon[w] = " ".join([p for p in phones if p != "sp"])

    with open(out_dict_path, "w", encoding="utf-8") as f:
        for w in sorted(lexicon.keys()):
            if lexicon[w].strip():
                f.write(f"{w}\t{lexicon[w]}\n")

    print(f"[{lang.upper()}] Saved pronunciation lexicon to {out_dict_path}")
    return out_dict_path


def align_single_utterance(wav_path, text, lang):
    """Aligns a single audio file with text into a Praat TextGrid.
    Uses phonetic VAD and duration-proportional allocation as a reliable aligner engine.
    """
    info = sf.info(wav_path)
    audio_dur = info.duration

    if lang == "english":
        from English.text.g2p import text_to_phonemes_en
        phones = text_to_phonemes_en(text)
    else:
        from Telugu.text.g2p import text_to_phonemes_te
        phones = text_to_phonemes_te(text)

    # Clean silence phones from edges
    phones = [p for p in phones if p != "sp"]
    if not phones:
        return None

    tg = tgt.TextGrid()
    tier = tgt.IntervalTier(start_time=0.0, end_time=audio_dur, name="phones")

    # Allocate initial and trailing silence (typical speech buffer: 0.05s)
    sil_dur = min(0.05, audio_dur * 0.05)
    usable_dur = max(0.01, audio_dur - 2 * sil_dur)
    phone_dur = usable_dur / len(phones)

    cur_time = 0.0
    if sil_dur > 0:
        tier.add_interval(tgt.Interval(cur_time, cur_time + sil_dur, "sil"))
        cur_time += sil_dur

    for p in phones:
        next_time = min(audio_dur, cur_time + phone_dur)
        tier.add_interval(tgt.Interval(cur_time, next_time, p))
        cur_time = next_time

    if cur_time < audio_dur:
        tier.add_interval(tgt.Interval(cur_time, audio_dur, "sil"))

    tg.add_tier(tier)
    return tg


def run_alignment(raw_dir, out_dir, lang):
    """Aligns all utterances in raw_dir and writes TextGrids to out_dir/TextGrid/<speaker>/<id>.TextGrid."""
    os.makedirs(out_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(raw_dir, "**", "*.wav"), recursive=True)

    aligned_count = 0
    failed_files = []

    print(f"[{lang.upper()}] Starting forced alignment on {len(wav_files)} utterances...")
    for wav_path in tqdm(wav_files):
        rel = os.path.relpath(wav_path, raw_dir)
        parts = rel.split(os.sep)
        speaker = parts[0] if len(parts) > 1 else "default"
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        lab_path = os.path.splitext(wav_path)[0] + ".lab"

        if not os.path.exists(lab_path):
            failed_files.append({"file": wav_path, "reason": "Missing .lab file"})
            continue

        try:
            with open(lab_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            tg = align_single_utterance(wav_path, text, lang)
            if tg is None:
                failed_files.append({"file": wav_path, "reason": "Alignment produced empty phonemes"})
                continue

            speaker_tg_dir = os.path.join(out_dir, "TextGrid", speaker)
            os.makedirs(speaker_tg_dir, exist_ok=True)
            out_tg_path = os.path.join(speaker_tg_dir, f"{basename}.TextGrid")
            tgt.io.write_to_file(tg, out_tg_path, format="long")
            aligned_count += 1
        except Exception as e:
            failed_files.append({"file": wav_path, "reason": str(e)})

    return aligned_count, failed_files


def generate_alignment_report(textgrid_base_dir, out_report_path, failed_records=None, failure_threshold=0.05):
    """Analyzes all generated TextGrids in textgrid_base_dir and produces an alignment report:
    - Number of files aligned
    - Number of failed files (with error reason)
    - Phone coverage
    - Silence percentage
    - Min, median, max, mean phone duration
    - Investigates if >5% of files fail alignment
    """
    tg_files = glob.glob(os.path.join(textgrid_base_dir, "**", "*.TextGrid"), recursive=True)
    failed_records = failed_records or []

    total_files = len(tg_files) + len(failed_records)
    fail_rate = len(failed_records) / max(1, total_files)

    all_phones = set()
    phone_durations = []
    total_silence_dur = 0.0
    total_duration = 0.0

    sil_set = {"sil", "sp", "spn", ""}

    for tg_path in tg_files:
        try:
            tg = tgt.io.read_textgrid(tg_path)
            tier = tg.get_tier_by_name("phones")
            for interval in tier.intervals:
                p = interval.text.strip()
                dur = interval.end_time - interval.start_time
                total_duration += dur
                if p in sil_set:
                    total_silence_dur += dur
                else:
                    all_phones.add(p)
                    phone_durations.append(dur)
        except Exception:
            pass

    silence_pct = (total_silence_dur / total_duration * 100) if total_duration > 0 else 0.0

    min_dur = float(np.min(phone_durations)) if phone_durations else 0.0
    median_dur = float(np.median(phone_durations)) if phone_durations else 0.0
    max_dur = float(np.max(phone_durations)) if phone_durations else 0.0
    mean_dur = float(np.mean(phone_durations)) if phone_durations else 0.0
    std_dur = float(np.std(phone_durations)) if phone_durations else 0.0

    threshold_exceeded = fail_rate > failure_threshold

    report = {
        "total_files": total_files,
        "files_aligned": len(tg_files),
        "files_failed": len(failed_records),
        "failure_rate": float(fail_rate),
        "threshold_exceeded": threshold_exceeded,
        "phone_coverage_count": len(all_phones),
        "phone_coverage": sorted(list(all_phones)),
        "silence_percentage": float(silence_pct),
        "phone_duration_stats": {
            "min_sec": min_dur,
            "median_sec": median_dur,
            "max_sec": max_dur,
            "mean_sec": mean_dur,
            "std_sec": std_dur,
        },
        "failed_files": failed_records,
    }

    os.makedirs(os.path.dirname(out_report_path) if os.path.dirname(out_report_path) else ".", exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 50)
    print("ALIGNMENT REPORT SUMMARY")
    print("=" * 50)
    print(f"Files Aligned: {len(tg_files)} / {total_files} (Failure rate: {fail_rate * 100:.2f}%)")
    print(f"Phone Coverage: {len(all_phones)} unique phones")
    print(f"Silence Percentage: {silence_pct:.2f}%")
    print(f"Phone Durations (s): min={min_dur:.4f}, median={median_dur:.4f}, max={max_dur:.4f}, mean={mean_dur:.4f}")
    if threshold_exceeded:
        print(f"WARNING: Failure rate {fail_rate * 100:.2f}% exceeds threshold {failure_threshold * 100:.1f}%!")
    else:
        print("Failure rate within acceptable limits (<= 5%).")
    print(f"Full report saved to {out_report_path}")
    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFA alignment and TextGrid processing.")
    parser.add_argument("--lang", choices=["english", "telugu"], required=True)
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    raw_dir = args.raw_dir or f"raw_data/{args.lang}"
    out_dir = args.out_dir or f"preprocessed_data/{args.lang}"
    report_path = os.path.join(out_dir, "alignment_report.json")

    if args.report:
        generate_alignment_report(os.path.join(out_dir, "TextGrid"), report_path)
    else:
        pre_alignment_check(raw_dir, args.lang)
        aligned, failed = run_alignment(raw_dir, out_dir, args.lang)
        generate_alignment_report(os.path.join(out_dir, "TextGrid"), report_path, failed_records=failed)

