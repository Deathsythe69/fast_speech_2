"""tools/sentence_stats.py
Extract and format per-sentence variance statistics for FastSpeech 2 test utterances.

Produces Results/<Lang>/sentence_stats.json and Results/<Lang>/sentence_stats.md
covering:
- #symbols
- total predicted duration (s)
- speaking rate (symbols/s)
- longest symbol and duration (ms, frames)
- mean silence duration (ms)
- voiced F0 mean, min, max, std (Hz)
- F0 range in semitones: 12 * log2(F0_max / F0_min)
- pitch at final voiced symbol vs sentence mean (declination in Hz and semitones)
- energy mean, max, std and peak energy symbol
- numeral/time word breakdowns for E4 and T4
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

SILENCE_TOKENS = {"sp", "spn", "sil", "pau", "<pad>"}

def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-sentence statistics for FastSpeech 2 inference outputs.")
    parser.add_argument("--lang", type=str, choices=["english", "telugu", "both"], default="both",
                        help="Language to analyze: 'english', 'telugu', or 'both'")
    parser.add_argument("--results_dir", type=str, default="Results",
                        help="Root directory where Results/<Lang> outputs are located")
    return parser.parse_args()


def compute_sentence_stats(json_path: Path, npz_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    npz = np.load(npz_path)

    phonemes = [str(p) for p in npz["phonemes"]]
    duration_rounded = npz["duration_rounded"]
    duration_ms = npz["duration_ms"]
    pitch_hz = npz["pitch_hz"]
    energy_val = npz["energy_val"]

    num_symbols = len(phonemes)
    audio_seconds = float(meta.get("stats", {}).get("audio_seconds", np.sum(duration_rounded) * 256.0 / 22050.0))
    speaking_rate = float(num_symbols / audio_seconds) if audio_seconds > 0 else 0.0

    # Longest symbol
    longest_idx = int(np.argmax(duration_ms))
    longest_sym = phonemes[longest_idx]
    longest_dur_ms = float(duration_ms[longest_idx])
    longest_frames = int(duration_rounded[longest_idx])

    # Silence/pause symbols
    sil_indices = [i for i, p in enumerate(phonemes) if p in SILENCE_TOKENS]
    if sil_indices:
        mean_silence_dur_ms = float(np.mean(duration_ms[sil_indices]))
        total_silence_dur_ms = float(np.sum(duration_ms[sil_indices]))
        silence_count = len(sil_indices)
    else:
        mean_silence_dur_ms = 0.0
        total_silence_dur_ms = 0.0
        silence_count = 0

    # Voiced symbols (exclude silence tokens and pitch <= 0)
    voiced_indices = [i for i, p in enumerate(phonemes) if p not in SILENCE_TOKENS and pitch_hz[i] > 0]
    if not voiced_indices:
        voiced_indices = list(range(num_symbols))
    
    voiced_f0 = pitch_hz[voiced_indices]
    f0_mean = float(np.mean(voiced_f0))
    f0_min = float(np.min(voiced_f0))
    f0_max = float(np.max(voiced_f0))
    f0_std = float(np.std(voiced_f0))
    f0_range_st = float(12.0 * np.log2(f0_max / f0_min)) if f0_min > 0 else 0.0

    final_voiced_f0 = float(voiced_f0[-1])
    declination_hz = float(final_voiced_f0 - f0_mean)
    declination_st = float(12.0 * np.log2(final_voiced_f0 / f0_mean)) if (f0_mean > 0 and final_voiced_f0 > 0) else 0.0

    # Energy
    energy_mean = float(np.mean(energy_val))
    energy_max = float(np.max(energy_val))
    energy_std = float(np.std(energy_val))
    peak_energy_idx = int(np.argmax(energy_val))
    peak_energy_sym = phonemes[peak_energy_idx]
    peak_energy_val = float(energy_val[peak_energy_idx])

    stats = {
        "id": meta.get("id"),
        "raw_text": meta.get("raw_text"),
        "normalized_text": meta.get("normalized_text"),
        "speaker": meta.get("speaker"),
        "checkpoint": meta.get("checkpoint"),
        "step": meta.get("step"),
        "num_symbols": num_symbols,
        "total_duration_s": round(audio_seconds, 4),
        "total_frames": int(np.sum(duration_rounded)),
        "speaking_rate_sym_per_s": round(speaking_rate, 2),
        "longest_symbol": {
            "symbol": longest_sym,
            "duration_ms": round(longest_dur_ms, 2),
            "frames": longest_frames,
            "index": longest_idx
        },
        "silence": {
            "count": silence_count,
            "mean_duration_ms": round(mean_silence_dur_ms, 2),
            "total_duration_ms": round(total_silence_dur_ms, 2)
        },
        "pitch_f0_hz": {
            "voiced_symbols_count": len(voiced_indices),
            "mean": round(f0_mean, 2),
            "min": round(f0_min, 2),
            "max": round(f0_max, 2),
            "std": round(f0_std, 2),
            "range_semitones": round(f0_range_st, 2),
            "final_voiced_f0": round(final_voiced_f0, 2),
            "declination_hz": round(declination_hz, 2),
            "declination_semitones": round(declination_st, 2)
        },
        "energy": {
            "mean": round(energy_mean, 4),
            "max": round(energy_max, 4),
            "std": round(energy_std, 4),
            "peak_symbol": peak_energy_sym,
            "peak_symbol_energy": round(peak_energy_val, 4),
            "peak_symbol_index": peak_energy_idx
        }
    }

    # Word-level numeral / time breakdown for E4 / T4
    sent_id = meta.get("id", "")
    word_map = meta.get("word_to_symbol_map", [])
    if sent_id in ("E4", "T4") and word_map:
        target_words = []
        if sent_id == "E4":
            target_words = ["10:45", "AM", "12", "November", "2026,"]
        elif sent_id == "T4":
            target_words = ["2026", "12వ", "10:45"]
        
        numeral_stats = []
        for item in word_map:
            w = item.get("word", "")
            # check if w is target or starts with target
            matched = any(tw in w for tw in target_words)
            if matched:
                s_idx = item.get("start_symbol_idx")
                e_idx = item.get("end_symbol_idx")
                w_phones = phonemes[s_idx:e_idx]
                w_durs_ms = duration_ms[s_idx:e_idx]
                w_durs_frames = duration_rounded[s_idx:e_idx]
                w_pitch = pitch_hz[s_idx:e_idx]
                w_energy = energy_val[s_idx:e_idx]

                w_voiced = [w_pitch[i] for i, p in enumerate(w_phones) if p not in SILENCE_TOKENS and w_pitch[i] > 0]
                numeral_stats.append({
                    "word": w,
                    "phones": w_phones,
                    "symbol_span": [s_idx, e_idx],
                    "total_duration_ms": round(float(np.sum(w_durs_ms)), 2),
                    "total_frames": int(np.sum(w_durs_frames)),
                    "mean_pitch_hz": round(float(np.mean(w_voiced)), 2) if w_voiced else 0.0,
                    "min_pitch_hz": round(float(np.min(w_voiced)), 2) if w_voiced else 0.0,
                    "max_pitch_hz": round(float(np.max(w_voiced)), 2) if w_voiced else 0.0,
                    "mean_energy": round(float(np.mean(w_energy)), 4) if len(w_energy) > 0 else 0.0
                })
        stats["numeral_time_words"] = numeral_stats

    return stats


def generate_markdown_table(lang: str, sentence_stats_list: list) -> str:
    md_lines = []
    title = f"# FastSpeech 2 Per-Sentence Variance Statistics — {lang.capitalize()}"
    md_lines.append(title)
    md_lines.append("")
    md_lines.append("Generated from synthesized test sentences using predicted duration, pitch (F0), and energy.")
    md_lines.append("")
    
    # Main summary table
    md_lines.append("| ID | # Sym | Dur (s) | Rate (sym/s) | Longest Sym (ms) | Mean Sil (ms) | F0 Mean±Std [Min, Max] (Hz) | F0 Range (st) | Declination (Hz) | Energy Mean±Std | Peak Energy Sym |")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for s in sentence_stats_list:
        sid = s["id"]
        ns = s["num_symbols"]
        dur = s["total_duration_s"]
        rate = s["speaking_rate_sym_per_s"]
        longest = f"`{s['longest_symbol']['symbol']}` ({s['longest_symbol']['duration_ms']} ms)"
        sil = f"{s['silence']['mean_duration_ms']} ms" if s['silence']['count'] > 0 else "N/A"
        f0 = f"{s['pitch_f0_hz']['mean']}±{s['pitch_f0_hz']['std']} [{s['pitch_f0_hz']['min']}, {s['pitch_f0_hz']['max']}]"
        f0_range = f"{s['pitch_f0_hz']['range_semitones']} st"
        decl = f"{s['pitch_f0_hz']['declination_hz']} Hz"
        energy = f"{s['energy']['mean']}±{s['energy']['std']}"
        peak_e = f"`{s['energy']['peak_symbol']}` ({s['energy']['peak_symbol_energy']})"

        row = f"| **{sid}** | {ns} | {dur:.2f} | {rate:.1f} | {longest} | {sil} | {f0} | {f0_range} | {decl} | {energy} | {peak_e} |"
        md_lines.append(row)

    md_lines.append("")

    # Sentence text breakdown
    md_lines.append("## Sentence Details")
    for s in sentence_stats_list:
        md_lines.append(f"### {s['id']}")
        md_lines.append(f"- **Raw Text**: {s['raw_text']}")
        md_lines.append(f"- **Normalized Text**: {s['normalized_text']}")
        md_lines.append(f"- **Checkpoint**: `{s['checkpoint']}` (Step {s['step']}) | **Speaker**: `{s['speaker']}`")
        md_lines.append(f"- **Speaking Rate**: {s['speaking_rate_sym_per_s']} symbols/s | **Total Frames**: {s['total_frames']}")
        md_lines.append(f"- **F0 Declination**: {s['pitch_f0_hz']['declination_hz']} Hz ({s['pitch_f0_hz']['declination_semitones']} st) relative to utterance mean ({s['pitch_f0_hz']['mean']} Hz)")
        
        if "numeral_time_words" in s and s["numeral_time_words"]:
            md_lines.append("")
            md_lines.append("#### Numeral & Time Word Breakdown")
            md_lines.append("| Word | Phones | Span | Duration (ms) | Frames | Mean F0 (Hz) | Min/Max F0 (Hz) | Mean Energy |")
            md_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
            for nw in s["numeral_time_words"]:
                ph_str = " ".join(nw["phones"][:8]) + ("..." if len(nw["phones"]) > 8 else "")
                span_str = f"[{nw['symbol_span'][0]}:{nw['symbol_span'][1]}]"
                f0_min_max = f"[{nw['min_pitch_hz']}, {nw['max_pitch_hz']}]"
                wrow = f"| `{nw['word']}` | `{ph_str}` | {span_str} | {nw['total_duration_ms']} | {nw['total_frames']} | {nw['mean_pitch_hz']} | {f0_min_max} | {nw['mean_energy']} |"
                md_lines.append(wrow)
        md_lines.append("")

    return "\n".join(md_lines) + "\n"


def process_language(lang: str, results_dir: Path):
    lang_dir = results_dir / lang.capitalize()
    audio_dir = lang_dir / "test_generated_audios"
    raw_dir = lang_dir / "variance_adaptor_plots" / "raw"

    if not audio_dir.exists() or not raw_dir.exists():
        print(f"[-] Directory for {lang} not found at {lang_dir}. Skipping.")
        return

    test_ids = ["E1", "E2a", "E2b", "E3", "E4"] if lang.lower() == "english" else ["T1", "T2a", "T2b", "T3", "T4"]
    stats_list = []

    for tid in test_ids:
        j_path = audio_dir / f"{tid}.json"
        npz_path = raw_dir / f"{tid}.npz"
        if not j_path.exists() or not npz_path.exists():
            print(f"[-] Warning: missing files for {tid}: {j_path.name} or {npz_path.name}")
            continue
        stats = compute_sentence_stats(j_path, npz_path)
        stats_list.append(stats)

    out_json = lang_dir / "sentence_stats.json"
    out_md = lang_dir / "sentence_stats.md"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats_list, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved {out_json}")

    md_content = generate_markdown_table(lang, stats_list)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Saved {out_md}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    languages = ["english", "telugu"] if args.lang == "both" else [args.lang]

    for lang in languages:
        process_language(lang, results_dir)


if __name__ == "__main__":
    main()

