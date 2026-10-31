#!/usr/bin/env python3
"""Phase 2: dataset selection and subsampling for FastSpeech 2 (English + Telugu)."""

import argparse
import hashlib
import json
import random
import re
import string
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

TELUGU_START, TELUGU_END = 0x0C00, 0x0C80
BRACKET_RE = re.compile(r"\[[^\]]*\]|<[^>]*>")
WS_RE = re.compile(r"\s+")
TELUGU_PUNCT = set(" .,!?-")


def parse_args():
    """Parse CLI options for inspect, select, and verify modes."""
    p = argparse.ArgumentParser(description="Select train/val subset for FastSpeech 2.")
    p.add_argument("--lang", required=True, choices=["english", "telugu"])
    p.add_argument("--src", default=None, help="Dataset root dir.")
    p.add_argument("--manifest", default=None, help="csv / json / jsonl manifest.")
    p.add_argument("--col_id", default=None); p.add_argument("--col_audio", default=None)
    p.add_argument("--col_text", default=None); p.add_argument("--col_speaker", default=None)
    p.add_argument("--out_dir", default=None); p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--train_hours", type=float, default=5.0); p.add_argument("--val_hours", type=float, default=1.0)
    p.add_argument("--min_dur", type=float, default=1.0); p.add_argument("--max_dur", type=float, default=12.0)
    p.add_argument("--speakers", default=None, help="Optional comma-separated speaker whitelist.")
    p.add_argument("--inspect", action="store_true", help="Inspect manifest layout and exit.")
    p.add_argument("--verify", action="store_true", help="Verify train.csv and val.csv splits.")
    args = p.parse_args()
    if not args.verify:
        req = [k for k, v in [("--src", args.src), ("--manifest", args.manifest), ("--col_id", args.col_id),
                             ("--col_audio", args.col_audio), ("--col_text", args.col_text)] if not v]
        if req: p.error(f"Missing required arguments: {', '.join(req)}")
    return args


def read_manifest(path):
    """Load a csv, json, or jsonl manifest into a DataFrame."""
    s = Path(path).suffix.lower()
    if s == ".csv": return pd.read_csv(path)
    if s in (".jsonl", ".json"): return pd.read_json(path, lines=(s == ".jsonl"))
    raise ValueError(f"Unsupported manifest extension: {s}")


def build_records(df, col_id, col_audio, col_text, col_speaker, src):
    """Convert dataframe rows to record dicts resolving audio paths."""
    records, src_p = [], Path(src) if src else Path(".")
    for _, row in df.iterrows():
        a = str(row[col_audio])
        p = a if Path(a).is_absolute() else str(src_p / a)
        spk = str(row[col_speaker]) if col_speaker and pd.notna(row[col_speaker]) else "speaker_0"
        records.append({"id": str(row[col_id]), "speaker": spk, "audio_path": p,
                        "text": str(row[col_text]) if pd.notna(row[col_text]) else ""})
    return records


def audio_duration(path):
    """Get audio duration in seconds via soundfile header without decoding."""
    try: return sf.info(path).duration
    except Exception: return None


def clean_text(text, lang):
    """Clean and filter text per language rules. Return None if rejected."""
    if lang == "english":
        t = WS_RE.sub(" ", str(text).strip())
        return None if BRACKET_RE.search(t) or not t else t
    t = WS_RE.sub(" ", unicodedata.normalize("NFC", str(text)).strip())
    allowed = set(string.digits) | TELUGU_PUNCT | {chr(c) for c in range(TELUGU_START, TELUGU_END)}
    return t if t and all(c in allowed for c in t) else None


def normalize_transcript(text):
    """Lowercase and strip punctuation while preserving letters, marks, and digits."""
    t = unicodedata.normalize("NFC", str(text).lower())
    chars = [" " if unicodedata.category(c).startswith("P") or c in string.punctuation else c for c in t]
    return WS_RE.sub(" ", "".join(chars)).strip()


def load_test_norms(lang):
    """Load normalized test sentences from test_sentences.json."""
    for p in (Path.cwd() / "test_sentences.json", Path(__file__).resolve().parents[1] / "test_sentences.json"):
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            k = lang.capitalize() if lang.capitalize() in data else lang.lower()
            return {normalize_transcript(v) for v in data.get(k, {}).values() if v and v != "PASTE_FROM_PDF"}
    return set()


def inspect(df, records):
    """Print manifest columns, sample rows, counts, percentiles, sample rates, then exit."""
    print("Columns:", list(df.columns))
    print(df.head(5).to_string(index=False))
    print(f"Number of rows: {len(records)}, Number of speakers: {len({r['speaker'] for r in records})}")
    durs = [r["duration"] for r in records if r.get("duration") is not None]
    if durs:
        print(f"Total hours: {sum(durs) / 3600.0:.4f}")
        print("Duration percentiles (5/50/95):", [round(float(x), 3) for x in np.percentile(durs, [5, 50, 95])])
    srs = [sf.info(r["audio_path"]).samplerate for r in records[:5]
           if r.get("duration") is not None and Path(r["audio_path"]).exists()]
    print("Sample rate of 5 files:", srs)


def select_split(groups, lang, val_s, train_s, seed):
    """Split transcript groups into val and train respecting speaker budgets and tolerances."""
    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)
    spk_totals = {}
    for _, recs in groups:
        for r in recs: spk_totals[r["speaker"]] = spk_totals.get(r["speaker"], 0.0) + r["duration"]
    tot_dur = sum(spk_totals.values())
    budget = {s: val_s * (d / tot_dur) for s, d in spk_totals.items()} if lang == "english" and tot_dur > 0 else {}
    used_spk = {s: 0.0 for s in spk_totals}
    val, train, val_used, train_used, val_ids = [], [], 0.0, 0.0, set()

    for norm, recs in shuffled:
        g_dur = sum(r["duration"] for r in recs)
        spk = recs[0]["speaker"]
        if lang == "english" and spk in budget:
            if val_used + g_dur <= val_s * 1.05 and used_spk[spk] + g_dur <= budget[spk] * 1.05 + 10.0:
                val.extend(recs); val_used += g_dur; used_spk[spk] += g_dur; val_ids.add(norm)
        elif lang == "telugu" and val_used + g_dur <= val_s * 1.05:
            val.extend(recs); val_used += g_dur; val_ids.add(norm)

    if val_used < val_s * 0.95:
        for norm, recs in shuffled:
            if norm in val_ids: continue
            g_dur = sum(r["duration"] for r in recs)
            if val_used + g_dur <= val_s * 1.05:
                val.extend(recs); val_used += g_dur; val_ids.add(norm)
                if val_used >= val_s * 0.95: break

    for norm, recs in shuffled:
        if norm in val_ids: continue
        g_dur = sum(r["duration"] for r in recs)
        if train_used + g_dur <= train_s * 1.03:
            train.extend(recs); train_used += g_dur
    return val, train


def convert_audio(record, out_wav):
    """Load, convert to mono 22050 Hz 16-bit PCM WAV, trim top_db=35 keeping 0.1s padding, and save."""
    y, sr = librosa.load(record["audio_path"], sr=None, mono=True)
    y_trim, (s_idx, e_idx) = librosa.effects.trim(y, top_db=35)
    pad = int(0.1 * sr)
    y_pad = y[max(0, s_idx - pad):min(len(y), e_idx + pad)]
    lp, rp = max(0, pad - s_idx), max(0, pad - (len(y) - e_idx))
    if lp > 0 or rp > 0: y_pad = np.pad(y_pad, (lp, rp))
    y_22k = librosa.resample(y_pad, orig_sr=sr, target_sr=22050) if sr != 22050 else y_pad
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, y_22k, 22050, subtype="PCM_16")
    out_wav.with_suffix(".lab").write_text(record["text"], encoding="utf-8")
    return len(y_22k) / 22050.0


def write_split(split_name, records, lang, raw_data_root, csv_dir):
    """Process audio files for a split, save .wav/.lab, and write split CSV."""
    out = []
    for r in tqdm(records, desc=f"Writing {split_name}"):
        out_wav = raw_data_root / lang / r["speaker"] / f"{r['id']}.wav"
        dur = convert_audio(r, out_wav)
        out.append({"id": r["id"], "speaker": r["speaker"], "wav_path": str(out_wav.as_posix()),
                    "text": r["text"], "duration_sec": round(dur, 4)})
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(csv_dir / f"{split_name}.csv", index=False)
    return out


def split_stats(records):
    """Calculate summary statistics for a split."""
    if not records:
        return {"hours": 0.0, "utterances": 0, "speakers": 0, "mean_dur": 0.0, "median_dur": 0.0, "vocab_size": 0}
    durs = [r["duration_sec"] for r in records]
    return {"hours": round(sum(durs) / 3600.0, 4), "utterances": len(records),
            "speakers": len({r["speaker"] for r in records}), "mean_dur": round(float(np.mean(durs)), 4),
            "median_dur": round(float(np.median(durs)), 4), "vocab_size": len(set("".join(r["text"] for r in records)))}


def write_stats(lang, train_recs, val_recs, params, results_root):
    """Write data_stats.json and data_stats.md table."""
    out_dir = results_root / lang.capitalize()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"seed": params["seed"], "parameters": params,
             "train": split_stats(train_recs), "val": split_stats(val_recs)}
    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Data stats — {lang.capitalize()}", "",
             "| Split | Hours | Utterances | Speakers | Mean dur (s) | Median dur (s) | Vocab size |",
             "|---|---|---|---|---|---|---|"]
    for split, recs in (("train", train_recs), ("val", val_recs)):
        s = split_stats(recs)
        lines.append(f"| {split} | {s['hours']} | {s['utterances']} | {s['speakers']} | {s['mean_dur']} | "
                     f"{s['median_dur']} | {s['vocab_size']} |")
    lines.extend(["", "Parameters: " + ", ".join(f"{k}={v}" for k, v in params.items())])
    (out_dir / "data_stats.md").write_text("\n".join(lines), encoding="utf-8")


def md5_file(path):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def verify(lang, csv_dir, train_hours=5.0, val_hours=1.0):
    """Verify integrity, disjointness, test exclusion, and duration tolerances."""
    tr_csv, val_csv = csv_dir / "train.csv", csv_dir / "val.csv"
    if not tr_csv.exists() or not val_csv.exists():
        print(f"FAIL: {tr_csv} or {val_csv} missing"); return
    train_df, val_df = pd.read_csv(tr_csv), pd.read_csv(val_csv)
    try:
        shared_ids = set(train_df["id"].astype(str)) & set(val_df["id"].astype(str))
        assert not shared_ids, f"shared IDs between train and val: {len(shared_ids)}"
        shared_hashes = {md5_file(p) for p in train_df["wav_path"]} & {md5_file(p) for p in val_df["wav_path"]}
        assert not shared_hashes, f"shared WAV hashes: {len(shared_hashes)}"
        tn = {normalize_transcript(t) for t in train_df["text"].fillna("")}
        vn = {normalize_transcript(t) for t in val_df["text"].fillna("")}
        assert not (tn & vn), f"shared transcripts: {len(tn & vn)}"
        test_present = (tn | vn) & load_test_norms(lang)
        assert not test_present, f"test sentence present in splits: {test_present}"
        th, vh = train_df["duration_sec"].sum() / 3600.0, val_df["duration_sec"].sum() / 3600.0
        assert train_hours * 0.97 <= th <= train_hours * 1.03, (
            f"train hours {th:.3f}h outside [{train_hours*0.97:.3f}h, {train_hours*1.03:.3f}h]")
        assert val_hours * 0.95 <= vh <= val_hours * 1.05, (
            f"val hours {vh:.3f}h outside [{val_hours*0.95:.3f}h, {val_hours*1.05:.3f}h]")
        print(f"Train hours: {th:.4f}, Val hours: {vh:.4f}\nPASS")
    except AssertionError as e:
        print(f"FAIL: {e}")


def main():
    """Main CLI entrypoint."""
    args = parse_args()
    lang, csv_dir = args.lang, Path(args.out_dir or f"data/{args.lang}")
    raw_data_root, results_root = Path("raw_data"), Path.cwd() / "Results"

    if args.verify:
        verify(lang, csv_dir, args.train_hours, args.val_hours); return

    df = read_manifest(args.manifest)
    records = build_records(df, args.col_id, args.col_audio, args.col_text, args.col_speaker, args.src)
    for r in records: r["duration"] = audio_duration(r["audio_path"])
    if args.inspect:
        inspect(df, records); return

    spk_filter = {s.strip() for s in args.speakers.split(",")} if args.speakers else None
    cleaned = []
    for r in records:
        if r["duration"] is None or not (args.min_dur <= r["duration"] <= args.max_dur): continue
        if spk_filter and r["speaker"] not in spk_filter: continue
        t = clean_text(r["text"], lang)
        if t is None: continue
        r["text"] = t; cleaned.append(r)

    test_norms = load_test_norms(lang)
    records = [r for r in cleaned if normalize_transcript(r["text"]) not in test_norms]
    groups = {}
    for r in records: groups.setdefault(normalize_transcript(r["text"]), []).append(r)

    val_recs, train_recs = select_split(list(groups.items()), lang, args.val_hours * 3600.0,
                                        args.train_hours * 3600.0, args.seed)
    train_recs = write_split("train", train_recs, lang, raw_data_root, csv_dir)
    val_recs = write_split("val", val_recs, lang, raw_data_root, csv_dir)
    params = {"lang": lang, "seed": args.seed, "train_hours": args.train_hours, "val_hours": args.val_hours,
              "min_dur": args.min_dur, "max_dur": args.max_dur, "speakers": args.speakers}
    write_stats(lang, train_recs, val_recs, params, results_root)
    print(f"Done. train={len(train_recs)} val={len(val_recs)}")


if __name__ == "__main__":
    main()
