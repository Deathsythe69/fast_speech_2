"""tools/run_pipeline.py
Orchestrator for FastSpeech 2 inference, variance plotting, sentence stats, and validation.

Executes:
1. <Lang>/inference.py
2. tools/plot_variance.py
3. tools/sentence_stats.py
4. tools/eval_val.py
5. Writes Results/<Lang>/inference_manifest.json
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrate inference and analysis pipeline.")
    parser.add_argument("--lang", type=str, choices=["english", "telugu", "both"], default="english",
                        help="Language to process ('english', 'telugu', or 'both')")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path to evaluate")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                        help="Device to run inference and eval (default: cpu)")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Speaker for English inference")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--provisional", action="store_true", default=True,
                        help="Flag manifest output as provisional dry-run")
    return parser.parse_args()


def compute_file_sha256(file_path: Path) -> str:
    if not file_path.exists():
        return "UNKNOWN"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_command(cmd_args: list):
    print(f"\n[RUN] {' '.join(cmd_args)}")
    res = subprocess.run(cmd_args, capture_output=True, text=True, encoding="utf-8")
    if res.stdout:
        print(res.stdout.strip())
    if res.returncode != 0:
        print(f"[ERROR] Command failed with exit code {res.returncode}")
        if res.stderr:
            print(res.stderr.strip())
        sys.exit(res.returncode)


def orchestrate_language(lang: str, args):
    root_dir = Path(__file__).resolve().parent.parent
    py_exec = sys.executable
    lang_name = lang.capitalize()
    lang_code = lang.lower()

    # Determine checkpoint
    ckpt_path_str = args.checkpoint
    if ckpt_path_str is None:
        cand_best = root_dir / "checkpoints" / lang_code / "best.pth"
        cand_last = root_dir / "checkpoints" / lang_code / "last.pth"
        if cand_best.exists():
            ckpt_path = cand_best
        elif cand_last.exists():
            ckpt_path = cand_last
        else:
            ckpt_path = cand_last
    else:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.is_absolute():
            ckpt_path = root_dir / ckpt_path

    ckpt_hash = compute_file_sha256(ckpt_path)

    # Check step from checkpoint
    step = 0
    if ckpt_path.exists():
        try:
            raw_ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(raw_ckpt, dict):
                step = raw_ckpt.get("step", 0)
        except Exception:
            step = 0

    results_dir = root_dir / "Results" / lang_name
    results_dir.mkdir(parents=True, exist_ok=True)

    commands_executed = []

    # 1. Inference
    inf_script = root_dir / lang_name / "inference.py"
    inf_cmd = [
        py_exec,
        str(inf_script),
        "--checkpoint", str(ckpt_path),
        "--out_dir", str(results_dir),
        "--device", args.device,
        "--pitch_control", str(args.pitch_control),
        "--energy_control", str(args.energy_control),
        "--duration_control", str(args.duration_control),
    ]
    if args.speaker and lang_code == "english":
        inf_cmd.extend(["--speaker", args.speaker])
    commands_executed.append(" ".join(inf_cmd))
    run_command(inf_cmd)

    # 2. Variance Plotting
    plot_script = root_dir / "tools" / "plot_variance.py"
    plot_cmd = [
        py_exec,
        str(plot_script),
        "--lang", lang_code,
        "--out_dir", str(results_dir)
    ]
    commands_executed.append(" ".join(plot_cmd))
    run_command(plot_cmd)

    # 3. Sentence Stats
    stats_script = root_dir / "tools" / "sentence_stats.py"
    stats_cmd = [
        py_exec,
        str(stats_script),
        "--lang", lang_code,
        "--results_dir", str(root_dir / "Results")
    ]
    commands_executed.append(" ".join(stats_cmd))
    run_command(stats_cmd)

    # 4. Quantitative Validation
    eval_script = root_dir / "tools" / "eval_val.py"
    eval_cmd = [
        py_exec,
        str(eval_script),
        "--lang", lang_code,
        "--checkpoint", str(ckpt_path),
        "--device", args.device,
        "--results_dir", str(root_dir / "Results")
    ]
    commands_executed.append(" ".join(eval_cmd))
    run_command(eval_cmd)

    # 5. Build Manifest
    audio_dir = results_dir / "test_generated_audios"
    plot_dir = results_dir / "variance_adaptor_plots"
    
    test_ids = ["E1", "E2a", "E2b", "E3", "E4"] if lang_code == "english" else ["T1", "T2a", "T2b", "T3", "T4"]
    wav_files = [str(audio_dir / f"{tid}.wav") for tid in test_ids if (audio_dir / f"{tid}.wav").exists()]
    json_files = [str(audio_dir / f"{tid}.json") for tid in test_ids if (audio_dir / f"{tid}.json").exists()]
    raw_npz_files = [str(plot_dir / "raw" / f"{tid}.npz") for tid in test_ids if (plot_dir / "raw" / f"{tid}.npz").exists()]

    speaker_used = "speaker_0" if lang_code == "english" else "default"
    if json_files:
        try:
            with open(json_files[0], "r", encoding="utf-8") as f:
                speaker_used = json.load(f).get("speaker", speaker_used)
        except Exception:
            pass

    manifest = {
        "language": lang_name,
        "provisional": bool(args.provisional or "provisional" in str(ckpt_path).lower() or step <= 1000),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(ckpt_path),
            "step": step,
            "sha256": ckpt_hash,
        },
        "speaker": speaker_used,
        "controls": {
            "duration": args.duration_control,
            "pitch": args.pitch_control,
            "energy": args.energy_control,
        },
        "commands": commands_executed,
        "outputs": {
            "audio_count": len(wav_files),
            "wav_files": [Path(p).name for p in wav_files],
            "json_files": [Path(p).name for p in json_files],
            "raw_npz_files": [Path(p).name for p in raw_npz_files],
            "sentence_stats_json": str(results_dir / "sentence_stats.json"),
            "sentence_stats_md": str(results_dir / "sentence_stats.md"),
            "val_metrics_json": str(results_dir / "val_metrics.json"),
            "val_metrics_md": str(results_dir / "val_metrics.md"),
        }
    }

    manifest_path = results_dir / "inference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[+] Saved pipeline manifest to {manifest_path}")


def main():
    args = parse_args()
    languages = ["english", "telugu"] if args.lang == "both" else [args.lang]
    for lang in languages:
        orchestrate_language(lang, args)


if __name__ == "__main__":
    main()

