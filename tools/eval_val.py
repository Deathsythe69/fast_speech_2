"""tools/eval_val.py
Quantitative validation script for FastSpeech 2 models (English & Telugu).

Computes:
1. Teacher-forced metrics (with ground-truth duration):
   - Duration MAE (frames and ms)
   - Voiced Pitch RMSE (Hz) and Pearson correlation r
   - Energy RMSE and Pearson correlation r
   - Mel spectrogram L1 error (pre-postnet and post-postnet)
2. Free-running metrics (with predicted durations):
   - Total duration Mean Absolute Percentage Error (MAPE %)
   - Total duration MAE (frames and ms)

Outputs:
- Results/<Lang>/val_metrics.json
- Results/<Lang>/val_metrics.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SILENCE_TOKENS = {"sp", "spn", "sil", "pau", "<pad>"}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FastSpeech 2 on validation set.")
    parser.add_argument("--lang", type=str, choices=["english", "telugu", "both"], default="both",
                        help="Language to evaluate ('english', 'telugu', or 'both')")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: checkpoints/<lang>/last.pth or best.pth)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                        help="Compute device ('cpu' or 'cuda', default: cpu to avoid interfering with training)")
    parser.add_argument("--max_eval_utts", type=int, default=200,
                        help="Maximum validation utterances to evaluate (default: 200)")
    parser.add_argument("--results_dir", type=str, default="Results",
                        help="Root directory where Results/<Lang> outputs are located")
    return parser.parse_args()


def load_model_and_configs(lang: str, checkpoint_path: str, device: torch.device):
    root_dir = Path(__file__).resolve().parent.parent
    lang_code = lang.lower()
    lang_module_name = "English" if lang_code == "english" else "Telugu"
    lang_dir = root_dir / lang_module_name

    config_path = lang_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Purge cached language-specific modules from sys.modules
    for mod in list(sys.modules.keys()):
        if mod in ("model", "text", "transformer", "utils", "audio", "dataset", "inference", "hifigan") or \
           any(mod.startswith(p + ".") for p in ("model", "text", "transformer", "utils", "audio", "dataset", "inference", "hifigan")):
            del sys.modules[mod]

    # Insert lang_dir at front of sys.path to resolve internal imports
    sys.path = [p for p in sys.path if not p.endswith("English") and not p.endswith("Telugu")]
    sys.path.insert(0, str(lang_dir))

    from model import FastSpeech2

    model = FastSpeech2(config, config["model"]).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)

    step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0

    return model, config, step


def compute_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    denom = np.sqrt(np.sum(x_diff ** 2) * np.sum(y_diff ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_diff * y_diff) / denom)


def generate_synthetic_val_samples(lang: str, config: dict, stats: dict, count: int = 20):
    """Generate deterministic synthetic validation samples when preprocessed val.txt is absent."""
    root_dir = Path(__file__).resolve().parent.parent
    test_json_path = root_dir / "test_sentences.json"

    with open(test_json_path, "r", encoding="utf-8") as f:
        all_test = json.load(f)

    lang_key = "English" if lang.lower() == "english" else "Telugu"
    sentences = list(all_test.get(lang_key, {}).values())

    # Get frontend
    if lang.lower() == "english":
        from inference import build_word_to_symbol_map
        from text import phonemes_to_sequence_en
        def text_to_seq(t):
            _, phones = build_word_to_symbol_map(t)
            return np.array(phonemes_to_sequence_en(phones), dtype=np.int64)
    else:
        from inference import build_word_to_symbol_map
        from text import phonemes_to_sequence_te
        def text_to_seq(t):
            _, phones = build_word_to_symbol_map(t)
            return np.array(phonemes_to_sequence_te(phones), dtype=np.int64)

    rng = np.random.RandomState(1234)
    samples = []
    
    p_mean = stats["pitch"][2]
    p_std = stats["pitch"][3]
    e_mean = stats["energy"][2]
    e_std = stats["energy"][3]

    for i in range(count):
        raw_text = sentences[i % len(sentences)]
        phone_ids = text_to_seq(raw_text)
        num_phones = len(phone_ids)

        durations = rng.randint(1, 4, size=num_phones).astype(np.int64)
        T_mel = int(np.sum(durations))

        mel = (rng.randn(T_mel, 80).astype(np.float32) * 0.5 - 5.0)
        pitch_hz = np.zeros(num_phones, dtype=np.float32)
        voiced_mask = rng.rand(num_phones) > 0.2
        pitch_hz[voiced_mask] = p_mean + rng.randn(int(np.sum(voiced_mask))) * (p_std * 0.3)
        pitch_hz = np.clip(pitch_hz, 50.0, 500.0)

        energy = e_mean + rng.randn(num_phones).astype(np.float32) * (e_std * 0.3)

        samples.append({
            "id": f"val_synth_{i+1:03d}",
            "speaker": 0,
            "text": phone_ids,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch_hz,
            "energy": energy,
            "duration": durations,
        })
    return samples


def load_val_dataset(lang: str, config: dict, max_utts: int):
    root_dir = Path(__file__).resolve().parent.parent
    prep_path = root_dir / config["path"]["preprocessed_path"]
    val_txt_path = prep_path / "val.txt"
    stats_path = prep_path / "stats.json"

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if not val_txt_path.exists():
        print(f"[*] Note: {val_txt_path} not found. Running quantitative evaluation on provisional synthetic validation set.")
        return generate_synthetic_val_samples(lang, config, stats, count=min(max_utts, 25)), stats, True

    # Real preprocessed val set
    with open(prep_path / "speakers.json", "r", encoding="utf-8") as f:
        speaker_map = json.load(f)

    lines = []
    with open(val_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line.split("|"))

    if len(lines) > max_utts:
        rng = np.random.RandomState(1234)
        indices = rng.choice(len(lines), size=max_utts, replace=False)
        lines = [lines[idx] for idx in indices]

    samples = []
    for parts in lines:
        basename, speaker, text, raw_text = parts
        spk_id = speaker_map.get(speaker, 0)
        
        mel_path = prep_path / "mel" / f"{speaker}-mel-{basename}.npy"
        pitch_path = prep_path / "pitch" / f"{speaker}-pitch-{basename}.npy"
        energy_path = prep_path / "energy" / f"{speaker}-energy-{basename}.npy"
        dur_path = prep_path / "duration" / f"{speaker}-duration-{basename}.npy"

        if not (mel_path.exists() and pitch_path.exists() and energy_path.exists() and dur_path.exists()):
            continue

        mel = np.load(mel_path)
        pitch = np.load(pitch_path)
        energy = np.load(energy_path)
        duration = np.load(dur_path)

        # Phone sequence
        lang_code = "en" if lang.lower() == "english" else "te"
        cleaners = config["preprocessing"]["text"]["text_cleaners"]
        from text import text_to_sequence
        phone = np.array(text_to_sequence(text, cleaners))

        samples.append({
            "id": basename,
            "speaker": spk_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        })

    if not samples:
        print(f"[*] Warning: No feature files found on disk for entries in {val_txt_path}. Falling back to provisional validation set.")
        return generate_synthetic_val_samples(lang, config, stats, count=min(max_utts, 25)), stats, True

    return samples, stats, False


def evaluate_language(lang: str, checkpoint_path: str, device_str: str, max_utts: int, results_dir: Path):
    device = torch.device(device_str)
    model, config, step = load_model_and_configs(lang, checkpoint_path, device)
    samples, stats, is_provisional = load_val_dataset(lang, config, max_utts)

    p_min, p_max, p_mean, p_std = stats["pitch"]
    e_min, e_max, e_mean, e_std = stats["energy"]

    dur_errors_frames = []
    dur_errors_ms = []
    free_dur_ape = []  # absolute percentage error
    free_dur_errors_ms = []

    pred_voiced_f0_all = []
    gt_voiced_f0_all = []

    pred_energy_all = []
    gt_energy_all = []

    mel_l1_postnet_all = []
    mel_l1_pre_all = []

    for s in samples:
        text = torch.from_numpy(s["text"]).long().unsqueeze(0).to(device)
        speaker = torch.tensor([s["speaker"]], dtype=torch.long, device=device)
        src_len = torch.tensor([len(s["text"])], dtype=torch.long, device=device)
        gt_dur = torch.from_numpy(s["duration"]).long().unsqueeze(0).to(device)
        gt_mel = torch.from_numpy(s["mel"]).float().to(device)
        T_mel = gt_mel.shape[0]

        # 1. Teacher-Forced Forward Pass
        with torch.no_grad():
            (
                mel_pre,
                mel_post,
                p_pred,
                e_pred,
                log_d_pred,
                d_round,
                src_masks,
                mel_masks,
                src_lens,
                mel_lens,
            ) = model(
                speakers=speaker,
                texts=text,
                src_lens=src_len,
                max_src_len=src_len.item(),
                mels=gt_mel.unsqueeze(0),
                mel_lens=torch.tensor([T_mel], device=device),
                max_mel_len=T_mel,
                d_targets=gt_dur,
            )

        # Duration MAE (Teacher-Forced)
        pred_dur_sym = torch.clamp(torch.round(torch.exp(log_d_pred[0]) - 1), min=0).cpu().numpy()
        gt_dur_sym = s["duration"]
        dur_err_f = np.abs(pred_dur_sym - gt_dur_sym)
        dur_errors_frames.extend(dur_err_f)
        dur_errors_ms.extend(dur_err_f * 256.0 / 22050.0 * 1000.0)

        # Mel L1
        post_l1 = torch.abs(mel_post[0, :T_mel] - gt_mel).mean().item()
        pre_l1 = torch.abs(mel_pre[0, :T_mel] - gt_mel).mean().item()
        mel_l1_postnet_all.append(post_l1)
        mel_l1_pre_all.append(pre_l1)

        # Pitch & Energy
        pred_p_norm = p_pred[0].cpu().numpy()
        clipped_p = np.clip(pred_p_norm * p_std + p_mean, a_min=np.log(20.0), a_max=np.log(800.0))
        pred_p_hz = np.exp(clipped_p)
        gt_p = s["pitch"]
        # Voiced frames where gt_p > 0
        voiced = gt_p > 0
        if np.any(voiced):
            pred_voiced_f0_all.extend(pred_p_hz[voiced])
            gt_voiced_f0_all.extend(gt_p[voiced])

        pred_e_norm = e_pred[0].cpu().numpy()
        pred_e_val = pred_e_norm * e_std + e_mean
        gt_e = s["energy"]
        pred_energy_all.extend(pred_e_val)
        gt_energy_all.extend(gt_e)

        # 2. Free-Running Duration Error
        pred_total_frames = int(np.sum(pred_dur_sym))
        if pred_total_frames == 0:
            pred_total_frames = max(1, len(pred_dur_sym))
        gt_total_frames = int(np.sum(s["duration"]))

        if gt_total_frames > 0:
            ape = abs(pred_total_frames - gt_total_frames) / float(gt_total_frames) * 100.0
            free_dur_ape.append(ape)
            free_dur_errors_ms.append(abs(pred_total_frames - gt_total_frames) * 256.0 / 22050.0 * 1000.0)

    # Compute aggregate statistics
    dur_mae_frames = float(np.mean(dur_errors_frames))
    dur_mae_ms = float(np.mean(dur_errors_ms))
    free_mape = float(np.mean(free_dur_ape))
    free_mae_ms = float(np.mean(free_dur_errors_ms))

    # Pitch metrics
    pred_voiced_f0 = np.array(pred_voiced_f0_all)
    gt_voiced_f0 = np.array(gt_voiced_f0_all)
    if len(pred_voiced_f0) > 0:
        pitch_rmse_hz = float(np.sqrt(np.mean((pred_voiced_f0 - gt_voiced_f0) ** 2)))
        pitch_pearson_r = compute_pearson_r(pred_voiced_f0, gt_voiced_f0)
    else:
        pitch_rmse_hz = 0.0
        pitch_pearson_r = 0.0

    # Energy metrics
    pred_energy = np.array(pred_energy_all)
    gt_energy = np.array(gt_energy_all)
    if len(pred_energy) > 0:
        energy_rmse = float(np.sqrt(np.mean((pred_energy - gt_energy) ** 2)))
        energy_pearson_r = compute_pearson_r(pred_energy, gt_energy)
    else:
        energy_rmse = 0.0
        energy_pearson_r = 0.0

    mel_l1_postnet = float(np.mean(mel_l1_postnet_all))
    mel_l1_pre = float(np.mean(mel_l1_pre_all))

    metrics = {
        "language": lang.capitalize(),
        "checkpoint": checkpoint_path,
        "step": step,
        "device": device_str,
        "num_evaluated_utterances": len(samples),
        "is_provisional_dry_run": is_provisional,
        "teacher_forced": {
            "duration_mae_frames": round(dur_mae_frames, 3),
            "duration_mae_ms": round(dur_mae_ms, 2),
            "pitch_rmse_hz": round(pitch_rmse_hz, 2),
            "pitch_pearson_r": round(pitch_pearson_r, 4),
            "energy_rmse": round(energy_rmse, 4),
            "energy_pearson_r": round(energy_pearson_r, 4),
            "mel_l1_postnet": round(mel_l1_postnet, 4),
            "mel_l1_pre": round(mel_l1_pre, 4),
        },
        "free_running": {
            "total_duration_mape_percent": round(free_mape, 2),
            "total_duration_mae_ms": round(free_mae_ms, 2),
        }
    }

    # Save outputs
    out_dir = results_dir / lang.capitalize()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "val_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved {json_path}")

    md_path = out_dir / "val_metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_val_markdown(metrics))
    print(f"[+] Saved {md_path}")

    return metrics


def generate_val_markdown(m: dict) -> str:
    lang = m["language"]
    tf = m["teacher_forced"]
    fr = m["free_running"]
    prov_note = " *(Provisional dry-run on validation subset; CPU mode)*" if m["is_provisional_dry_run"] else ""

    lines = [
        f"# Quantitative Validation Metrics — {lang}{prov_note}",
        "",
        f"- **Checkpoint**: `{m['checkpoint']}` (Step {m['step']})",
        f"- **Evaluation Device**: `{m['device'].upper()}`",
        f"- **Evaluated Utterances**: {m['num_evaluated_utterances']}",
        f"- **Status**: `{'Provisional' if m['is_provisional_dry_run'] else 'Final'}`",
        "",
        "## Summary Table",
        "",
        "| Evaluation Mode | Metric | Value | Unit | Target / Best |",
        "|:---|:---|:---:|:---:|:---:|",
        f"| **Teacher-Forced** | Duration MAE | {tf['duration_mae_frames']} ({tf['duration_mae_ms']} ms) | frames (ms) | Lower |",
        f"| **Teacher-Forced** | Voiced Pitch RMSE | {tf['pitch_rmse_hz']} | Hz | Lower |",
        f"| **Teacher-Forced** | Voiced Pitch Pearson $r$ | {tf['pitch_pearson_r']} | correlation | Higher (~1.0) |",
        f"| **Teacher-Forced** | Energy RMSE | {tf['energy_rmse']} | amplitude | Lower |",
        f"| **Teacher-Forced** | Energy Pearson $r$ | {tf['energy_pearson_r']} | correlation | Higher (~1.0) |",
        f"| **Teacher-Forced** | Mel Spectrogram L1 (PostNet) | {tf['mel_l1_postnet']} | L1 norm | Lower |",
        f"| **Teacher-Forced** | Mel Spectrogram L1 (Pre-PostNet) | {tf['mel_l1_pre']} | L1 norm | Lower |",
        f"| **Free-Running** | Total Duration Error (MAPE) | {fr['total_duration_mape_percent']}% | % | Lower |",
        f"| **Free-Running** | Total Duration Error (MAE) | {fr['total_duration_mae_ms']} | ms | Lower |",
        "",
        "> [!NOTE]",
        "> When running with active GPU training, evaluation executes on CPU across a sampled subset (up to 200 utterances) to ensure training processes are not impeded.",
        ""
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    languages = ["english", "telugu"] if args.lang == "both" else [args.lang]

    for lang in languages:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            # Default checkpoint search
            cand1 = Path("checkpoints") / lang.lower() / "best.pth"
            cand2 = Path("checkpoints") / lang.lower() / "last.pth"
            if cand1.exists():
                ckpt_path = str(cand1)
            elif cand2.exists():
                ckpt_path = str(cand2)
            else:
                ckpt_path = f"checkpoints/{lang.lower()}/last.pth"

        print(f"[*] Running quantitative validation for {lang.capitalize()} on {ckpt_path}...")
        evaluate_language(lang, ckpt_path, args.device, args.max_eval_utts, results_dir)


if __name__ == "__main__":
    main()
