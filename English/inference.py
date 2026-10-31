"""Inference and synthesis script for English FastSpeech 2.
Synthesizes test sentences, exports 22050 Hz 16-bit PCM WAVs, JSON metadata,
and raw variance adaptor representations (.npz) for plotting and evaluation.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml

dir_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(dir_path)
for p in [dir_path, parent_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hifigan import AttrDict
from hifigan.models import Generator as HiFiGANGenerator
from model.fastspeech2 import FastSpeech2
from text.g2p import phonemes_to_sequence_en, text_to_phonemes_en
from text.normalizer import normalize_text_en
from utils.tools import get_mask_from_lengths


def load_vocoder(vocoder_dir, device):
    """Loads pretrained HiFi-GAN vocoder from vocoder/ directory."""
    config_path = os.path.join(vocoder_dir, "config.json")
    ckpt_path = os.path.join(vocoder_dir, "generator_universal.pth.tar")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = AttrDict(json.load(f))

    vocoder = HiFiGANGenerator(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def get_dominant_speaker(train_csv_path):
    """Returns speaker with the most training audio from data/english/train.csv, or default 'speaker_0'."""
    if os.path.exists(train_csv_path):
        import pandas as pd
        try:
            df = pd.read_csv(train_csv_path)
            if "speaker" in df.columns and "duration" in df.columns:
                spk_dur = df.groupby("speaker")["duration"].sum()
                dominant = spk_dur.idxmax()
                return str(dominant)
        except Exception:
            pass
    return "speaker_0"


def build_word_to_symbol_map(raw_text):
    """Builds word to phoneme index mapping using the English frontend."""
    words = raw_text.split()
    all_phones = text_to_phonemes_en(raw_text)

    mapping = []
    phone_idx = 0

    for w in words:
        w_phones = text_to_phonemes_en(w)
        w_phones_clean = [p for p in w_phones if p != "sp"]
        start_idx = phone_idx

        # Locate w_phones_clean in all_phones starting from phone_idx
        match_len = len(w_phones_clean)
        found = False
        for i in range(phone_idx, len(all_phones) - match_len + 1):
            if all_phones[i : i + match_len] == w_phones_clean:
                start_idx = i
                phone_idx = i + match_len
                found = True
                break
        if not found:
            end_idx = min(len(all_phones), start_idx + max(1, match_len))
        else:
            end_idx = phone_idx

        mapping.append({
            "word": w,
            "start_symbol_idx": start_idx,
            "end_symbol_idx": end_idx,
            "phones": w_phones_clean,
        })

    return mapping, all_phones


def run_inference(args):
    # Set fixed seeds for deterministic reproducibility
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[English Inference] Running on device: {device}")

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Output directory setup
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "test_generated_audios"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "variance_adaptor_plots", "raw"), exist_ok=True)

    # 1. Load FastSpeech 2 Model
    model = FastSpeech2(config, config["model"]).to(device)
    print(f"[English Inference] Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    checkpoint_step = 0
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        checkpoint_step = ckpt.get("step", 0)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise ValueError("Invalid checkpoint format.")

    model.eval()
    model.requires_grad_ = False

    # 2. Load Vocoder
    vocoder_dir = os.path.join(parent_path, "vocoder")
    if not os.path.exists(vocoder_dir):
        vocoder_dir = "vocoder"
    vocoder = load_vocoder(vocoder_dir, device)

    # 3. Load stats
    stats_path = os.path.join(config["path"]["preprocessed_path"], "stats.json")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(parent_path, "preprocessed_data", "english", "stats.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
    energy_min, energy_max, energy_mean, energy_std = stats["energy"]

    # 4. Speaker setup
    train_csv = os.path.join(parent_path, "data", "english", "train.csv")
    chosen_speaker = args.speaker or get_dominant_speaker(train_csv)
    speakers_map_path = os.path.join(config["path"]["preprocessed_path"], "speakers.json")
    if not os.path.exists(speakers_map_path):
        speakers_map_path = os.path.join(parent_path, "preprocessed_data", "english", "speakers.json")

    speaker_id = 0
    if os.path.exists(speakers_map_path):
        with open(speakers_map_path, "r", encoding="utf-8") as f:
            spk_map = json.load(f)
        speaker_id = spk_map.get(chosen_speaker, 0)

    # 5. Load test sentences
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)["English"]

    print(f"\n{'='*70}")
    print(f"{'ID':<6} | {'Symbols':<8} | {'Frames':<8} | {'Audio (s)':<10} | {'RTF':<8}")
    print(f"{'='*70}")

    sr = config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]

    for test_id, raw_text in test_data.items():
        t0 = time.time()

        # Normalization and frontend phonemization
        norm_text = normalize_text_en(raw_text)
        word_map, phones = build_word_to_symbol_map(raw_text)
        seq = phonemes_to_sequence_en(phones)

        # Prepare tensors
        seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
        src_len = torch.LongTensor([len(seq)]).to(device)
        src_mask = get_mask_from_lengths(src_len, len(seq))

        speaker_tensor = torch.LongTensor([speaker_id]).to(device) if config["model"]["multi_speaker"] else None

        with torch.no_grad():
            encoder_out = model.encoder(seq_tensor, src_mask)
            if model.speaker_emb is not None:
                encoder_out = encoder_out + model.speaker_emb(speaker_tensor).unsqueeze(1).expand(-1, len(seq), -1)

            # Variance Adaptor predictions
            log_d_pred = model.variance_adaptor.duration_predictor(encoder_out, src_mask)
            raw_duration = (torch.exp(log_d_pred) - 1.0).squeeze(0).cpu().numpy()
            durations_rounded = torch.clamp(
                torch.round((torch.exp(log_d_pred) - 1.0) * args.duration_control), min=0
            ).long()

            # Ensure at least 1 frame per symbol for synthesis
            durations_rounded = torch.clamp(durations_rounded, min=1)
            duration_rounded_arr = durations_rounded.squeeze(0).cpu().numpy()
            duration_ms = duration_rounded_arr * hop_length / sr * 1000.0

            pitch_norm, pitch_emb = model.variance_adaptor.get_pitch_embedding(
                encoder_out, None, src_mask, args.pitch_control
            )
            energy_norm, energy_emb = model.variance_adaptor.get_energy_embedding(
                encoder_out + pitch_emb, None, src_mask, args.energy_control
            )

            pitch_norm_arr = pitch_norm.squeeze(0).cpu().numpy()
            energy_norm_arr = energy_norm.squeeze(0).cpu().numpy()

            # Denormalize
            pitch_hz = np.where(pitch_norm_arr > -2.5, pitch_norm_arr * pitch_std + pitch_mean, 0.0)
            pitch_hz = np.clip(pitch_hz, 0.0, 600.0)
            energy_val = np.maximum(0.0, energy_norm_arr * energy_std + energy_mean)

            # Length regulation to frame-level
            mel_len = torch.sum(durations_rounded, dim=-1)
            mel_mask = get_mask_from_lengths(mel_len)

            adapted_x, _ = model.variance_adaptor.length_regulator(
                encoder_out + pitch_emb + energy_emb, durations_rounded
            )
            expanded_pitch_norm, _ = model.variance_adaptor.length_regulator(
                pitch_norm.unsqueeze(-1), durations_rounded
            )
            expanded_energy_norm, _ = model.variance_adaptor.length_regulator(
                energy_norm.unsqueeze(-1), durations_rounded
            )

            expanded_pitch_norm = expanded_pitch_norm.squeeze(-1).squeeze(0).cpu().numpy()
            expanded_energy_norm = expanded_energy_norm.squeeze(-1).squeeze(0).cpu().numpy()

            expanded_pitch_hz = np.where(expanded_pitch_norm > -2.5, expanded_pitch_norm * pitch_std + pitch_mean, 0.0)
            expanded_pitch_hz = np.clip(expanded_pitch_hz, 0.0, 600.0)
            expanded_energy_val = np.maximum(0.0, expanded_energy_norm * energy_std + energy_mean)

            # Decoder & Postnet
            decoder_out, _ = model.decoder(adapted_x, mel_mask)
            mel_linear = model.mel_linear(decoder_out)
            mel_postnet = model.postnet(mel_linear) + mel_linear

            # HiFi-GAN synthesis
            mel_tensor = mel_postnet.transpose(1, 2)
            wav = vocoder(mel_tensor).squeeze(1).squeeze(0).cpu().numpy()

        elapsed = time.time() - t0
        audio_dur = len(wav) / sr
        rtf = elapsed / max(1e-4, audio_dur)
        num_frames = mel_postnet.shape[1]

        # 1. Save 16-bit 22050 Hz WAV
        wav_path = os.path.join(out_dir, "test_generated_audios", f"{test_id}.wav")
        # Normalize amplitude to [-0.95, 0.95]
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav_norm = (wav / max_val) * 0.95
        else:
            wav_norm = wav
        sf.write(wav_path, wav_norm, sr, subtype="PCM_16")

        # 2. Save JSON metadata
        json_path = os.path.join(out_dir, "test_generated_audios", f"{test_id}.json")
        meta = {
            "id": test_id,
            "raw_text": raw_text,
            "normalized_text": norm_text,
            "phonemes": phones,
            "symbol_ids": seq,
            "speaker": chosen_speaker,
            "speaker_id": speaker_id,
            "checkpoint": args.checkpoint,
            "step": checkpoint_step,
            "controls": {
                "duration": args.duration_control,
                "pitch": args.pitch_control,
                "energy": args.energy_control,
            },
            "stats": {
                "symbols_count": len(seq),
                "frames_count": num_frames,
                "audio_seconds": audio_dur,
                "rtf": rtf,
            },
            "word_to_symbol_map": word_map,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 3. Save raw arrays to .npz
        npz_path = os.path.join(out_dir, "variance_adaptor_plots", "raw", f"{test_id}.npz")
        np.savez_compressed(
            npz_path,
            phonemes=np.array(phones),
            raw_duration=raw_duration,
            duration_rounded=duration_rounded_arr,
            duration_ms=duration_ms,
            pitch_norm=pitch_norm_arr,
            pitch_hz=pitch_hz,
            energy_norm=energy_norm_arr,
            energy_val=energy_val,
            expanded_pitch_hz=expanded_pitch_hz,
            expanded_energy=expanded_energy_val,
            mel=mel_postnet.squeeze(0).transpose(0, 1).cpu().numpy(),
            checkpoint_path=args.checkpoint,
            checkpoint_step=checkpoint_step,
        )

        print(f"{test_id:<6} | {len(seq):<8} | {num_frames:<8} | {audio_dur:<10.2f} | {rtf:<8.4f}")

    print(f"{'='*70}\n")
    print(f"[English Inference] Completed synthesis for all {len(test_data)} test sentences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="English FastSpeech 2 Inference CLI.")
    parser.add_argument("--config", type=str, default=os.path.join(dir_path, "config.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_json", type=str, default=os.path.join(parent_path, "test_sentences.json"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(parent_path, "Results", "English"))
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--all_speakers_demo", action="store_true", default=False)
    args = parser.parse_args()

    run_inference(args)

