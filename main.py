"""main.py
Main entrypoint for FastSpeech 2 TTS (English & Telugu).

Usage:
  # Synthesize custom text:
  python main.py --lang english --text "Welcome to the FastSpeech 2 speech synthesis system."
  python main.py --lang telugu --text "నమస్కారం, ఇది తెలుగు స్పీచ్ సింథసిస్."

  # Run full evaluation & inference pipeline:
  python main.py --pipeline

  # Run test suite:
  python main.py --test
"""

import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description="FastSpeech 2 TTS Main Program")
    parser.add_argument("--lang", type=str, choices=["english", "telugu"], default="english",
                        help="Language for synthesis ('english' or 'telugu')")
    parser.add_argument("--text", type=str, default=None,
                        help="Custom text string to synthesize")
    parser.add_argument("--out", type=str, default="output.wav",
                        help="Output audio path (default: output.wav)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Custom checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for synthesis ('cpu' or 'cuda')")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--pipeline", action="store_true",
                        help="Run full inference, plotting, and validation pipeline")
    parser.add_argument("--test", action="store_true",
                        help="Run pytest verification suite")
    return parser.parse_args()


def synthesize_custom_text(lang: str, text: str, out_wav: str, checkpoint: str = None,
                           device: str = "cpu", p_ctrl: float = 1.0, e_ctrl: float = 1.0, d_ctrl: float = 1.0):
    import torch
    import yaml
    import json
    import soundfile as sf

    lang_code = lang.lower()
    lang_dir = ROOT_DIR / ("English" if lang_code == "english" else "Telugu")

    # Clear cached modules
    for mod in list(sys.modules.keys()):
        if mod in ("model", "text", "transformer", "utils", "audio", "dataset", "inference", "hifigan") or \
           any(mod.startswith(p + ".") for p in ("model", "text", "transformer", "utils", "audio", "dataset", "inference", "hifigan")):
            del sys.modules[mod]

    sys.path = [p for p in sys.path if not p.endswith("English") and not p.endswith("Telugu")]
    sys.path.insert(0, str(lang_dir))

    from model import FastSpeech2
    from inference import load_vocoder, synthesize_audio

    # Config & model
    with open(lang_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if checkpoint is None:
        cand_best = ROOT_DIR / "checkpoints" / lang_code / "best.pth"
        cand_last = ROOT_DIR / "checkpoints" / lang_code / "last.pth"
        checkpoint = str(cand_best if cand_best.exists() else cand_last)

    dev = torch.device(device)
    model = FastSpeech2(config, config["model"]).to(dev)
    model.eval()

    ckpt = torch.load(checkpoint, map_location=dev)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)

    vocoder = load_vocoder(str(ROOT_DIR / "vocoder"), dev)

    # Phonemization
    if lang_code == "english":
        from inference import normalize_text_en, build_word_to_symbol_map
        from text import phonemes_to_sequence_en
        norm_text = normalize_text_en(text)
        _, phones = build_word_to_symbol_map(text)
        seq = phonemes_to_sequence_en(phones)
    else:
        from inference import normalize_text_te, build_word_to_symbol_map
        from text import phonemes_to_sequence_te
        norm_text = normalize_text_te(text)
        _, phones = build_word_to_symbol_map(text)
        seq = phonemes_to_sequence_te(phones)

    # Tensor preparation
    from utils.tools import get_mask_from_lengths
    seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(dev)
    src_len = torch.LongTensor([len(seq)]).to(dev)
    src_mask = get_mask_from_lengths(src_len, len(seq))
    speaker_tensor = torch.LongTensor([0]).to(dev) if config["model"]["multi_speaker"] else None

    with torch.no_grad():
        enc_out = model.encoder(seq_tensor, src_mask)
        if speaker_tensor is not None and model.speaker_emb is not None:
            enc_out = enc_out + model.speaker_emb(speaker_tensor).unsqueeze(1).expand(-1, len(seq), -1)

        d_pred = model.variance_adaptor.duration_predictor(enc_out, src_mask)
        d_round = torch.clamp((torch.round(torch.exp(d_pred) - 1) * d_ctrl), min=0)
        if d_round.sum() == 0:
            d_round = torch.clamp(d_round, min=1)

        x_adapt, mel_lens = model.variance_adaptor.length_regulator(enc_out, d_round)
        mel_mask = get_mask_from_lengths(mel_lens)

        p_pred, p_emb = model.variance_adaptor.get_pitch_embedding(enc_out, None, src_mask, p_ctrl)
        e_pred, e_emb = model.variance_adaptor.get_energy_embedding(enc_out, None, src_mask, e_ctrl)
        x_adapt = x_adapt + model.variance_adaptor.length_regulator.expand(p_emb[0], d_round[0]).unsqueeze(0)
        x_adapt = x_adapt + model.variance_adaptor.length_regulator.expand(e_emb[0], d_round[0]).unsqueeze(0)

        dec_out, _ = model.decoder(x_adapt, mel_mask)
        mel_pred = model.mel_linear(dec_out)
        postnet_mel = model.postnet(mel_pred) + mel_pred

        audio = synthesize_audio(postnet_mel[0].T.cpu().numpy(), vocoder, dev)

    out_path = Path(out_wav)
    sf.write(str(out_path), audio, 22050, subtype="PCM_16")

    print("\n" + "=" * 60)
    print(f"[{lang.capitalize()} Synthesis Successful]")
    print(f"Raw Input:        {text}")
    print(f"Normalized:       {norm_text}")
    print(f"Phonemes ({len(phones)}):  {' '.join(phones[:12])}{'...' if len(phones) > 12 else ''}")
    print(f"Audio Duration:   {len(audio)/22050:.2f} seconds")
    print(f"Output File:      {out_path.resolve()}")
    print("=" * 60 + "\n")
    return out_path


def main():
    args = parse_args()

    if args.test:
        import subprocess
        print("[*] Running full test suite...")
        res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
        sys.exit(res.returncode)

    if args.pipeline:
        from tools.run_pipeline import orchestrate_language
        print("[*] Running end-to-end inference and evaluation pipeline...")
        class PipelineArgs:
            checkpoint = args.checkpoint
            device = args.device
            speaker = None
            pitch_control = args.pitch_control
            energy_control = args.energy_control
            duration_control = args.duration_control
            provisional = True
        orchestrate_language("english", PipelineArgs())
        orchestrate_language("telugu", PipelineArgs())
        return

    # Direct speech synthesis
    if args.text is not None:
        synthesize_custom_text(
            lang=args.lang,
            text=args.text,
            out_wav=args.out,
            checkpoint=args.checkpoint,
            device=args.device,
            p_ctrl=args.pitch_control,
            e_ctrl=args.energy_control,
            d_ctrl=args.duration_control,
        )
    else:
        # Default showcase if no text provided: synthesize test sentence 1
        default_en = "FastSpeech 2 fast and high quality text to speech synthesis."
        print(f"[*] No custom text provided. Synthesizing default demonstration sentence in English:")
        synthesize_custom_text(lang="english", text=default_en, out_wav="output_demo.wav")


if __name__ == "__main__":
    main()
