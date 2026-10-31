"""Unit tests for FastSpeech 2 preprocessing and feature extraction (Phase 3)."""

import json
import os

import numpy as np
import pytest
import soundfile as sf
import tgt

from English.preprocessor.preprocessor import Preprocessor as EnPreprocessor


@pytest.fixture
def dummy_utterance(tmp_path):
    """Creates a synthetic sine-wave utterance with F0=200 Hz and a matching TextGrid."""
    sr = 22050
    dur_sec = 1.0
    t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)
    # 200 Hz harmonic tone
    audio = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.25 * np.sin(2 * np.pi * 400 * t)

    speaker = "test_spk"
    basename = "test_utt_01"

    spk_dir = tmp_path / "raw" / speaker
    spk_dir.mkdir(parents=True)
    wav_path = str(spk_dir / f"{basename}.wav")
    lab_path = str(spk_dir / f"{basename}.lab")
    sf.write(wav_path, audio, sr)

    with open(lab_path, "w", encoding="utf-8") as f:
        f.write("test speech")

    # Matching TextGrid with 3 phones
    tg_dir = tmp_path / "preprocessed" / "TextGrid" / speaker
    tg_dir.mkdir(parents=True)
    tg_path = str(tg_dir / f"{basename}.TextGrid")

    tg = tgt.TextGrid()
    tier = tgt.IntervalTier(start_time=0.0, end_time=dur_sec, name="phones")
    tier.add_interval(tgt.Interval(0.0, 0.3, "T"))
    tier.add_interval(tgt.Interval(0.3, 0.7, "EH1"))
    tier.add_interval(tgt.Interval(0.7, 1.0, "S"))
    tg.add_tier(tier)
    tgt.io.write_to_file(tg, tg_path, format="long")

    config = {
        "dataset": "test",
        "path": {
            "raw_path": str(tmp_path / "raw"),
            "preprocessed_path": str(tmp_path / "preprocessed"),
        },
        "preprocessing": {
            "val_size": 0.2,
            "text": {"text_cleaners": ["english_cleaners"], "language": "en"},
            "audio": {"sampling_rate": 22050, "max_wav_value": 32768.0},
            "stft": {"filter_length": 1024, "hop_length": 256, "win_length": 1024},
            "mel": {"n_mel_channels": 80, "mel_fmin": 0, "mel_fmax": 8000},
            "pitch": {"feature": "frame_level", "normalization": True, "f0_min": 50.0, "f0_max": 500.0},
            "energy": {"feature": "frame_level", "normalization": True},
        },
    }

    return {
        "config": config,
        "speaker": speaker,
        "basename": basename,
        "wav_path": wav_path,
        "tg_path": tg_path,
        "tmp_path": tmp_path,
    }


def test_feature_extraction_shapes_and_validity(dummy_utterance):
    """Assert:
    - mel shape is (80, T)
    - pitch is (T,)
    - energy is (T,)
    - sum(duration) == T
    - pitch has no NaN/inf, voiced frames > 0, F0 in reasonable range (50-500 Hz).
    """
    config = dummy_utterance["config"]
    speaker = dummy_utterance["speaker"]
    basename = dummy_utterance["basename"]
    wav_path = dummy_utterance["wav_path"]
    tg_path = dummy_utterance["tg_path"]

    pre = EnPreprocessor(config)
    os.makedirs(os.path.join(config["path"]["preprocessed_path"], "mel"), exist_ok=True)
    os.makedirs(os.path.join(config["path"]["preprocessed_path"], "pitch"), exist_ok=True)
    os.makedirs(os.path.join(config["path"]["preprocessed_path"], "energy"), exist_ok=True)
    os.makedirs(os.path.join(config["path"]["preprocessed_path"], "duration"), exist_ok=True)

    res = pre.process_utterance(speaker, basename, wav_path=wav_path, tg_path=tg_path)
    assert res is not None, "process_utterance returned None"

    info_str, pitch, energy, T_mel, is_valid = res
    assert is_valid, "Utterance flagged as invalid"

    # Load saved features
    out_dir = config["path"]["preprocessed_path"]
    mel_loaded = np.load(os.path.join(out_dir, "mel", f"{speaker}-mel-{basename}.npy"))  # shape (T, 80)
    pitch_loaded = np.load(os.path.join(out_dir, "pitch", f"{speaker}-pitch-{basename}.npy"))
    energy_loaded = np.load(os.path.join(out_dir, "energy", f"{speaker}-energy-{basename}.npy"))
    duration_loaded = np.load(os.path.join(out_dir, "duration", f"{speaker}-duration-{basename}.npy"))

    # Assert mel shape is (80, T) when transposed (or loaded T, 80)
    assert mel_loaded.shape[1] == 80, f"Expected 80 mel channels, got {mel_loaded.shape[1]}"
    T = mel_loaded.shape[0]

    # Assert pitch and energy shape is (T,)
    assert pitch_loaded.shape == (T,), f"Pitch shape {pitch_loaded.shape} != ({T},)"
    assert energy_loaded.shape == (T,), f"Energy shape {energy_loaded.shape} != ({T},)"

    # Assert sum(duration) == T
    assert np.sum(duration_loaded) == T, f"sum(duration)={np.sum(duration_loaded)} != T={T}"

    # Assert pitch has no NaN/inf, voiced frames > 0, F0 in reasonable range (50-500 Hz)
    assert not np.isnan(pitch_loaded).any(), "Pitch contains NaN"
    assert not np.isinf(pitch_loaded).any(), "Pitch contains Inf"
    voiced = pitch_loaded[pitch_loaded > 0]
    assert len(voiced) > 0, "No voiced frames found"
    assert np.all(voiced >= 50.0) and np.all(voiced <= 500.0), (
        f"Voiced pitch out of range [50, 500]: min={np.min(voiced)}, max={np.max(voiced)}"
    )


def test_val_split_not_used_for_stats_computation(tmp_path):
    """Assert that the validation split is strictly NOT used for stats.json computation."""
    sr = 22050
    dur_sec = 0.5
    t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)

    raw_dir = tmp_path / "raw" / "spk1"
    raw_dir.mkdir(parents=True)
    out_dir = tmp_path / "preprocessed"
    tg_dir = out_dir / "TextGrid" / "spk1"
    tg_dir.mkdir(parents=True)

    # 2 train utterances (200 Hz tone) and 1 val utterance (400 Hz tone)
    utts = [
        ("train_1", 200),
        ("train_2", 200),
        ("val_1", 400),
    ]

    for name, freq in utts:
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        wav_path = str(raw_dir / f"{name}.wav")
        sf.write(wav_path, audio, sr)
        with open(str(raw_dir / f"{name}.lab"), "w") as f:
            f.write("speech")

        tg = tgt.TextGrid()
        tier = tgt.IntervalTier(start_time=0.0, end_time=dur_sec, name="phones")
        tier.add_interval(tgt.Interval(0.0, dur_sec, "AA1"))
        tg.add_tier(tier)
        tgt.io.write_to_file(tg, str(tg_dir / f"{name}.TextGrid"), format="long")

    # Create data/english/train.csv and val.csv
    data_dir = tmp_path / "data" / "english"
    data_dir.mkdir(parents=True)
    with open(data_dir / "train.csv", "w") as f:
        f.write("id,audio,text\ntrain_1,,speech\ntrain_2,,speech\n")
    with open(data_dir / "val.csv", "w") as f:
        f.write("id,audio,text\nval_1,,speech\n")

    config = {
        "dataset": "test",
        "path": {
            "raw_path": str(tmp_path / "raw"),
            "preprocessed_path": str(out_dir),
        },
        "preprocessing": {
            "val_size": 1,
            "text": {"text_cleaners": ["english_cleaners"], "language": "en"},
            "audio": {"sampling_rate": 22050, "max_wav_value": 32768.0},
            "stft": {"filter_length": 1024, "hop_length": 256, "win_length": 1024},
            "mel": {"n_mel_channels": 80, "mel_fmin": 0, "mel_fmax": 8000},
            "pitch": {"feature": "frame_level", "normalization": True, "f0_min": 50.0, "f0_max": 500.0},
            "energy": {"feature": "frame_level", "normalization": True},
        },
    }

    pre = EnPreprocessor(config)
    # Monkey-patch load_train_val_manifest to point to test data_dir
    pre.load_train_val_manifest = lambda: ({"train_1", "train_2"}, {"val_1"})

    pre.build_from_path()

    stats_path = os.path.join(out_dir, "stats.json")
    assert os.path.exists(stats_path)
    with open(stats_path, "r") as f:
        stats = json.load(f)

    pitch_mean = stats["pitch"][2]
    # The train utterances have 200 Hz tone, val utterance has 400 Hz tone.
    # If val was included, mean would be ~266 Hz.
    # If train only, mean is ~200 Hz (+/- 5 Hz).
    assert 190.0 <= pitch_mean <= 210.0, (
        f"Pitch mean {pitch_mean} indicates validation set was leaked into stats computation!"
    )

