"""Unit tests for FastSpeech 2 inference, variance plots, and quantitative outputs (Phase 5)."""

import os
import json
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "Results"


@pytest.mark.parametrize("lang,test_ids", [
    ("English", ["E1", "E2a", "E2b", "E3", "E4"]),
    ("Telugu", ["T1", "T2a", "T2b", "T3", "T4"]),
])
def test_audio_properties(lang, test_ids):
    """Assert all 5 wavs per language exist, are non-silent, non-NaN, and of sane length (0.5s - 20s)."""
    lang_dir = RESULTS_DIR / lang / "test_generated_audios"
    assert lang_dir.exists(), f"Audio directory missing: {lang_dir}"

    for tid in test_ids:
        wav_path = lang_dir / f"{tid}.wav"
        assert wav_path.exists(), f"WAV file missing: {wav_path}"

        audio, sr = sf.read(str(wav_path))
        assert sr == 22050, f"Expected 22050 Hz sampling rate, got {sr}"

        # Assert no NaN or Inf
        assert not np.isnan(audio).any(), f"Audio {wav_path.name} contains NaN values"
        assert not np.isinf(audio).any(), f"Audio {wav_path.name} contains Inf values"

        # Assert sane duration (roughly 0.5s to 20.0s, allowing short sentences >= 0.45s)
        dur_sec = len(audio) / sr
        assert 0.45 <= dur_sec <= 20.0, f"Duration {dur_sec:.2f}s for {tid} is outside sane range [0.45, 20.0]"

        # Assert non-silent
        rms = np.sqrt(np.mean(audio ** 2))
        assert rms > 1e-4, f"Audio {wav_path.name} is silent (RMS = {rms:.2e} <= 1e-4)"


@pytest.mark.parametrize("lang,test_ids", [
    ("English", ["E1", "E2a", "E2b", "E3", "E4"]),
    ("Telugu", ["T1", "T2a", "T2b", "T3", "T4"]),
])
def test_variance_plots_exist(lang, test_ids):
    """Assert all 15 variance plots (5 duration, 5 pitch, 5 energy) exist and are non-empty."""
    base_plot_dir = RESULTS_DIR / lang / "variance_adaptor_plots"
    for feature in ["duration", "pitch", "energy"]:
        fdir = base_plot_dir / feature
        assert fdir.exists(), f"Plot sub-directory missing: {fdir}"
        for tid in test_ids:
            pfile = fdir / f"{tid}.png"
            assert pfile.exists(), f"Missing variance plot: {pfile}"
            assert pfile.stat().st_size > 1024, f"Plot file {pfile} is suspiciously small (< 1KB)"


def test_comparison_plots_exist():
    """Assert comparative phonetic analysis plots exist for both languages."""
    en_comp = RESULTS_DIR / "English" / "variance_adaptor_plots" / "comparisons" / "conduct_E2a_vs_E2b.png"
    te_comp = RESULTS_DIR / "Telugu" / "variance_adaptor_plots" / "comparisons" / "pattu_T2a_vs_T2b.png"

    assert en_comp.exists(), f"Missing English comparison plot: {en_comp}"
    assert en_comp.stat().st_size > 1024, "English comparison plot is empty or corrupt"

    assert te_comp.exists(), f"Missing Telugu comparison plot: {te_comp}"
    assert te_comp.stat().st_size > 1024, "Telugu comparison plot is empty or corrupt"


@pytest.mark.parametrize("lang,test_ids", [
    ("English", ["E1", "E2a", "E2b", "E3", "E4"]),
    ("Telugu", ["T1", "T2a", "T2b", "T3", "T4"]),
])
def test_npz_array_shapes_and_consistency(lang, test_ids):
    """Assert raw npz shapes agree:
    len(duration) == len(pitch) == len(energy) == #symbols
    sum(duration) == mel frames
    """
    raw_dir = RESULTS_DIR / lang / "variance_adaptor_plots" / "raw"
    assert raw_dir.exists(), f"Raw npz directory missing: {raw_dir}"

    for tid in test_ids:
        npz_path = raw_dir / f"{tid}.npz"
        assert npz_path.exists(), f"Missing raw npz: {npz_path}"

        d = np.load(str(npz_path))
        required_keys = ["phonemes", "duration_rounded", "pitch_hz", "energy_val", "mel"]
        for rk in required_keys:
            assert rk in d, f"Key '{rk}' missing from {npz_path.name}"

        phones = d["phonemes"]
        dur = d["duration_rounded"]
        pitch = d["pitch_hz"]
        energy = d["energy_val"]
        mel = d["mel"]

        n_syms = len(phones)
        assert len(dur) == n_syms, f"Duration length {len(dur)} != symbols {n_syms} for {tid}"
        assert len(pitch) == n_syms, f"Pitch length {len(pitch)} != symbols {n_syms} for {tid}"
        assert len(energy) == n_syms, f"Energy length {len(energy)} != symbols {n_syms} for {tid}"

        # In saved npz, mel is shape (n_mels, T_mel) = (80, T_mel)
        mel_frames = mel.shape[1] if mel.ndim == 2 and mel.shape[0] == 80 else mel.shape[0]
        assert np.sum(dur) == mel_frames, (
            f"sum(duration)={np.sum(dur)} != mel frames={mel_frames} for {tid}"
        )


@pytest.mark.parametrize("lang", ["English", "Telugu"])
def test_pipeline_reports_and_manifest_exist(lang):
    """Assert sentence_stats and val_metrics reports and manifest are present and valid."""
    ldir = RESULTS_DIR / lang
    
    # Reports
    assert (ldir / "sentence_stats.json").exists(), f"Missing sentence_stats.json for {lang}"
    assert (ldir / "sentence_stats.md").exists(), f"Missing sentence_stats.md for {lang}"
    assert (ldir / "val_metrics.json").exists(), f"Missing val_metrics.json for {lang}"
    assert (ldir / "val_metrics.md").exists(), f"Missing val_metrics.md for {lang}"

    # Manifest
    manifest_path = ldir / "inference_manifest.json"
    assert manifest_path.exists(), f"Missing inference_manifest.json for {lang}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["language"] == lang
    assert "checkpoint" in m
    assert "commands" in m

