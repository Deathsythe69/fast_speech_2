"""Plotting script for FastSpeech 2 variance adaptor predictions (Phase 5).
Generates high-resolution (200 DPI) duration, pitch, and energy figures per test sentence,
along with focused comparison figures for English ('conduct') and Telugu ('పట్టు').
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


def setup_matplotlib_fonts(lang="english"):
    """Finds and registers fonts with complete Telugu and Latin glyph coverage."""
    windir = os.environ.get("WINDIR", "C:\\Windows")
    nirmala_path = os.path.join(windir, "Fonts", "Nirmala.ttc")
    noto_path = os.path.join(workspace_root, "assets", "fonts", "NotoSansTelugu-Regular.ttf")

    font_family = ["DejaVu Sans", "Segoe UI", "Arial"]

    if os.path.exists(noto_path):
        try:
            fe = fm.FontEntry(fname=noto_path, name="NotoSansTelugu")
            fm.fontManager.ttflist.insert(0, fe)
            font_family.insert(0, "NotoSansTelugu")
        except Exception:
            pass

    if os.path.exists(nirmala_path):
        try:
            fe = fm.FontEntry(fname=nirmala_path, name="NirmalaUI")
            fm.fontManager.ttflist.insert(0, fe)
            font_family.insert(0, "NirmalaUI")
        except Exception:
            pass

    plt.rcParams["font.sans-serif"] = font_family
    plt.rcParams["axes.unicode_minus"] = False


def plot_single_sentence(npz_path, json_path, out_base_dir, lang):
    """Generates duration, pitch, and energy 2-panel plots for a single test utterance."""
    data = np.load(npz_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    test_id = meta["id"]
    raw_text = meta["raw_text"]
    step = meta.get("step", 0)

    phones = data["phonemes"]
    dur_frames = data["duration_rounded"]
    dur_ms = data["duration_ms"]
    pitch_hz = data["pitch_hz"]
    energy_val = data["energy_val"]
    expanded_pitch = data["expanded_pitch_hz"]
    expanded_energy = data["expanded_energy"]
    mel = data["mel"]  # (80, T)

    word_map = meta.get("word_to_symbol_map", [])

    N_symbols = len(phones)
    T_mel = mel.shape[1]
    sr = 22050
    hop_length = 256
    time_frames = np.arange(T_mel) * (hop_length / sr)
    total_time_s = T_mel * hop_length / sr

    x_indices = np.arange(N_symbols)
    sil_set = {"sp", "spn", "sil", ""}

    def add_word_boundaries(ax, y_pos):
        for w_info in word_map:
            s_idx = w_info["start_symbol_idx"]
            e_idx = w_info["end_symbol_idx"]
            if s_idx < N_symbols:
                ax.axvline(x=s_idx - 0.5, color="gray", linestyle=":", alpha=0.6, linewidth=1.0)
                mid_x = (s_idx + min(e_idx, N_symbols) - 1) / 2.0
                ax.text(
                    mid_x,
                    y_pos,
                    w_info["word"],
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#2c3e50",
                    weight="bold",
                    clip_on=True,
                )

    fig_width = max(14, int(N_symbols * 0.28))

    # =========================================================================
    # 1. DURATION PLOT
    # =========================================================================
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fig_width, 7.5), gridspec_kw={"height_ratios": [1.1, 1.0]}, dpi=200
    )

    bars = ax_top.bar(x_indices, dur_ms, color="#3498db", edgecolor="#2980b9", width=0.75, alpha=0.85)
    max_ms = np.max(dur_ms) if len(dur_ms) else 100
    for idx, (b, f_count) in enumerate(zip(bars, dur_frames)):
        ax_top.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + (max_ms * 0.02),
            f"{int(f_count)}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#2c3e50",
        )

    ax_top.set_xticks(x_indices)
    ax_top.set_xticklabels(phones, rotation=60, fontsize=8, ha="right")
    ax_top.set_ylabel("Duration (ms)", fontsize=10, weight="semibold")
    ax_top.set_ylim(0, max_ms * 1.25)
    add_word_boundaries(ax_top, max_ms * 1.1)
    ax_top.set_title(
        f"{lang.capitalize()} | {test_id} | Duration Predictor | checkpoint step {step}\n\"{raw_text}\"",
        fontsize=11,
        weight="bold",
        pad=12,
    )
    ax_top.grid(axis="y", linestyle="--", alpha=0.4)

    im = ax_bot.imshow(
        mel,
        aspect="auto",
        origin="lower",
        extent=[0, total_time_s, 0, 80],
        cmap="viridis",
    )
    cum_frames = np.cumsum(dur_frames)
    cum_times = cum_frames * (hop_length / sr)
    for ct in cum_times[:-1]:
        ax_bot.axvline(x=ct, color="white", linestyle="--", alpha=0.5, linewidth=0.75)

    ax_bot.set_xlabel("Time (seconds)", fontsize=10, weight="semibold")
    ax_bot.set_ylabel("Mel Channels (80)", fontsize=10, weight="semibold")
    ax_bot.set_xlim(0, total_time_s)

    plt.tight_layout()
    dur_out = os.path.join(out_base_dir, "variance_adaptor_plots", "duration", f"{test_id}.png")
    os.makedirs(os.path.dirname(dur_out), exist_ok=True)
    plt.savefig(dur_out, dpi=200)
    plt.close(fig)

    # =========================================================================
    # 2. PITCH PLOT
    # =========================================================================
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fig_width, 7.5), gridspec_kw={"height_ratios": [1.1, 1.0]}, dpi=200
    )

    voiced_mask = (pitch_hz > 10.0) & np.array([p not in sil_set for p in phones])
    max_p = np.max(pitch_hz[voiced_mask]) if np.any(voiced_mask) else 300.0
    min_p = np.min(pitch_hz[voiced_mask]) if np.any(voiced_mask) else 100.0
    y_floor = max(0, min_p - 40)
    y_ceil = max_p + 50

    ax_top.plot(x_indices[voiced_mask], pitch_hz[voiced_mask], color="#e74c3c", linewidth=1.5, linestyle="-", alpha=0.7)
    ax_top.scatter(
        x_indices[voiced_mask],
        pitch_hz[voiced_mask],
        color="#c0392b",
        s=40,
        zorder=3,
        label="Voiced (Hz)",
    )
    unvoiced_mask = ~voiced_mask
    ax_top.scatter(
        x_indices[unvoiced_mask],
        np.full(np.sum(unvoiced_mask), y_floor + 15),
        color="#95a5a6",
        facecolors="none",
        marker="s",
        s=30,
        label="Unvoiced / Silence",
        alpha=0.7,
    )

    ax_top.set_xticks(x_indices)
    ax_top.set_xticklabels(phones, rotation=60, fontsize=8, ha="right")
    ax_top.set_ylabel("Fundamental Frequency F0 (Hz)", fontsize=10, weight="semibold")
    ax_top.set_ylim(y_floor, y_ceil)
    add_word_boundaries(ax_top, y_ceil - (y_ceil - y_floor) * 0.1)
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.set_title(
        f"{lang.capitalize()} | {test_id} | Pitch Predictor | checkpoint step {step}\n\"{raw_text}\"",
        fontsize=11,
        weight="bold",
        pad=12,
    )
    ax_top.grid(axis="y", linestyle="--", alpha=0.4)

    ax_bot.imshow(mel, aspect="auto", origin="lower", extent=[0, total_time_s, 0, 80], cmap="viridis")
    ax_bot_twin = ax_bot.twinx()
    voiced_frame_mask = expanded_pitch > 10.0
    pitch_masked = np.where(voiced_frame_mask, expanded_pitch, np.nan)
    ax_bot_twin.plot(time_frames, pitch_masked, color="#f39c12", linewidth=2.0, label="F0 Contour")
    ax_bot_twin.set_ylabel("F0 (Hz)", color="#f39c12", fontsize=10, weight="semibold")
    ax_bot_twin.tick_params(axis="y", labelcolor="#f39c12")
    ax_bot_twin.set_ylim(y_floor, y_ceil)

    ax_bot.set_xlabel("Time (seconds)", fontsize=10, weight="semibold")
    ax_bot.set_ylabel("Mel Channels (80)", fontsize=10, weight="semibold")
    ax_bot.set_xlim(0, total_time_s)

    plt.tight_layout()
    pitch_out = os.path.join(out_base_dir, "variance_adaptor_plots", "pitch", f"{test_id}.png")
    os.makedirs(os.path.dirname(pitch_out), exist_ok=True)
    plt.savefig(pitch_out, dpi=200)
    plt.close(fig)

    # =========================================================================
    # 3. ENERGY PLOT
    # =========================================================================
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fig_width, 7.5), gridspec_kw={"height_ratios": [1.1, 1.0]}, dpi=200
    )

    max_e = np.max(energy_val) if len(energy_val) else 50.0
    ax_top.bar(x_indices, energy_val, color="#2ecc71", edgecolor="#27ae60", width=0.75, alpha=0.85)
    ax_top.set_xticks(x_indices)
    ax_top.set_xticklabels(phones, rotation=60, fontsize=8, ha="right")
    ax_top.set_ylabel("Energy (L2 Norm)", fontsize=10, weight="semibold")
    ax_top.set_ylim(0, max_e * 1.25)
    add_word_boundaries(ax_top, max_e * 1.1)
    ax_top.set_title(
        f"{lang.capitalize()} | {test_id} | Energy Predictor | checkpoint step {step}\n\"{raw_text}\"",
        fontsize=11,
        weight="bold",
        pad=12,
    )
    ax_top.grid(axis="y", linestyle="--", alpha=0.4)

    ax_bot.imshow(mel, aspect="auto", origin="lower", extent=[0, total_time_s, 0, 80], cmap="viridis")
    ax_bot_twin = ax_bot.twinx()
    ax_bot_twin.plot(time_frames, expanded_energy, color="#2ecc71", linewidth=2.0, label="Energy Curve")
    ax_bot_twin.set_ylabel("Energy", color="#27ae60", fontsize=10, weight="semibold")
    ax_bot_twin.tick_params(axis="y", labelcolor="#27ae60")
    ax_bot_twin.set_ylim(0, np.max(expanded_energy) * 1.25 if len(expanded_energy) else 60.0)

    ax_bot.set_xlabel("Time (seconds)", fontsize=10, weight="semibold")
    ax_bot.set_ylabel("Mel Channels (80)", fontsize=10, weight="semibold")
    ax_bot.set_xlim(0, total_time_s)

    plt.tight_layout()
    energy_out = os.path.join(out_base_dir, "variance_adaptor_plots", "energy", f"{test_id}.png")
    os.makedirs(os.path.dirname(energy_out), exist_ok=True)
    plt.savefig(energy_out, dpi=200)
    plt.close(fig)


def plot_conduct_comparison(out_base_dir):
    """Generates side-by-side comparison figure for English E2a vs E2b focused on 'conduct'."""
    e2a_npz = os.path.join(out_base_dir, "variance_adaptor_plots", "raw", "E2a.npz")
    e2b_npz = os.path.join(out_base_dir, "variance_adaptor_plots", "raw", "E2b.npz")
    e2a_json = os.path.join(out_base_dir, "test_generated_audios", "E2a.json")
    e2b_json = os.path.join(out_base_dir, "test_generated_audios", "E2b.json")

    if not (os.path.exists(e2a_npz) and os.path.exists(e2b_npz)):
        print("[Comparison] E2a/E2b raw npz missing; skipping conduct comparison.")
        return

    d_a = np.load(e2a_npz)
    d_b = np.load(e2b_npz)
    with open(e2a_json, "r", encoding="utf-8") as f:
        m_a = json.load(f)
    with open(e2b_json, "r", encoding="utf-8") as f:
        m_b = json.load(f)

    def find_conduct(meta, data):
        for w_info in meta.get("word_to_symbol_map", []):
            if "conduct" in w_info["word"].lower():
                s = w_info["start_symbol_idx"]
                e = w_info["end_symbol_idx"]
                return {
                    "phones": list(data["phonemes"][s:e]),
                    "dur_ms": list(data["duration_ms"][s:e]),
                    "pitch_hz": list(data["pitch_hz"][s:e]),
                    "energy": list(data["energy_val"][s:e]),
                }
        p_list = list(data["phonemes"])
        for i in range(len(p_list) - 6):
            if p_list[i] == "K" and p_list[i+6] == "T":
                return {
                    "phones": list(data["phonemes"][i:i+7]),
                    "dur_ms": list(data["duration_ms"][i:i+7]),
                    "pitch_hz": list(data["pitch_hz"][i:i+7]),
                    "energy": list(data["energy_val"][i:i+7]),
                }
        return None

    c_a = find_conduct(m_a, d_a)
    c_b = find_conduct(m_b, d_b)

    if not (c_a and c_b):
        print("[Comparison] Could not isolate 'conduct' symbols.")
        return

    fig, (ax_dur, ax_pitch, ax_energy) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)

    n_phones = min(len(c_a["phones"]), len(c_b["phones"]))
    x = np.arange(n_phones)
    width = 0.35

    ax_dur.bar(x - width/2, c_a["dur_ms"][:n_phones], width, label="E2a (Noun: /K AA1 N D AH0 K T/)", color="#3498db")
    ax_dur.bar(x + width/2, c_b["dur_ms"][:n_phones], width, label="E2b (Verb: /K AH0 N D AH1 K T/)", color="#e67e22")
    ax_dur.set_xticks(x)
    labels = [f"E2a: {p_a}\nE2b: {p_b}" for p_a, p_b in zip(c_a["phones"][:n_phones], c_b["phones"][:n_phones])]
    ax_dur.set_xticklabels(labels, fontsize=8)
    ax_dur.set_ylabel("Duration (ms)", weight="semibold")
    ax_dur.set_title("Duration: 'conduct' (Noun vs Verb)", weight="bold", fontsize=10)
    ax_dur.legend(fontsize=8)
    ax_dur.grid(axis="y", linestyle="--", alpha=0.4)

    ax_pitch.plot(x, c_a["pitch_hz"][:n_phones], "o-", label="E2a Noun", color="#3498db", linewidth=2)
    ax_pitch.plot(x, c_b["pitch_hz"][:n_phones], "s-", label="E2b Verb", color="#e67e22", linewidth=2)
    ax_pitch.set_xticks(x)
    ax_pitch.set_xticklabels(labels, fontsize=8)
    ax_pitch.set_ylabel("F0 (Hz)", weight="semibold")
    ax_pitch.set_title("Pitch F0: 'conduct' (Stress Shift)", weight="bold", fontsize=10)
    ax_pitch.legend(fontsize=8)
    ax_pitch.grid(axis="y", linestyle="--", alpha=0.4)

    ax_energy.bar(x - width/2, c_a["energy"][:n_phones], width, label="E2a Noun", color="#3498db")
    ax_energy.bar(x + width/2, c_b["energy"][:n_phones], width, label="E2b Verb", color="#e67e22")
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(labels, fontsize=8)
    ax_energy.set_ylabel("Energy (L2 Norm)", weight="semibold")
    ax_energy.set_title("Energy: 'conduct'", weight="bold", fontsize=10)
    ax_energy.legend(fontsize=8)
    ax_energy.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("English Heteronym Analysis: 'conduct' in E2a (Noun) vs E2b (Verb)", fontsize=12, weight="bold")
    plt.tight_layout()

    out_comp = os.path.join(out_base_dir, "variance_adaptor_plots", "comparisons", "conduct_E2a_vs_E2b.png")
    os.makedirs(os.path.dirname(out_comp), exist_ok=True)
    plt.savefig(out_comp, dpi=200)
    plt.close(fig)
    print(f"[English Comparison] Saved conduct comparison figure to {out_comp}")


def plot_pattu_comparison(out_base_dir):
    """Generates side-by-side comparison figure for Telugu T2a vs T2b focused on root word 'పట్టు'."""
    t2a_npz = os.path.join(out_base_dir, "variance_adaptor_plots", "raw", "T2a.npz")
    t2b_npz = os.path.join(out_base_dir, "variance_adaptor_plots", "raw", "T2b.npz")
    t2a_json = os.path.join(out_base_dir, "test_generated_audios", "T2a.json")
    t2b_json = os.path.join(out_base_dir, "test_generated_audios", "T2b.json")

    if not (os.path.exists(t2a_npz) and os.path.exists(t2b_npz)):
        print("[Comparison] T2a/T2b raw npz missing; skipping pattu comparison.")
        return

    d_a = np.load(t2a_npz)
    d_b = np.load(t2b_npz)
    with open(t2a_json, "r", encoding="utf-8") as f:
        m_a = json.load(f)
    with open(t2b_json, "r", encoding="utf-8") as f:
        m_b = json.load(f)

    # In T2a, the word is 'పట్టు' (5 phones: p, a, t_, t_, u).
    # In T2b, the word is 'పట్టుచీరను' (compound word). We isolate the prefix root 'పట్టు' (first 5 phones).
    def find_pattu(meta, data):
        for w_info in meta.get("word_to_symbol_map", []):
            if "పట్టు" in w_info["word"]:
                s = w_info["start_symbol_idx"]
                # Root 'పట్టు' consists of 5 phones: p, a, t_, t_, u
                e = s + 5
                return {
                    "phones": list(data["phonemes"][s:e]),
                    "dur_ms": list(data["duration_ms"][s:e]),
                    "pitch_hz": list(data["pitch_hz"][s:e]),
                    "energy": list(data["energy_val"][s:e]),
                }
        p_list = list(data["phonemes"])
        for i in range(len(p_list) - 4):
            if p_list[i] == "p" and p_list[i+4] == "u":
                return {
                    "phones": list(data["phonemes"][i:i+5]),
                    "dur_ms": list(data["duration_ms"][i:i+5]),
                    "pitch_hz": list(data["pitch_hz"][i:i+5]),
                    "energy": list(data["energy_val"][i:i+5]),
                }
        return None

    c_a = find_pattu(m_a, d_a)
    c_b = find_pattu(m_b, d_b)

    if not (c_a and c_b):
        print("[Comparison] Could not isolate 'పట్టు' symbols.")
        return

    fig, (ax_dur, ax_pitch, ax_energy) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)

    n_phones = min(len(c_a["phones"]), len(c_b["phones"]))
    x = np.arange(n_phones)
    width = 0.35

    ax_dur.bar(x - width/2, c_a["dur_ms"][:n_phones], width, label="T2a (Grip: పట్టు)", color="#9b59b6")
    ax_dur.bar(x + width/2, c_b["dur_ms"][:n_phones], width, label="T2b (Silk: పట్టుచీరను)", color="#1abc9c")
    ax_dur.set_xticks(x)
    ax_dur.set_xticklabels(c_a["phones"][:n_phones], fontsize=10, weight="bold")
    ax_dur.set_ylabel("Duration (ms)", weight="semibold")
    ax_dur.set_title("Duration: 'పట్టు' (Grip vs Silk)", weight="bold", fontsize=10)
    ax_dur.legend(fontsize=8)
    ax_dur.grid(axis="y", linestyle="--", alpha=0.4)

    ax_pitch.plot(x, c_a["pitch_hz"][:n_phones], "o-", label="T2a Grip", color="#9b59b6", linewidth=2)
    ax_pitch.plot(x, c_b["pitch_hz"][:n_phones], "s-", label="T2b Silk", color="#1abc9c", linewidth=2)
    ax_pitch.set_xticks(x)
    ax_pitch.set_xticklabels(c_a["phones"][:n_phones], fontsize=10, weight="bold")
    ax_pitch.set_ylabel("F0 (Hz)", weight="semibold")
    ax_pitch.set_title("Pitch F0: 'పట్టు' (Context Variation)", weight="bold", fontsize=10)
    ax_pitch.legend(fontsize=8)
    ax_pitch.grid(axis="y", linestyle="--", alpha=0.4)

    ax_energy.bar(x - width/2, c_a["energy"][:n_phones], width, label="T2a Grip", color="#9b59b6")
    ax_energy.bar(x + width/2, c_b["energy"][:n_phones], width, label="T2b Silk", color="#1abc9c")
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(c_a["phones"][:n_phones], fontsize=10, weight="bold")
    ax_energy.set_ylabel("Energy (L2 Norm)", weight="semibold")
    ax_energy.set_title("Energy: 'పట్టు'", weight="bold", fontsize=10)
    ax_energy.legend(fontsize=8)
    ax_energy.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Telugu Homograph Analysis: 'పట్టు' in T2a (Grip) vs T2b (Silk)", fontsize=12, weight="bold")
    plt.tight_layout()

    out_comp = os.path.join(out_base_dir, "variance_adaptor_plots", "comparisons", "pattu_T2a_vs_T2b.png")
    os.makedirs(os.path.dirname(out_comp), exist_ok=True)
    plt.savefig(out_comp, dpi=200)
    plt.close(fig)
    print(f"[Telugu Comparison] Saved pattu comparison figure to {out_comp}")


def run_plot_variance(lang, out_base_dir=None):
    setup_matplotlib_fonts(lang)

    if out_base_dir is None:
        out_base_dir = os.path.join(workspace_root, "Results", lang.capitalize())

    raw_dir = os.path.join(out_base_dir, "variance_adaptor_plots", "raw")
    npz_files = glob.glob(os.path.join(raw_dir, "*.npz"))

    if not npz_files:
        print(f"[Plotting] No .npz files found in {raw_dir}")
        return

    print(f"[{lang.capitalize()} Plotting] Generating variance adaptor plots for {len(npz_files)} utterances...")
    for npz_path in npz_files:
        test_id = os.path.splitext(os.path.basename(npz_path))[0]
        json_path = os.path.join(out_base_dir, "test_generated_audios", f"{test_id}.json")
        plot_single_sentence(npz_path, json_path, out_base_dir, lang)

    if lang.lower() == "english":
        plot_conduct_comparison(out_base_dir)
    elif lang.lower() == "telugu":
        plot_pattu_comparison(out_base_dir)

    print(f"[{lang.capitalize()} Plotting] All figures successfully written to {out_base_dir}/variance_adaptor_plots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variance Adaptor Plotting Tool.")
    parser.add_argument("--lang", choices=["english", "telugu"], required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    run_plot_variance(args.lang, args.out_dir)
