"""Preprocessor module for English FastSpeech 2.
Extracts 80-channel mel-spectrograms, PyWorld F0, STFT energy, and phone durations.
Computes normalization statistics strictly on the train split.
"""

import glob
import json
import os
import random
import sys

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

try:
    from English.audio.stft import TacotronSTFT
    from English.audio.tools import get_mel_from_wav
except (ImportError, ValueError):
    from audio.stft import TacotronSTFT
    from audio.tools import get_mel_from_wav


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"].get("raw_path", "raw_data/english")
        self.out_dir = config["path"].get("preprocessed_path", "preprocessed_data/english")
        self.val_size = config["preprocessing"].get("val_size", 0.1)
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def load_train_val_manifest(self):
        """Loads train and val utterance IDs from data/<lang>/train.csv and val.csv if available."""
        train_ids = set()
        val_ids = set()

        train_csv = os.path.join("data", "english", "train.csv")
        val_csv = os.path.join("data", "english", "val.csv")

        if os.path.exists(train_csv):
            with open(train_csv, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                id_idx = 0
                for line in f:
                    parts = line.strip().split(",")
                    if parts and parts[id_idx]:
                        train_ids.add(parts[id_idx].strip())

        if os.path.exists(val_csv):
            with open(val_csv, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                id_idx = 0
                for line in f:
                    parts = line.strip().split(",")
                    if parts and parts[id_idx]:
                        val_ids.add(parts[id_idx].strip())

        return train_ids, val_ids

    def build_from_path(self):
        os.makedirs(os.path.join(self.out_dir, "mel"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "pitch"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "energy"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "duration"), exist_ok=True)

        print("[English] Extracting features...")
        train_ids, val_ids = self.load_train_val_manifest()
        has_csv_split = len(train_ids) > 0

        speakers = {}
        all_records = []
        n_frames = 0

        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Track train-only pitch and energy for min/max
        train_pitch_list = []
        train_energy_list = []

        if not os.path.exists(self.in_dir):
            print(f"[English] Input directory {self.in_dir} does not exist.")
            return []

        speaker_dirs = [d for d in os.listdir(self.in_dir) if os.path.isdir(os.path.join(self.in_dir, d))]
        if not speaker_dirs:
            speaker_dirs = ["default"]

        for i, speaker in enumerate(speaker_dirs):
            speakers[speaker] = i
            spk_dir = os.path.join(self.in_dir, speaker) if speaker != "default" else self.in_dir
            wav_files = glob.glob(os.path.join(spk_dir, "*.wav"))

            for wav_path in tqdm(wav_files, desc=f"Speaker {speaker}"):
                basename = os.path.splitext(os.path.basename(wav_path))[0]
                tg_path = os.path.join(self.out_dir, "TextGrid", speaker, f"{basename}.TextGrid")

                if not os.path.exists(tg_path):
                    continue

                res = self.process_utterance(speaker, basename, wav_path=wav_path, tg_path=tg_path)
                if res is None:
                    continue

                info_str, pitch, energy, n, is_valid = res
                if not is_valid:
                    continue

                all_records.append((basename, speaker, info_str, pitch, energy, n))
                n_frames += n

        if not all_records:
            print("[English] No valid utterances processed.")
            return []

        # Determine train vs val split
        train_records = []
        val_records = []

        if has_csv_split:
            for rec in all_records:
                bname = rec[0]
                if bname in val_ids:
                    val_records.append(rec)
                else:
                    train_records.append(rec)
        else:
            # Fallback split based on val_size
            random.seed(1234)
            shuffled = list(all_records)
            random.shuffle(shuffled)
            val_count = int(len(shuffled) * (self.val_size if self.val_size < 1.0 else (self.val_size / len(shuffled))))
            val_records = shuffled[:val_count]
            train_records = shuffled[val_count:]

        print(f"[English] Split: {len(train_records)} train utterances, {len(val_records)} val utterances.")

        # STRICT REQUIREMENT: Fit pitch and energy scalers ON THE TRAIN SPLIT ONLY!
        for rec in train_records:
            pitch = rec[3]
            energy = rec[4]
            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                train_pitch_list.append(pitch)
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))
                train_energy_list.append(energy)

        pitch_mean = float(pitch_scaler.mean_[0]) if self.pitch_normalization else 0.0
        pitch_std = float(pitch_scaler.scale_[0]) if self.pitch_normalization else 1.0
        energy_mean = float(energy_scaler.mean_[0]) if self.energy_normalization else 0.0
        energy_std = float(energy_scaler.scale_[0]) if self.energy_normalization else 1.0

        # Compute train-only min and max after normalization
        train_pitch_all = (np.concatenate(train_pitch_list) - pitch_mean) / pitch_std if train_pitch_list else np.array([0.0])
        train_energy_all = (np.concatenate(train_energy_list) - energy_mean) / energy_std if train_energy_list else np.array([0.0])

        pitch_min = float(np.min(train_pitch_all))
        pitch_max = float(np.max(train_pitch_all))
        energy_min = float(np.min(train_energy_all))
        energy_max = float(np.max(train_energy_all))

        # Normalize and save pitch and energy files
        for rec in all_records:
            bname, spk = rec[0], rec[1]
            p_file = os.path.join(self.out_dir, "pitch", f"{spk}-pitch-{bname}.npy")
            e_file = os.path.join(self.out_dir, "energy", f"{spk}-energy-{bname}.npy")

            if os.path.exists(p_file):
                p_val = (np.load(p_file) - pitch_mean) / pitch_std
                np.save(p_file, p_val)
            if os.path.exists(e_file):
                e_val = (np.load(e_file) - energy_mean) / energy_std
                np.save(e_file, e_val)

        # Save speakers.json
        with open(os.path.join(self.out_dir, "speakers.json"), "w", encoding="utf-8") as f:
            json.dump(speakers, f, indent=2)

        # Save stats.json (Train split only!)
        stats = {
            "pitch": [pitch_min, pitch_max, pitch_mean, pitch_std],
            "energy": [energy_min, energy_max, energy_mean, energy_std],
        }
        with open(os.path.join(self.out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        # Save train.txt and val.txt
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for rec in train_records:
                f.write(rec[2] + "\n")

        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for rec in val_records:
                f.write(rec[2] + "\n")

        print(f"[English] Successfully finished feature extraction for {len(all_records)} files.")
        return all_records

    def process_utterance(self, speaker, basename, wav_path=None, tg_path=None, raw_text=None):
        """Processes a single utterance: extracts mel, F0, energy, and durations."""
        if wav_path is None:
            wav_path = os.path.join(self.in_dir, speaker, f"{basename}.wav")
        if tg_path is None:
            tg_path = os.path.join(self.out_dir, "TextGrid", speaker, f"{basename}.TextGrid")

        if not os.path.exists(tg_path) or not os.path.exists(wav_path):
            return None

        # 1. Parse TextGrid
        textgrid = tgt.io.read_textgrid(tg_path)
        phone_tier = textgrid.get_tier_by_name("phones")
        phone, duration, start, end = self.get_alignment(phone_tier)
        if start >= end or len(phone) == 0:
            return None

        text = "{" + " ".join(phone) + "}"

        # 2. Read and slice audio
        wav, sr = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start) : int(self.sampling_rate * end)].astype(np.float32)

        if len(wav) < self.hop_length:
            return None

        # 3. Read raw transcript if available
        if raw_text is None:
            lab_path = os.path.splitext(wav_path)[0] + ".lab"
            if os.path.exists(lab_path):
                with open(lab_path, "r", encoding="utf-8") as f:
                    raw_text = f.readline().strip()
            else:
                raw_text = text

        # 4. Mel spectrogram and STFT energy
        mel_spectrogram, energy = get_mel_from_wav(wav, self.STFT)
        T_mel = mel_spectrogram.shape[1]

        # 5. Ensure sum(durations) == T_mel
        dur_sum = sum(duration)
        if dur_sum != T_mel and duration:
            diff = T_mel - dur_sum
            max_idx = int(np.argmax(duration))
            if duration[max_idx] + diff >= 1:
                duration[max_idx] += diff
            else:
                while sum(duration) != T_mel:
                    cur_diff = T_mel - sum(duration)
                    if cur_diff > 0:
                        duration[int(np.argmax(duration))] += 1
                    else:
                        cand = [i for i, d in enumerate(duration) if d > 1]
                        if not cand:
                            break
                        duration[cand[int(np.argmax([duration[i] for i in cand]))]] -= 1

        # 6. PyWorld Pitch (F0)
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: sum(duration)]

        # Voiced frames check
        voiced_count = np.sum(pitch > 0)
        is_valid = voiced_count > 0 and not np.isnan(pitch).any() and not np.isinf(pitch).any()

        # 7. Phoneme-level averaging
        if self.pitch_phoneme_averaging and len(duration) > 0:
            nonzero_ids = np.where(pitch != 0)[0]
            if len(nonzero_ids) > 0:
                interp_fn = interp1d(
                    nonzero_ids,
                    pitch[nonzero_ids],
                    fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                    bounds_error=False,
                )
                pitch = interp_fn(np.arange(0, len(pitch)))
            else:
                pitch = np.zeros(len(pitch))

            p_avg = np.zeros(len(duration))
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    p_avg[i] = np.mean(pitch[pos : pos + d])
                pos += d
            pitch = p_avg

        if self.energy_phoneme_averaging and len(duration) > 0:
            e_avg = np.zeros(len(duration))
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    e_avg[i] = np.mean(energy[pos : pos + d])
                pos += d
            energy = e_avg

        # 8. Save un-normalized features to disk
        dur_filename = f"{speaker}-duration-{basename}.npy"
        np.save(os.path.join(self.out_dir, "duration", dur_filename), np.array(duration))

        pitch_filename = f"{speaker}-pitch-{basename}.npy"
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = f"{speaker}-energy-{basename}.npy"
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = f"{speaker}-mel-{basename}.npy"
        np.save(os.path.join(self.out_dir, "mel", mel_filename), mel_spectrogram.T)

        info_line = "|".join([basename, speaker, text, raw_text])
        return (
            info_line,
            self.remove_outlier(pitch[pitch > 0]) if np.sum(pitch > 0) > 0 else np.array([0.0]),
            self.remove_outlier(energy),
            T_mel,
            is_valid,
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", ""]
        phones = []
        durations = []
        start_time = 0.0
        end_time = 0.0
        end_idx = 0

        for t in tier.intervals:
            s, e, p = t.start_time, t.end_time, t.text.strip()
            if not phones and p in sil_phones:
                continue
            if not phones:
                start_time = s

            phones.append(p)
            if p not in sil_phones:
                end_time = e
                end_idx = len(phones)

            d = int(
                np.round(e * self.sampling_rate / self.hop_length)
                - np.round(s * self.sampling_rate / self.hop_length)
            )
            durations.append(max(1, d))

        phones = phones[:end_idx]
        durations = durations[:end_idx]
        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        if len(values) < 4:
            return values
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values >= lower, values <= upper)
        return values[normal_indices] if np.any(normal_indices) else values
