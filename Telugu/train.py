"""FastSpeech 2 Training Entrypoint for Telugu (IndicTTS).
Production-grade training loop supporting single-speaker female prosody, gradient accumulation,
dynamic learning rate scheduling, full validation set evaluation, deterministic checkpointing,
loss logging (CSV + TensorBoard), crash state dumping, and smoke test mode.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

dir_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(dir_path)
for p in [dir_path, parent_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset import Dataset
from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from utils.model import get_param_num, get_vocoder
from utils.tools import log, synth_one_sample, to_device


class SyntheticDataset(TorchDataset):
    """Synthetic dataset for smoke-test verification and dry-run testing."""
    def __init__(self, n_samples=32, n_symbols=25, n_frames=80, n_mels=80):
        self.n_samples = n_samples
        self.n_symbols = n_symbols
        self.n_frames = n_frames
        self.n_mels = n_mels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate valid mock tensors for Telugu (58-phone inventory)
        phone = np.random.randint(1, 55, size=(self.n_symbols,), dtype=np.int64)
        mel = np.random.randn(self.n_frames, self.n_mels).astype(np.float32)
        pitch = np.random.uniform(150.0, 350.0, size=(self.n_symbols,)).astype(np.float32)
        energy = np.random.uniform(0.1, 1.0, size=(self.n_symbols,)).astype(np.float32)
        
        # Partition n_frames across n_symbols
        durations = np.ones(self.n_symbols, dtype=np.int64)
        rem = self.n_frames - self.n_symbols
        if rem > 0:
            adds = np.random.multinomial(rem, [1.0 / self.n_symbols] * self.n_symbols)
            durations += adds

        return {
            "id": f"synth_utt_{idx:04d}",
            "speaker": 0,
            "text": phone,
            "raw_text": "సింథటిక్ నమూనా వాక్యం",
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": durations,
        }

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        idx_sort = np.argsort(-len_arr)
        batch = [batch[i] for i in idx_sort]

        ids = [d["id"] for d in batch]
        raw_texts = [d["raw_text"] for d in batch]
        speakers = np.array([d["speaker"] for d in batch])
        texts = [d["text"] for d in batch]
        text_lens = np.array([len(d["text"]) for d in batch])
        mel_lens = np.array([d["mel"].shape[0] for d in batch])

        max_text_len = max(text_lens)
        max_mel_len = max(mel_lens)

        texts_pad = np.zeros((len(batch), max_text_len), dtype=np.int64)
        mels_pad = np.zeros((len(batch), max_mel_len, self.n_mels), dtype=np.float32)
        pitches_pad = np.zeros((len(batch), max_text_len), dtype=np.float32)
        energies_pad = np.zeros((len(batch), max_text_len), dtype=np.float32)
        durations_pad = np.zeros((len(batch), max_text_len), dtype=np.int64)

        for i, d in enumerate(batch):
            tl = text_lens[i]
            ml = mel_lens[i]
            texts_pad[i, :tl] = d["text"]
            mels_pad[i, :ml] = d["mel"]
            pitches_pad[i, :tl] = d["pitch"]
            energies_pad[i, :tl] = d["energy"]
            durations_pad[i, :tl] = d["duration"]

        return [
            (
                ids,
                raw_texts,
                speakers,
                texts_pad,
                text_lens,
                max_text_len,
                mels_pad,
                mel_lens,
                max_mel_len,
                pitches_pad,
                energies_pad,
                durations_pad,
            )
        ]


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=parent_path).decode("ascii").strip()
    except Exception:
        return "UNKNOWN"


def compute_sha256(filepath: str) -> str:
    try:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return "UNKNOWN"


def set_deterministic_seeds(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_run_metadata(log_dir: str, config_path: str, configs: tuple, seed: int, device: str):
    os.makedirs(log_dir, exist_ok=True)
    run_info_path = os.path.join(log_dir, "run_info.json")
    
    preprocess_config, model_config, train_config = configs
    
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_revision_hash(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "seed": seed,
        "config_path": config_path,
        "config_sha256": compute_sha256(config_path),
        "dataset": preprocess_config.get("dataset", "IndicTTS"),
        "multi_speaker": model_config.get("multi_speaker", False),
        "batch_size": train_config["optimizer"]["batch_size"],
        "grad_acc_step": train_config["optimizer"]["grad_acc_step"],
        "warm_up_step": train_config["optimizer"]["warm_up_step"],
        "total_step": train_config["step"]["total_step"],
    }
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train FastSpeech 2 (Telugu).")
    parser.add_argument("--config", type=str, default=os.path.join(dir_path, "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--restore_step", type=int, default=0, help="Step to restore from")
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint file to resume from")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run quick 100-step smoke test")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total steps (e.g. 100 for dry run)")
    args = parser.parse_args()

    # 1. Enforce determinism
    seed = 1234
    set_deterministic_seeds(seed)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Telugu Train] Initializing training on device: {device}")

    # 2. Load configurations
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preprocess_config = cfg
    model_config = cfg["model"]
    train_config = cfg["train"]
    train_config["path"] = cfg.get("path", {})
    configs = (preprocess_config, model_config, train_config)

    # Output paths
    ckpt_dir = os.path.join(dir_path, "checkpoints")
    log_dir = os.path.join(dir_path, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_run_metadata(log_dir, args.config, configs, seed, str(device))

    # 3. Prepare Datasets & DataLoaders
    batch_size = train_config["optimizer"]["batch_size"]
    preprocessed_path = preprocess_config["path"].get("preprocessed_path", "preprocessed_data/telugu")
    train_txt_path = os.path.join(preprocessed_path, "train.txt")
    val_txt_path = os.path.join(preprocessed_path, "val.txt")

    use_synthetic = args.smoke_test or not (os.path.exists(train_txt_path) and os.path.exists(val_txt_path))

    if use_synthetic:
        print("[Telugu Train] Operating in SMOKE TEST / SYNTHETIC DATASET mode.")
        train_dataset = SyntheticDataset(n_samples=max(32, batch_size * 4))
        val_dataset = SyntheticDataset(n_samples=max(16, batch_size * 2))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )
    else:
        print(f"[Telugu Train] Loading real dataset from {preprocessed_path}")
        train_dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
        val_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
        group_size = 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size * group_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )

    # 4. Initialize Model, Loss, Optimizer, Vocoder
    model = FastSpeech2(preprocess_config, model_config).to(device)
    if torch.cuda.device_count() > 1 and str(device) == "cuda":
        model = nn.DataParallel(model)

    num_param = get_param_num(model)
    print(f"[Telugu Train] Number of FastSpeech 2 parameters: {num_param:,}")

    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)

    try:
        vocoder = get_vocoder(model_config, device)
    except Exception as e:
        print(f"[Telugu Train] Warning: Vocoder could not be loaded ({e}); continuing without audio logging.")
        vocoder = None

    optimizer = ScheduledOptim(
        model.module if hasattr(model, "module") else model,
        train_config,
        model_config,
        args.restore_step,
    )

    # 5. Handle Checkpoint Resumption
    step = 1
    epoch = 1
    best_val_loss = float("inf")

    resume_path = args.resume
    if not resume_path and args.restore_step > 0:
        resume_path = os.path.join(ckpt_dir, f"step_{args.restore_step}.pth")

    if resume_path and os.path.exists(resume_path):
        print(f"[Telugu Train] Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "step" in ckpt:
            step = ckpt["step"] + 1
        if "epoch" in ckpt:
            epoch = ckpt["epoch"]
        if "best_val_loss" in ckpt:
            best_val_loss = ckpt["best_val_loss"]
        if "rng_states" in ckpt:
            rng = ckpt["rng_states"]
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].cpu())
            if "cuda" in rng and rng["cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        print(f"[Telugu Train] Resumed successfully at step {step}, epoch {epoch}")

    # 6. Setup Loggers & Loss CSVs
    train_log_path = os.path.join(log_dir, "train")
    val_log_path = os.path.join(log_dir, "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    train_csv_path = os.path.join(log_dir, "train_losses.csv")
    val_csv_path = os.path.join(log_dir, "val_losses.csv")

    csv_header = "step,epoch,lr,total,mel,postnet_mel,pitch,energy,duration,grad_norm,seconds_per_step\n"
    if not os.path.exists(train_csv_path):
        with open(train_csv_path, "w", encoding="utf-8") as f:
            f.write(csv_header)
    if not os.path.exists(val_csv_path):
        with open(val_csv_path, "w", encoding="utf-8") as f:
            f.write(csv_header)

    # 7. Training Parameters
    total_step = args.max_steps if args.max_steps else (100 if args.smoke_test else train_config["step"]["total_step"])
    log_step = 10 if args.smoke_test else train_config["step"]["log_step"]
    val_step = 25 if args.smoke_test else train_config["step"]["val_step"]
    save_step = 50 if args.smoke_test else train_config["step"]["save_step"]
    synth_step = 50 if args.smoke_test else train_config["step"]["synth_step"]
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]

    print(f"[Telugu Train] Starting training: {total_step} total steps (log every {log_step}, val every {val_step}, save every {save_step})")

    pbar = tqdm(total=total_step, initial=step - 1, desc="Training")
    step_start_time = time.time()

    model.train()

    while step <= total_step:
        for batchs in train_loader:
            for batch in batchs:
                if step > total_step:
                    break

                batch = to_device(batch, device)

                # Forward pass
                output = model(*(batch[2:]))

                # Compute loss
                losses = loss_fn(batch, output)
                total_loss = losses[0]

                # Check for NaN / Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    crash_path = os.path.join(ckpt_dir, f"crash_step_{step}.pth")
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save({
                        "step": step,
                        "epoch": epoch,
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, crash_path)
                    print(f"\n[CRITICAL ERROR] NaN/Inf loss detected at step {step}! Crash checkpoint saved to {crash_path}")
                    sys.exit(1)

                # Gradient accumulation
                loss_scaled = total_loss / grad_acc_step
                loss_scaled.backward()

                grad_norm = 0.0
                if step % grad_acc_step == 0:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                step_duration = time.time() - step_start_time
                step_start_time = time.time()

                # Train Logging
                if step % log_step == 0:
                    lr = optimizer.get_lr()
                    loss_vals = [l.item() for l in losses]
                    row_data = [step, epoch, lr] + loss_vals + [grad_norm, step_duration]
                    with open(train_csv_path, "a", encoding="utf-8") as f:
                        f.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row_data) + "\n")

                    # TensorBoard
                    train_logger.add_scalar("Loss/total", loss_vals[0], step)
                    train_logger.add_scalar("Loss/mel", loss_vals[1], step)
                    train_logger.add_scalar("Loss/postnet_mel", loss_vals[2], step)
                    train_logger.add_scalar("Loss/pitch", loss_vals[3], step)
                    train_logger.add_scalar("Loss/energy", loss_vals[4], step)
                    train_logger.add_scalar("Loss/duration", loss_vals[5], step)
                    train_logger.add_scalar("Training/lr", lr, step)
                    train_logger.add_scalar("Training/grad_norm", grad_norm, step)

                    pbar.set_postfix({
                        "Loss": f"{loss_vals[0]:.4f}",
                        "Mel": f"{loss_vals[1]:.4f}",
                        "LR": f"{lr:.2e}",
                    })

                # Synthesis logging
                if step % synth_step == 0 and vocoder is not None:
                    try:
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch, output, vocoder, model_config, preprocess_config
                        )
                        log(train_logger, fig=fig, tag=f"Training/step_{step}_{tag}")
                    except Exception:
                        pass

                # Validation Loop
                if step % val_step == 0:
                    model.eval()
                    val_loss_sums = [0.0 for _ in range(6)]
                    total_val_samples = 0
                    with torch.no_grad():
                        for v_batchs in val_loader:
                            for v_batch in v_batchs:
                                v_batch = to_device(v_batch, device)
                                v_output = model(*(v_batch[2:]))
                                v_losses = loss_fn(v_batch, v_output)
                                bsz = len(v_batch[0])
                                total_val_samples += bsz
                                for i in range(len(v_losses)):
                                    val_loss_sums[i] += v_losses[i].item() * bsz

                    val_loss_means = [s / max(1, total_val_samples) for s in val_loss_sums]
                    val_lr = optimizer.get_lr()
                    val_row = [step, epoch, val_lr] + val_loss_means + [0.0, 0.0]
                    with open(val_csv_path, "a", encoding="utf-8") as f:
                        f.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in val_row) + "\n")

                    val_logger.add_scalar("Loss/total", val_loss_means[0], step)
                    val_logger.add_scalar("Loss/mel", val_loss_means[1], step)
                    val_logger.add_scalar("Loss/postnet_mel", val_loss_means[2], step)
                    val_logger.add_scalar("Loss/pitch", val_loss_means[3], step)
                    val_logger.add_scalar("Loss/energy", val_loss_means[4], step)
                    val_logger.add_scalar("Loss/duration", val_loss_means[5], step)

                    # Update best model
                    save_prefix = "smoke_" if args.smoke_test else ""
                    if val_loss_means[0] < best_val_loss:
                        best_val_loss = val_loss_means[0]
                        raw_model = model.module if hasattr(model, "module") else model
                        best_ckpt_path = os.path.join(ckpt_dir, f"{save_prefix}best.pth")
                        torch.save({
                            "step": step,
                            "epoch": epoch,
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                            "rng_states": {
                                "python": random.getstate(),
                                "numpy": np.random.get_state(),
                                "torch": torch.get_rng_state(),
                                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            },
                        }, best_ckpt_path)

                    model.train()

                # Checkpoint saving
                if step % save_step == 0 or step == total_step:
                    save_prefix = "smoke_" if args.smoke_test else ""
                    raw_model = model.module if hasattr(model, "module") else model
                    ckpt_payload = {
                        "step": step,
                        "epoch": epoch,
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "rng_states": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                    }
                    torch.save(ckpt_payload, os.path.join(ckpt_dir, f"{save_prefix}last.pth"))
                    torch.save(ckpt_payload, os.path.join(ckpt_dir, f"{save_prefix}step_{step}.pth"))

                step += 1
                pbar.update(1)

        epoch += 1

    pbar.close()
    print(f"[Telugu Train] Training loop complete at step {total_step}. Checkpoints saved in {ckpt_dir}")


if __name__ == "__main__":
    main()
