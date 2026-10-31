"""Initializes provisional checkpoints for English and Telugu to test the inference pipeline."""

import os
import sys
import yaml
import torch

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from English.model.fastspeech2 import FastSpeech2 as EnFastSpeech2
from Telugu.model.fastspeech2 import FastSpeech2 as TeFastSpeech2


def create_provisional_checkpoints():
    # 1. English
    en_config_path = os.path.join(workspace_root, "English", "config.yaml")
    with open(en_config_path, "r", encoding="utf-8") as f:
        en_config = yaml.safe_load(f)

    en_model = EnFastSpeech2(en_config, en_config["model"])
    os.makedirs(os.path.join(workspace_root, "checkpoints", "english"), exist_ok=True)
    en_ckpt_path = os.path.join(workspace_root, "checkpoints", "english", "last.pth")
    torch.save(
        {
            "model": en_model.state_dict(),
            "step": 1000,
            "epoch": 1,
        },
        en_ckpt_path,
    )
    print(f"Created English provisional checkpoint: {en_ckpt_path}")

    # 2. Telugu
    te_config_path = os.path.join(workspace_root, "Telugu", "config.yaml")
    with open(te_config_path, "r", encoding="utf-8") as f:
        te_config = yaml.safe_load(f)

    te_model = TeFastSpeech2(te_config, te_config["model"])
    os.makedirs(os.path.join(workspace_root, "checkpoints", "telugu"), exist_ok=True)
    te_ckpt_path = os.path.join(workspace_root, "checkpoints", "telugu", "last.pth")
    torch.save(
        {
            "model": te_model.state_dict(),
            "step": 1000,
            "epoch": 1,
        },
        te_ckpt_path,
    )
    print(f"Created Telugu provisional checkpoint: {te_ckpt_path}")


if __name__ == "__main__":
    create_provisional_checkpoints()

