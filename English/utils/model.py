import json
import os

import hifigan
import numpy as np
import torch
from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            f"{args.restore_step}.pth.tar",
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_candidates = [
            os.path.join(base_dir, "hifigan", "config.json"),
            os.path.join(base_dir, "..", "vocoder", "config.json"),
            os.path.join(base_dir, "vocoder", "config.json"),
            "hifigan/config.json",
            "vocoder/config.json",
        ]
        ckpt_candidates = [
            os.path.join(base_dir, "hifigan", f"generator_{speaker}.pth.tar"),
            os.path.join(base_dir, "..", "vocoder", f"generator_{speaker}.pth.tar"),
            os.path.join(base_dir, "vocoder", f"generator_{speaker}.pth.tar"),
            f"hifigan/generator_{speaker}.pth.tar",
            f"vocoder/generator_{speaker}.pth.tar",
        ]
        cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), cfg_candidates[0])
        ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), ckpt_candidates[0])

        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load(ckpt_path, map_location=device)
        vocoder.load_state_dict(ckpt.get("generator", ckpt))
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
