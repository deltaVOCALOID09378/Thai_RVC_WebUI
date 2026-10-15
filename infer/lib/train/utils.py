# -*- coding: utf-8 -*-
# ==============================================================================
# วัน/เดือน/ปี: 12/10/2569 | 12/10/2026
# ชื่อไฟล์: utils.py
# ลำดับเวอร์ชัน: 1.5
# Made And Checked By DELTA SYNTH [ ปฎิภัทร วงศ์ใหญ่ ] & All Code Agentic AI Engine
# Original by "RVC-Project"
# ตำแหน่งที่วางไฟล์: infer/lib/train/utils.py
# หน้าที่: ระบบเครื่องมือยูทิลิตี้เสริมสำหรับการฝึกสอนและจัดการโมเดล
# รายละเอียดการแก้ไขงาน: เพิ่มระบบ Auto-Fallback ป้องกัน JSONDecodeError และกำจัด Bare Excepts
# ==============================================================================
import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import traceback

import numpy as np
import torch
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_checkpoint_d(checkpoint_path, combd, sbd, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def go(model, bkey):
        saved_state_dict = checkpoint_dict[bkey]
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict[k]
                if saved_state_dict[k].shape != state_dict[k].shape:
                    logger.warning("shape-%s-mismatch. need: %s, get: %s", k, state_dict[k].shape, saved_state_dict[k].shape)
                    raise KeyError
            except Exception as e:
                logger.info("%s is not in the checkpoint (Fallback executed)", k)
                new_state_dict[k] = v
        if hasattr(model, "module"):
            model.module.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)
        return model

    go(combd, "combd")
    model = go(sbd, "sbd")
    logger.info("Loaded model weights")

    iteration = checkpoint_dict.get("iteration", 0)
    learning_rate = checkpoint_dict.get("learning_rate", 1e-4)
    if optimizer is not None and load_opt == 1:
        try:
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
        except Exception as e:
            logger.warning(f"Failed to load optimizer: {str(e)}")
            
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration

def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    saved_state_dict = checkpoint_dict["model"]
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warning("shape-%s-mismatch|need-%s|get-%s", k, state_dict[k].shape, saved_state_dict[k].shape)
                raise KeyError
        except Exception:
            logger.info("%s is not in the checkpoint", k)
            new_state_dict[k] = v
            
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict.get("iteration", 0)
    learning_rate = checkpoint_dict.get("learning_rate", 1e-4)
    if optimizer is not None and load_opt == 1:
        try:
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
        except Exception as e:
            logger.warning(f"Failed to load optimizer: {str(e)}")
            
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(f"Saving model and optimizer state at epoch {iteration} to {checkpoint_path}")
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "learning_rate": learning_rate,
    }, checkpoint_path)

def save_checkpoint_d(combd, sbd, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(f"Saving model and optimizer state at epoch {iteration} to {checkpoint_path}")
    state_dict_combd = combd.module.state_dict() if hasattr(combd, "module") else combd.state_dict()
    state_dict_sbd = sbd.module.state_dict() if hasattr(sbd, "module") else sbd.state_dict()
    torch.save({
        "combd": state_dict_combd,
        "sbd": state_dict_sbd,
        "iteration": iteration,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "learning_rate": learning_rate,
    }, checkpoint_path)

def summarize(writer, global_step, scalars=None, histograms=None, images=None, audios=None, audio_sampling_rate=22050):
    scalars = scalars or {}
    histograms = histograms or {}
    images = images or {}
    audios = audios or {}
    for k, v in scalars.items(): writer.add_scalar(k, v, global_step)
    for k, v in histograms.items(): writer.add_histogram(k, v, global_step)
    for k, v in images.items(): writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items(): writer.add_audio(k, v, global_step, audio_sampling_rate)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.debug(x)
    return x

def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-se", "--save_every_epoch", type=int, required=True, help="checkpoint save frequency")
    parser.add_argument("-te", "--total_epoch", type=int, required=True, help="total_epoch")
    parser.add_argument("-pg", "--pretrainG", type=str, default="", help="Pretrained Generator path")
    parser.add_argument("-pd", "--pretrainD", type=str, default="", help="Pretrained Discriminator path")
    parser.add_argument("-g", "--gpus", type=str, default="0", help="split by -")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("-e", "--experiment_dir", type=str, required=True, help="experiment dir")
    parser.add_argument("-sr", "--sample_rate", type=str, required=True, help="sample rate")
    parser.add_argument("-sw", "--save_every_weights", type=str, default="0", help="save extracted model")
    parser.add_argument("-v", "--version", type=str, required=True, help="model version")
    parser.add_argument("-f0", "--if_f0", type=int, required=True, help="use f0, 1 or 0")
    parser.add_argument("-l", "--if_latest", type=int, required=True, help="save latest only, 1 or 0")
    parser.add_argument("-c", "--if_cache_data_in_gpu", type=int, required=True, help="cache dataset in GPU, 1 or 0")

    args = parser.parse_args()
    name = args.experiment_dir
    experiment_dir = os.path.join("./logs", args.experiment_dir)
    config_save_path = os.path.join(experiment_dir, "config.json")

    config = {}
    try:
        with open(config_save_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"🔴 [ERROR] ตรวจพบไฟล์ตั้งค่าเสียหาย (Config corrupted) {config_save_path}: {e}")
        logger.warning(f"🟡 [WARNING] กำลังดึงค่าเริ่มต้นอัตโนมัติ (Falling back to default config)...")
        default_cfg = f"v1/{args.sample_rate}.json" if args.version == "v1" or args.sample_rate == "40k" else f"v2/{args.sample_rate}.json"
        try:
            with open(os.path.join("configs", default_cfg), "r", encoding="utf-8") as f_def:
                config = json.load(f_def)
        except Exception as fallback_e:
            logger.error(f"🔴 [FATAL] ไม่สามารถกู้คืนค่าเริ่มต้นได้ (Fallback failed): {fallback_e}")
            sys.exit(1)

    hparams = HParams(**config)
    hparams.model_dir = hparams.experiment_dir = experiment_dir
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.version = args.version
    hparams.gpus = args.gpus
    if not hasattr(hparams, "train"): hparams.train = HParams()
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    if not hasattr(hparams, "data"): hparams.data = HParams()
    hparams.data.training_files = f"{experiment_dir}/filelist.txt"
    return hparams

def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_save_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {config_save_path}: {e}")
        config = {}
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def get_hparams_from_file(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {config_path}: {e}")
        config = {}
    return HParams(**config)

def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        return
    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warning(f"git hash values differ. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)")
    else:
        open(path, "w").write(cur_hash)

def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    os.makedirs(model_dir, exist_ok=True)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v
    def keys(self): return self.__dict__.keys()
    def items(self): return self.__dict__.items()
    def values(self): return self.__dict__.values()
    def __len__(self): return len(self.__dict__)
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): setattr(self, key, value)
    def __contains__(self, key): return key in self.__dict__
    def __repr__(self): return repr(self.__dict__)