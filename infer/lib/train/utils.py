import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
from scipy.io.wavfile import read
from torch.nn import Module
from torch.optim import Optimizer

# ==============================================================================
# การตั้งค่า Logger ระดับ Global (Global Logger Configuration)
# ==============================================================================
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# สถานะเพื่อป้องกันการนำเข้า Matplotlib ซ้ำซ้อน
MATPLOTLIB_FLAG = False

# ==============================================================================
# ฟังก์ชันสำหรับการจัดการ Checkpoint (Checkpoint Management)
# ==============================================================================

def load_checkpoint(
    checkpoint_path: str, 
    model: Module, 
    optimizer: Optional[Optimizer] = None, 
    load_opt: int = 1
) -> Tuple[Module, Optional[Optimizer], float, int]:
    """
    โหลดน้ำหนักของโมเดล สถานะของ Optimizer และค่าพารามิเตอร์อื่นๆ จาก Checkpoint
    
    Args:
        checkpoint_path (str): เส้นทางไปยังไฟล์ Checkpoint
        model (torch.nn.Module): โมเดลที่ต้องการโหลดค่าน้ำหนัก
        optimizer (torch.optim.Optimizer, optional): Optimizer สำหรับโหลดสถานะ
        load_opt (int): ค่า 1 เพื่อโหลดสถานะ Optimizer, 0 เพื่อข้ามการโหลด
        
    Returns:
        tuple: (โมเดล, Optimizer, Learning Rate, รอบการฝึกสอนปัจจุบัน)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ Checkpoint ที่ระบุ: {checkpoint_path}")

    # ใช้ weights_only=True หากเป็นไปได้เพื่อความปลอดภัย (PyTorch รุ่นใหม่ๆ)
    try:
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        # ถอยกลับไปใช้แบบเดิมหาก PyTorch เป็นรุ่นเก่า
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict.get("model")
    if saved_state_dict is None:
        raise KeyError("ไฟล์ Checkpoint ไม่มีคีย์ 'model'")

    # ตรวจสอบว่าโมเดลถูกห่อหุ้มด้วย DataParallel หรือ DistributedDataParallel หรือไม่
    is_module = hasattr(model, "module")
    state_dict = model.module.state_dict() if is_module else model.state_dict()
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in saved_state_dict:
            if saved_state_dict[k].shape == state_dict[k].shape:
                new_state_dict[k] = saved_state_dict[k]
            else:
                logger.warning(
                    f"พบความไม่สอดคล้องของขนาด (Shape Mismatch) สำหรับคีย์ '{k}' | "
                    f"ต้องการ: {state_dict[k].shape} | ได้รับ: {saved_state_dict[k].shape}"
                )
                new_state_dict[k] = v # ใช้ค่าเริ่มต้นของโมเดลแทน
        else:
            logger.info(f"ไม่พบคีย์ '{k}' ใน Checkpoint (อาจเป็นเลเยอร์ใหม่) จะใช้ค่าเริ่มต้น")
            new_state_dict[k] = v

    # โหลดค่าน้ำหนักที่ถูกปรับปรุงแล้วกลับเข้าไปในโมเดล
    if is_module:
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
        
    logger.info("โหลดค่าน้ำหนักของโมเดลสำเร็จ")

    iteration = checkpoint_dict.get("iteration", 0)
    learning_rate = checkpoint_dict.get("learning_rate", 0.0)

    # โหลดสถานะของ Optimizer หากได้รับการร้องขอ
    if optimizer is not None and load_opt == 1 and "optimizer" in checkpoint_dict:
        try:
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
        except ValueError as e:
            logger.warning(f"ไม่สามารถโหลดสถานะ Optimizer ได้ อาจมีการเปลี่ยนแปลงโครงสร้าง: {e}")

    logger.info(f"ดึงข้อมูล Checkpoint: '{checkpoint_path}' ที่รอบฝึกที่ (Epoch): {iteration} สำเร็จ")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: Module, 
    optimizer: Optimizer, 
    learning_rate: float, 
    iteration: int, 
    checkpoint_path: str
) -> None:
    """
    บันทึกสถานะการฝึกสอนปัจจุบันลงในไฟล์ Checkpoint
    """
    logger.info(f"กำลังบันทึกสถานะโมเดลและ Optimizer ที่รอบฝึกที่ {iteration} ไปยัง {checkpoint_path}")
    
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def latest_checkpoint_path(dir_path: str, regex: str = "G_*.pth") -> Optional[str]:
    """ค้นหาไฟล์ Checkpoint ล่าสุดในไดเรกทอรีที่ระบุ"""
    f_list = glob.glob(os.path.join(dir_path, regex))
    if not f_list:
        return None
        
    # เรียงลำดับไฟล์ตามตัวเลขที่อยู่ในชื่อไฟล์
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f))) or -1))
    latest_file = f_list[-1]
    logger.debug(f"พบ Checkpoint ล่าสุดที่: {latest_file}")
    return latest_file


# ==============================================================================
# ฟังก์ชันสำหรับการแสดงผลภาพ (Visualization Utils)
# ==============================================================================

def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """แปลงข้อมูล Spectrogram ให้เป็นอาเรย์ของภาพ (Numpy Array) สำหรับ TensorBoard"""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg") # ใช้ Backend ที่ไม่ต้องใช้ GUI
        MATPLOTLIB_FLAG = True
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        
    import matplotlib.pylab as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("เฟรม (Frames)")
    plt.ylabel("แชนเนล (Channels)")
    plt.tight_layout()

    fig.canvas.draw()
    # ดึงค่าพิกเซลออกมาเป็น Numpy Array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return data

# ==============================================================================
# ฟังก์ชันการจัดการไฟล์และข้อมูล (Data & Config Management)
# ==============================================================================

def load_wav_to_torch(full_path: str) -> Tuple[torch.Tensor, int]:
    """โหลดไฟล์เสียง .wav และแปลงเป็น PyTorch Tensor"""
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class HParams:
    """
    คลาสสำหรับจัดการ Hyperparameters (พารามิเตอร์ที่ใช้ควบคุมโมเดล)
    ให้สามารถเข้าถึงผ่าน dot notation (เช่น hparams.batch_size) ได้สะดวก
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

def get_hparams(init: bool = True) -> HParams:
    """
    ระบบรับค่าพารามิเตอร์จาก Command Line อย่างเป็นทางการ
    ผสานรวมการอ่านค่าจากไฟล์ config.json
    """
    parser = argparse.ArgumentParser(description="ตั้งค่าพารามิเตอร์สำหรับการฝึกสอน (Training Configuration)")
    
    # อาร์กิวเมนต์หลักสำหรับการควบคุมการฝึกสอน
    parser.add_argument("-se", "--save_every_epoch", type=int, required=True, help="ความถี่ในการบันทึก Checkpoint (จำนวนรอบ)")
    parser.add_argument("-te", "--total_epoch", type=int, required=True, help="จำนวนรอบการฝึกสอนรวมทั้งหมด")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="ขนาดของ Batch")
    parser.add_argument("-e", "--experiment_dir", type=str, required=True, help="ชื่อโฟลเดอร์สำหรับเก็บผลลัพธ์การทดลอง")
    parser.add_argument("-sr", "--sample_rate", type=str, required=True, help="อัตราการสุ่มสัญญาณเสียง (เช่น 32k/40k/48k)")
    parser.add_argument("-v", "--version", type=str, required=True, help="เวอร์ชันของสถาปัตยกรรมโมเดล")
    parser.add_argument("-f0", "--if_f0", type=int, required=True, help="เปิดใช้งาน F0 (Pitch) หรือไม่ (1=เปิด, 0=ปิด)")
    
    # อาร์กิวเมนต์เพิ่มเติม
    parser.add_argument("-pg", "--pretrainG", type=str, default="", help="เส้นทางไฟล์โมเดลตั้งต้นของ Generator (Pretrained)")
    parser.add_argument("-pd", "--pretrainD", type=str, default="", help="เส้นทางไฟล์โมเดลตั้งต้นของ Discriminator (Pretrained)")
    parser.add_argument("-g", "--gpus", type=str, default="0", help="รหัสการ์ดจอที่จะใช้งาน (คั่นด้วย -)")
    parser.add_argument("-sw", "--save_every_weights", type=str, default="0", help="ความถี่ในการแตกไฟล์น้ำหนัก (Weights Extraction)")
    parser.add_argument("-l", "--if_latest", type=int, required=True, help="บันทึกเฉพาะไฟล์ Checkpoint ล่าสุดหรือไม่ (1=ใช่, 0=ไม่)")
    parser.add_argument("-c", "--if_cache_data_in_gpu", type=int, required=True, help="โหลดข้อมูลทั้งหมดเข้าสู่ GPU เลยหรือไม่ (1=ใช่, 0=ไม่)")

    args = parser.parse_args()
    
    # กำหนดเส้นทางหลัก
    name = args.experiment_dir
    experiment_dir = os.path.join("./logs", args.experiment_dir)
    config_save_path = os.path.join(experiment_dir, "config.json")
    
    # อ่านค่าเริ่มต้นจากไฟล์ JSON
    if not os.path.exists(config_save_path):
        raise FileNotFoundError(f"ไม่พบไฟล์การตั้งค่าที่: {config_save_path}")
        
    with open(config_save_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # ผสานการตั้งค่า
    hparams = HParams(**config)
    
    # อัปเดตพารามิเตอร์ที่ได้รับจาก Command Line ทับลงไป
    hparams.model_dir = experiment_dir
    hparams.experiment_dir = experiment_dir
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.version = args.version
    hparams.gpus = args.gpus
    
    # จัดการโครงสร้างย่อยของ Train และ Data อย่างระมัดระวัง
    if not hasattr(hparams, "train"):
        hparams.train = HParams()
    hparams.train.batch_size = args.batch_size
    
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    
    if not hasattr(hparams, "data"):
        hparams.data = HParams()
    hparams.data.training_files = os.path.join(experiment_dir, "filelist.txt")
    
    return hparams

def get_logger(model_dir: str, filename: str = "train.log") -> logging.Logger:
    """
    กำหนดค่าและเตรียมใช้งานระบบบันทึกเหตุการณ์ (Logger) ประจำการฝึกสอน
    """
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    # จัดรูปแบบข้อความให้สวยงามและเป็นทางการ
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # ป้องกันการเพิ่ม Handler ซ้ำเมื่อถูกเรียกหลายครั้ง
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(model_dir, filename), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
