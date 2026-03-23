import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from i18n.i18n import I18nAuto
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np
import gradio as gr
import faiss
import fairseq
import pathlib
import json
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging

# เพิ่มส่วนตรวจสอบ DirectML
try:
    import torch_directml
    has_dml = True
except ImportError:
    has_dml = False

logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)
   
if config.dml == True:
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

i18n = I18nAuto()
logger.info(i18n)

# ==========================================
# ส่วนตรวจสอบ GPU (NVIDIA & AMD/DirectML)
# ==========================================
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 1. ตรวจสอบการ์ดจอ NVIDIA ก่อน
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN",
            ]
        ):
            if_gpu_ok = True  
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4
                )
            )

# 2. ถ้าไม่เจอ NVIDIA ให้ตรวจสอบ AMD ผ่าน DirectML (ถ้ามีติดตั้งไว้)
if not if_gpu_ok and has_dml and torch_directml.is_available():
    if_gpu_ok = True
    dml_device_count = torch_directml.device_count()
    for i in range(dml_device_count):
        gpu_name = torch_directml.device_name(i)
        gpu_infos.append("%s\t%s (DirectML)" % (i, gpu_name))
        # DirectML ไม่ได้คืนค่า memory ง่ายๆ เหมือน CUDA จึงกำหนดค่ากลางไว้เผื่อคำนวณ batch_size
        mem.append(8) 

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2 if mem else 4 # ป้องกัน error ถ้า memory ว่าง
else:
    gpu_info = "Unfortunately, there is no compatible GPU available for training. Falling back to CPU."
    default_batch_size = 1

gpus = "-".join([i[0].split('\t')[0] for i in gpu_infos]) if gpu_infos else ""
# ==========================================

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }

def clean():
    return {"value": "", "__type__": "update"}

def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo

    eo(ModelPath, ExportedPath)

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  
                    ps.append(p)
                done = [False]
                threading.Thread(
                    target=if_done_multi,  
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  
        ps.append(p)
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == "Yes" else 0,
                1 if if_cache_gpu17 == "Yes" else 0,
                1 if if_save_every_weights18 == "Yes" else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == "Yes" else 0,
                1 if if_cache_gpu17 == "Yes" else 0,
                1 if if_save_every_weights18 == "Yes" else 0,
                version19,
            )
        )
    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "Training finished. You can check the console log or train.log in the experiment folder."


def train_index(exp_dir1, version19):
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "Please perform feature extraction first!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform feature extraction first!"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    infos.append("Training index...")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("Adding index...")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "Index built successfully: added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    yield "\n".join(infos)


def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    yield get_info_str("Step No. 1 Processing data...")
    [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    yield get_info_str("Step No. 2 Extracting pitch & features...")
    [
        get_info_str(_)
        for _ in extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
        )
    ]

    yield get_info_str("Step No. 3  Training model...")
    click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
    )
    yield get_info_str("The Training For Model is Complete. Please Check to the train.log's File in the experiment log folder.")

    [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    yield get_info_str("All Processes is Completed.!!!")


def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = config.dml == False


def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


with gr.Blocks(title="The RVC WebUI And Voice Remover WebUI Editor By DELTA SYNTH.") as app:
    gr.Markdown("**The Official English RVC Editor WebUI.**")
    gr.Markdown(
        value="***This Software is open source under the MIT License. The author has no control over the software. Users and distributors of the exported voices bear full responsibility.***"
    )
    with gr.Tabs():
        with gr.TabItem("Use The Inference for Model to Rendering."):
            with gr.Row():
                sid0 = gr.Dropdown(label="Choose to Inference the Model's Voice To Use it.", choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button("Click to Refresh For All Model Voice List", variant="primary")
                    clean_button = gr.Button("Click to Reset Voice File To Clear VRAM ( Save VRAM )", variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Input to the Speaker's ID",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem("1. Inference For 1 File Using."):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            vc_transform0 = gr.Number(
                                label="1.1. Input how much to Pitch Shift ( Integer, Semitones, +12 Octave Up is Same the kid Voice , -12 Octave Down is same the Adult guy voice. )", value=0
                            )
                            input_audio0 = gr.Textbox(
                                label="1.2. Input Audio Path (Example of correct format)",
                                placeholder="H:\Export to UTAU\Test.wav",
                                value="H:\Export to UTAU\Test.wav",
                            )
                            file_index1 = gr.Textbox(
                                label="1.3. Input Index Path (Leave blank to use dropdown)",
                                placeholder="Logs\A.index",
                                value="Logs\A.index",
                                interactive=True,
                            )
                            file_index2 = gr.Dropdown(
                                label="Auto-detect Index Path (Dropdown)",
                                value="Logs\A.index",
                                choices=sorted(index_paths),
                                interactive=True,
                            )
                            f0method0 = gr.Radio(
                                label="1.4. Click to Use the Pitch Extraction Algorithm ( The pm is fast but low quality, The harvest have a the bass upgrade but is so long time, The crepe have more the better Quality of Voice and Save down the GPU using, The RMVPE have the best quality of Voice and the best save down and slight the GPU using. )",
                                choices=["pm", "harvest", "crepe", "rmvpe"]
                                if config.dml == False
                                else ["pm", "harvest", "rmvpe"],
                                value="rmvpe",
                                interactive=True,
                            )

                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label="1.5. Input Resample Rate File.",
                                value=48000,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="1.6. Input for The Volume Envelope Mix Rate (Closer to 1 = Use output envelope)",
                                value=0.3,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label="1.7. Input to the Protect Voiceless Consonants&Breath rate. ( For 0.5 = disabled this, lower = stronger protection but may reduce index accuracy. )",
                                value=0.40,
                                step=0.01,
                                interactive=True,
                            )
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label="1.8. Input to The Median Filter Radius for Harvest Pitch ( more than 3 is enables, reduces breathiness )",
                                value=1,
                                step=1,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="1.9. Input to the Retrieval Ratio Feature rate ( Index Rate )",
                                value=0,
                                interactive=True,
                            )
                            f0_file = gr.File(
                                label="F0 Curve File (Optional, replaces default F0)",
                                visible=False,
                            )

                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=[sid0, file_index2],
                                api_name="Refresh The Inference.",
                            )

                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button("Click to Start Convert File.", variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label="The Output File's Info For Converting.")
                            vc_output2 = gr.Audio(label="The Output Audio's File is Complete, Listening This... ( Click to the ... For Downloading The Voice's File. )")

                        but0.click(
                            vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="The Multiple Infer Convert...",
                        )
            with gr.TabItem("2. Inference Voice For Multiple File in One Ordering."):
                gr.Markdown(
                    value="Use for Inference Voice For the Multiple File in One Ordering. Please Input to the Voice of Input File's Address For Importing All File And Input to the Voice of Output File's Address For Export All File To The Correct Address Before Click All The Process First!!!."
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label="2.1. Input the Pitch Shift (Integer, Semitones, +12 Octave Up, -12 Octave Down)", value=0
                        )
                        opt_input = gr.Textbox(label="2.2. Input the Output Folder", value="H:\F. Render file to UTAU\Singer_NAME\Language...")
                        file_index3 = gr.Textbox(
                            label="2.3. Input the Index Path (Leave blank to use dropdown)",
                            value="Logs\A.index",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label="2.4. Input the Auto-detect Index Path (Dropdown)",
                            choices=sorted(index_paths),
                            value="Logs\A.index",
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label="2.5. Click to Use the Pitch Extraction Algorithm ( The pm is fast but low quality, The harvest have a the bass upgrade but is so long time, The crepe have more the better Quality of Voice and Save down the GPU using, The RMVPE have the best quality of Voice and the best save down and slight the GPU using. )",
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label="2.6. Input to Export Formatting.",
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )

                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="Click To Refresh",
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="2.7. Input the Sample Rate For Export File. ( 0 Hz for none )",
                            value=48000,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="2.8. Input the Volume Envelope Mix Rate. (Closer to 1 = Use output envelope)",
                            value=0.3,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="2.9. Input the Protect Voiceless Consonants/Breath Rate. (0.5 = disabled, lower = stronger protection but may reduce index accuracy)",
                            value=0.4,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="2.10. Input the Median Filter Radius for Harvest Pitch Rate (>=3 enables, reduces breathiness)",
                            value=1,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="2.11. Input the Retrieval Ratio Feature Rate. (Index Rate)",
                            value=0,
                            interactive=True,
                        )
                with gr.Row():
                        dir_input = gr.Textbox(
                            label="2.12. Input the path of the Import audio folder to be the Processed ( Please Ctrl + C to copy it from the Flie address bar of the file manager And Paste in here.):",
                            value="H:\\Export to UTAU\\EN",
                        )
                        inputs = gr.File(
                            file_count="2.13. Input the Multiple File For Importing", label="Upload Audio Files (Optional, folder path takes priority)"
                        )

                with gr.Row():
                    but1 = gr.Button("Click to Starting to Convert", variant="primary")
                    vc_output3 = gr.Textbox(label="About All The Output's File...")
                    but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="The Infer Convert Batch...",
                    )
                    import threading

# สร้าง Lock ไว้ด้านบนสุดของ Class หรือไฟล์
processing_lock = threading.Lock()

def vc_multi(...):
    if processing_lock.locked():
        print("งานเก่ากำลังทำอยู่ รอคิวนะจ๊ะ")
        return "Already processing, please wait..."
    
    with processing_lock:
        # Code การแปลงเสียงเดิมๆ ของคุณอยู่ตรงนี้
        ...
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index2, file_index4],
                    api_name="The Inference Voice To Change...",
                )
        with gr.TabItem("2. All Separation, Reverb/Echo, Instrument And Vocal Remover."):
            with gr.Group():
                gr.Markdown(
                    value="Batch vocal/accompaniment separation using UVR5. <br>1. Keep Vocals: Use HP2/HP3. <br>2. Main Vocal Only: Use HP5. <br>3. DeReverb/DeEcho: MDX-Net for dual-channel reverb, DeEcho for delay."
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label="Input Audio Folder Path",
                            placeholder="C:\\Users\\Desktop\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple", label="Upload Audio Files (Optional, folder path takes priority)"
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(label="Model", choices=uvr5_names)
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label="Vocal Extraction Aggressiveness",
                            value=10,
                            interactive=True,
                            visible=False,  
                        )
                        opt_vocal_root = gr.Textbox(
                            label="Output Folder for Main Vocals", value="opt"
                        )
                        opt_ins_root = gr.Textbox(
                            label="Output Folder for Instrumentals/Others", value="opt"
                        )
                        format0 = gr.Radio(
                            label="Export Format",
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                but2 = gr.Button("Convert", variant="primary")
                vc_output4 = gr.Textbox(label="Output Info")
                but2.click(
                    uvr,
                    [
                        model_choose,
                        dir_wav_input,
                        opt_vocal_root,
                        wav_inputs,
                        opt_ins_root,
                        agg,
                        format0,
                    ],
                    [vc_output4],
                    api_name="uvr_convert",
                )
        with gr.TabItem("Training"):
            gr.Markdown(
                value="Step No. 1 : Input For About Model Information And the voice's Rate to Start the Process.."
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label="1.1. Input Model Name For Making..", value="NameAI_1")
                sr2 = gr.Radio(
                    label="Choose For The Sample Rate using...",
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label="1.2. Input To Confirm to Use The Pitch Guidance ( Click True for Singing or Click False for Speech )",
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                version19 = gr.Radio(
                    label="1.3. Input The Model  Version",
                    choices=["v1", "v2"],
                    value="v1",
                    interactive=True,
                    visible=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label="1.4. Input to how much for using the CPU's Threads for This Model Making..",
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Group():  
                gr.Markdown(
                    value="Step No. 2 : Input The Voice's File Address to Checking The Voice's File Using... ( The Spece Using For the Data File is no more other's file without .wav File And No More than 16 Bit and 40,000-48,000 HZ rate for This file. )"
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label="2.1. Input To the Data File's Address For Start The Project...", value="H:\\Data\\Singer_Name"
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label="2.2. Input the Speaker's ID. ( No. 1-4 ).",
                        value=1,
                        interactive=True,
                    )
                but1 = gr.Button("Start the check for the Data File", variant="primary")
                info1 = gr.Textbox(label="Input the Output Info", value="")
                but1.click(
                    preprocess_dataset,
                    [trainset_dir4, exp_dir1, sr2, np7],
                    [info1],
                    api_name="Status_for_the_Process....",
                )
            with gr.Group():
                gr.Markdown(value="Step No. 3 : Input the Engine for Extracking Pitch & Features to This Model Makking.")
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label="3.1. Input About the GPU ID's (dash-separated, e.g., 0-1-2)",
                            value=gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                        gpu_info9 = gr.Textbox(
                            label="About GPU Info...", value=gpu_info, visible=F0GPUVisible
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label="3.2. Click to Choose the Pitch of All Features For Algorithm to This Model Makking.",
                            choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                            value="rmvpe",
                            interactive=True,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label="3.3. Input For RMVPE GPU Configuration's File...",
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                but2 = gr.Button("Start For Extract the Features File...", variant="primary")
                info2 = gr.Textbox(label="Output Info", value="", max_lines=8)
                f0method8.change(
                    fn=change_f0_method,
                    inputs=[f0method8],
                    outputs=[gpus_rmvpe],
                )
                but2.click(
                    extract_f0_feature,
                    [
                        gpus6,
                        np7,
                        f0method8,
                        if_f0_3,
                        exp_dir1,
                        version19,
                        gpus_rmvpe,
                    ],
                    [info2],
                    api_name="Status_for_the_Process....",
                )
            with gr.Group():
                gr.Markdown(value="Step No. 4 : Settings a lot of the Train in Epoch's time  & Start Training.")
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label="4.1. Input The Checkpoint For Save The Model to Testing...( It's Save Model File For Every Checkpoint )",
                        value=40,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label="4.2. Input For Maximum Epoch For this Making The Model.",
                        value=200,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label="4.3. Input for the Batch Size per GPU Using ( Start For 1 )...",
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label="4.4. Use To Save the latest .ckpt File ( To Save The HDD Storage. )",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label="4.5. Use to Save the Vram For The Forever Time to Model ( No More Than 10 min. for that data only )",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label="4.6. Use for Save The No. of Model Checkpoit Time...",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label="4.7. Input File Address for Pre-trained Generator (G) Path..",
                        value="assets/pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label="4.8. Input File Address for Pre-trained Discriminator (D) Path..",
                        value="assets/pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label="4.9. Input No. for GPU IDs ( dash-separated, e.g., 0-1-2 )",
                        value=gpus,
                        interactive=True,
                    )
                but3 = gr.Button("Click to Start the Training To Make This Model Now...", variant="primary")
                but4 = gr.Button("Click to Start Build the Model Index's File...", variant="primary")
                but5 = gr.Button("Click to Start All Building This Model...", variant="primary")
                info3 = gr.Textbox(label="Input for Output Info No.", value="1", max_lines=10)
                but3.click(
                    click_train,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        spk_id5,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                    ],
                    info3,
                    api_name="Status_for_the_Process....",
                )
                but4.click(train_index, [exp_dir1, version19], info3)
                but5.click(
                    train1key,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        trainset_dir4,
                        spk_id5,
                        np7,
                        f0method8,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        gpus_rmvpe,
                    ],
                    info3,
                    api_name="train_start_all",
                )

        with gr.TabItem("Use Checkpoint Processing"):
            with gr.Group():
                gr.Markdown(value="Model Fusion (Merge Models)")
                with gr.Row():
                    ckpt_a = gr.Textbox(label="Model A Path", value="", interactive=True)
                    ckpt_b = gr.Textbox(label="Model B Path", value="", interactive=True)
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label="Model A Weight",
                        value=0.5,
                        interactive=True,
                    )
                with gr.Row():
                    sr_ = gr.Radio(
                        label="Target Sample Rate",
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label="Has Pitch Guidance?",
                        choices=["Yes", "No"],
                        value="Yes",
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label="Model Info to Insert", value="", max_lines=8, interactive=True
                    )
                    name_to_save0 = gr.Textbox(
                        label="Save Name (no extension)",
                        value="NameAI...",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label="Model Version",
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                with gr.Row():
                    but6 = gr.Button("Merge", variant="primary")
                    info4 = gr.Textbox(label="Output Info", value="", max_lines=8)
                but6.click(
                    merge,
                    [
                        ckpt_a,
                        ckpt_b,
                        alpha_a,
                        sr_,
                        if_f0_,
                        info__,
                        name_to_save0,
                        version_2,
                    ],
                    info4,
                    api_name="ckpt_merge",
                ) 
            with gr.Group():
                gr.Markdown(value="Modify Model Info")
                with gr.Row():
                    ckpt_path0 = gr.Textbox(
                        label="Model Path", value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label="Info to Modify", value="", max_lines=8, interactive=True
                    )
                    name_to_save1 = gr.Textbox(
                        label="Save Filename (blank to overwrite)",
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Row():
                    but7 = gr.Button("Modify", variant="primary")
                    info5 = gr.Textbox(label="Output Info", value="", max_lines=8)
                but7.click(
                    change_info,
                    [ckpt_path0, info_, name_to_save1],
                    info5,
                    api_name="ckpt_modify",
                )
            with gr.Group():
                gr.Markdown(value="View Model Info")
                with gr.Row():
                    ckpt_path1 = gr.Textbox(
                        label="Model Path", value="", interactive=True
                    )
                    but8 = gr.Button("View", variant="primary")
                    info6 = gr.Textbox(label="Output Info", value="", max_lines=8)
                but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
            with gr.Group():
                gr.Markdown(
                    value="Extract Small Model from Logs"
                )
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        label="Model Path",
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label="Save Name", value="", interactive=True
                    )
                    sr__ = gr.Radio(
                        label="Target Sample Rate",
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0__ = gr.Radio(
                        label="Pitch Guidance (1: Yes, 0: No)",
                        choices=["1", "0"],
                        value="1",
                        interactive=True,
                    )
                    version_1 = gr.Radio(
                        label="Model Version",
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label="Model Info to Insert", value="", max_lines=8, interactive=True
                    )
                    but9 = gr.Button("Extract", variant="primary")
                    info7 = gr.Textbox(label="Output Info", value="", max_lines=8)
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
                but9.click(
                    extract_small_model,
                    [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                    info7,
                    api_name="ckpt_extract",
                )

        with gr.TabItem("ONNX Export"):
            with gr.Row():
                ckpt_dir = gr.Textbox(label="RVC Model Path", value="", interactive=True)
            with gr.Row():
                onnx_dir = gr.Textbox(
                    label="ONNX Output Path", value="", interactive=True
                )
            with gr.Row():
                infoOnnx = gr.Label(label="info")
            with gr.Row():
                butOnnx = gr.Button("Export ONNX Model", variant="primary")
            butOnnx.click(
                export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
            )

        with gr.TabItem("FAQ"):
            try:
                with open("docs/en/faq_en.md", "r", encoding="utf8") as f:
                    info = f.read()
                gr.Markdown(value=info)
            except:
                gr.Markdown("FAQ file not found. Please ensure docs/en/faq_en.md exists.")

   # ==========================================
    # ส่วนแก้ไขปัญหา Gradio Timeout (ReadTimeout)
    # ==========================================
    import gradio.networking
    
    # สร้างฟังก์ชันดักจับ (Monkey Patch) เพื่อข้ามการเช็ค URL ที่ทำให้เกิด Timeout
    original_url_ok = gradio.networking.url_ok
    def custom_url_ok(url):
        try:
            return original_url_ok(url)
        except Exception:
            # ถ้าโหลดเกิน 3 วินาทีจน Error ให้บังคับ return True เพื่อให้รันหน้าเว็บต่อได้เลย ไม่ต้องแครช
            return True 
            
    gradio.networking.url_ok = custom_url_ok

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            
            # --- วิธีเปลี่ยน Port ---
            # หากรันแล้วยังมีปัญหาพอร์ตค้างหรือชนกัน ให้ลบเครื่องหมาย # หน้าบรรทัด server_port=9378, ออก
            # และใส่ # หน้าบรรทัด server_port=config.listen_port, แทนครับ
            # server_port=9378, 
            
            server_port=config.listen_port,
            quiet=True,
        )
