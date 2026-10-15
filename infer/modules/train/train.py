# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : train.py
# เวอร์ชั่นโค้ด (Version)      : 3.5 (Real-Time File Log Sync, Direct Single-Process & NaN Loss Guard)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/train.py
# หน้าที่ (Function/Role)     : สคริปต์หลักในการฝึกสอนโมเดล RVC v1/v2
#                             อัปเกรดให้เขียนสถานะลง train.log ทันทีเพื่อให้ Program.py แสดงผล Real-time,
#                             ตัดการสร้าง Subprocess ซ้อนเมื่อรันบน CPU/1-GPU และป้องกันปัญหา NaN Loss
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================

import os
import sys

os.environ["USE_LIBUV"] = "0"

import logging
import warnings
warnings.filterwarnings("ignore")
import datetime

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)
PYTHON_EXE = os.path.join(now_dir, "runtime", "python.exe")

from random import randint
from time import sleep, time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import PoliteLogger
from infer.lib.train import utils
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from infer.lib.train import losses, mel_processing, process_ckpt
from infer.lib.infer_pack import commons
from infer.modules.train.rvc_train_utils import (
    CheckpointHelper,
    DeviceHelper,
    GpuDataCache,
    load_pretrained,
)

logger = logging.getLogger(__name__)
global_step = 0
ACTIVE_LOG_FILE = ""


def cmd_notify(prefix: str, msg: str, model_dir: str = ""):
    global ACTIVE_LOG_FILE
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    formatted = f"[{ts}] [{prefix}] {msg}"
    try:
        print(formatted, flush=True)
    except Exception:
        pass
    target_log = os.path.join(model_dir, "train.log") if model_dir else ACTIVE_LOG_FILE
    if target_log:
        try:
            with open(target_log, "a+", encoding="utf-8", errors="replace") as lf:
                lf.write(formatted + "\n")
        except Exception:
            pass


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ใช้เวลาต่อรอบ (Elapsed): {elapsed_time_str}"


def _create_models(hps, device):
    if hps.version == "v1":
        from infer.lib.infer_pack.models import (
            MultiPeriodDiscriminator,
            SynthesizerTrnMs256NSFsid as RVC_Model_f0,
            SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
        )
    else:
        from infer.lib.infer_pack.models import (
            SynthesizerTrnMs768NSFsid as RVC_Model_f0,
            SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
        )
    common_kwargs = {"is_half": hps.train.fp16_run, **hps.model}
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **common_kwargs,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **common_kwargs,
        )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    return net_g.to(device), net_d.to(device)


def _create_train_loader(hps, rank, n_gpus):
    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
        collate_fn = TextAudioCollate()
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    has_cuda = torch.cuda.is_available()
    num_w = min(4, max(0, (os.cpu_count() or 2) // 2)) if has_cuda else 0
    loader_kwargs = {
        "num_workers": num_w,
        "shuffle": False,
        "pin_memory": has_cuda,
        "collate_fn": collate_fn,
        "batch_sampler": train_sampler,
    }
    if num_w > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(train_dataset, **loader_kwargs)


def main():
    global ACTIVE_LOG_FILE
    hps = utils.get_hparams()
    os.makedirs(hps.model_dir, exist_ok=True)
    ACTIVE_LOG_FILE = os.path.join(hps.model_dir, "train.log")

    if hps.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    gpu_list = [g for g in str(hps.gpus).split("-") if g.strip() != ""]
    n_gpus = len(gpu_list) if torch.cuda.is_available() else 1
    visible = torch.cuda.device_count()
    if torch.cuda.is_available() and n_gpus > visible:
        n_gpus = max(1, visible)
    if not torch.cuda.is_available():
        hps.train.fp16_run = False
        n_gpus = 1

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    main_logger = utils.get_logger(hps.model_dir)
    cmd_notify("SYSTEM", "=" * 65, hps.model_dir)
    cmd_notify("SYSTEM", "🚀 เริ่มต้นระบบฝึกสอนโมเดล RVC v3.5 (Starting RVC Model Training)", hps.model_dir)
    cmd_notify("SYSTEM", f"• ชื่อโมเดล (Model Name): {hps.name} (Version: {hps.version.upper()})", hps.model_dir)
    cmd_notify("SYSTEM", f"• โฟลเดอร์จัดเก็บ (Model Dir): {hps.model_dir}", hps.model_dir)
    cmd_notify("SYSTEM", f"• อุปกรณ์ที่ใช้ (Device): {'CUDA GPU (' + str(hps.gpus) + ')' if torch.cuda.is_available() else 'CPU Mode'} ({n_gpus} Worker)", hps.model_dir)
    cmd_notify("SYSTEM", f"• พารามิเตอร์ (Params): SR {hps.data.sampling_rate} Hz | Batch {hps.train.batch_size} | Epochs {hps.train.epochs} | FP16: {hps.train.fp16_run}", hps.model_dir)
    cmd_notify("SYSTEM", "=" * 65, hps.model_dir)

    # หากใช้ 1 GPU หรือ CPU ให้รันโดยตรงไม่ต้อง স্পอน (spawn) โปรเซสซ้อนเพื่อความรวดเร็วและเสถียรสูงสุด
    if n_gpus == 1:
        run(0, 1, hps, main_logger, use_ddp=False)
    else:
        children = []
        for i in range(n_gpus):
            subproc = mp.Process(target=run, args=(i, n_gpus, hps, main_logger, True))
            children.append(subproc)
            subproc.start()
        for subproc in children:
            subproc.join()


def run(rank, n_gpus, hps, logger: logging.Logger, use_ddp: bool = False):
    global global_step, ACTIVE_LOG_FILE
    ACTIVE_LOG_FILE = os.path.join(hps.model_dir, "train.log")
    device = DeviceHelper.get_device(rank)

    if use_ddp:
        backend = DeviceHelper.select_ddp_backend()
        dist.init_process_group(backend=backend, init_method="env://", world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    DeviceHelper.set_device(rank)

    writer = SummaryWriter(log_dir=hps.model_dir) if rank == 0 else None

    train_loader = _create_train_loader(hps, rank, n_gpus)
    if rank == 0:
        total_samples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
        cmd_notify("DATA", f"📦 โหลดชุดข้อมูลสำเร็จ (Dataset Loaded) จำนวน {total_samples} รายการ ({len(train_loader)} batches ต่อ Epoch)", hps.model_dir)
        if len(train_loader) == 0:
            cmd_notify("ERROR", "❌ จำนวน Batch เป็น 0 (อาจเกิดจากจำนวนไฟล์น้อยกว่า Batch Size) กรุณาลด Batch Size ลง", hps.model_dir)
            sys.exit(1)

    net_g, net_d = _create_models(hps, device)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    if use_ddp and n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank]) if torch.cuda.is_available() else DDP(net_g)
        net_d = DDP(net_d, device_ids=[rank]) if torch.cuda.is_available() else DDP(net_d)

    GradScaler, autocast = DeviceHelper.setup_amp()
    scaler = GradScaler(enabled=hps.train.fp16_run)

    epoch_str, global_step = 1, 0
    try:
        _, _, _, epoch_str = CheckpointHelper.load_checkpoint(
            CheckpointHelper.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str = CheckpointHelper.load_checkpoint(
            CheckpointHelper.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        if rank == 0:
            cmd_notify("MODEL", f"✅ โหลด Checkpoint เดิมสำเร็จ สานต่อจาก Epoch ที่ {epoch_str} (Step {global_step})", hps.model_dir)
    except Exception:
        load_pretrained(net_g, hps.pretrainG, logger)
        load_pretrained(net_d, hps.pretrainD, logger)
        if rank == 0:
            cmd_notify("MODEL", "✅ โหลดโมเดลพื้นฐาน Pretrained G & D สำเร็จ เริ่มต้นที่ Epoch 1", hps.model_dir)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    cache = GpuDataCache(enabled=hps.if_cache_data_in_gpu, use_f0=hps.if_f0 == 1, device=device)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        net_g.train()
        net_d.train()
        train_loader.batch_sampler.set_epoch(epoch)
        data_iterator = cache.iter_epoch(train_loader, epoch)
        epoch_recorder = EpochRecorder()

        for batch_idx, info in data_iterator:
            if hps.if_f0 == 1:
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info
            else:
                phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info

            if not cache:
                phone = phone.to(device, non_blocking=True)
                phone_lengths = phone_lengths.to(device, non_blocking=True)
                sid = sid.to(device, non_blocking=True)
                spec = spec.to(device, non_blocking=True)
                spec_lengths = spec_lengths.to(device, non_blocking=True)
                wave = wave.to(device, non_blocking=True)
                wave_lengths = wave_lengths.to(device, non_blocking=True)
                if hps.if_f0 == 1:
                    pitch = pitch.to(device, non_blocking=True)
                    pitchf = pitchf.to(device, non_blocking=True)

            with autocast(enabled=hps.train.fp16_run):
                if hps.if_f0 == 1:
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                    )
                else:
                    y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                        phone, phone_lengths, spec, spec_lengths, sid
                    )

                mel = mel_processing.spec_to_mel_torch(
                    spec, hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax
                )
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)

                with autocast(enabled=False):
                    y_hat_mel = mel_processing.mel_spectrogram_torch(
                        y_hat.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                        hps.data.mel_fmin, hps.data.mel_fmax
                    )
                if hps.train.fp16_run:
                    y_hat_mel = y_hat_mel.half()
                wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = losses.discriminator_loss(y_d_hat_r, y_d_hat_g)

            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = losses.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = losses.feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = losses.generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            # ป้องกันกรณีเกิด NaN/Inf Loss ไม่ให้อัปเดตค่าน้ำหนักที่เพี้ยนลงโมเดล
            if torch.isnan(loss_gen_all) or torch.isinf(loss_gen_all):
                cmd_notify("WARN", f"⚠️ ตรวจพบค่า NaN/Inf Loss ที่ Step {global_step} ข้ามการอัปเดต Batch นี้เพื่อปกป้องโมเดล", hps.model_dir)
                optim_g.zero_grad()
                scaler.update()
                global_step += 1
                continue

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if rank == 0 and global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                log_mel, log_kl = min(loss_mel.item(), 75.0), min(loss_kl.item(), 9.0)
                batch_pct = 100.0 * (batch_idx + 1) / len(train_loader) if len(train_loader) > 0 else 0.0
                cmd_notify(
                    "TRAIN",
                    f"📈 Epoch [{epoch}/{hps.train.epochs} | {batch_pct:.1f}%] Step {global_step} | LR {lr:.2e} | "
                    f"Loss(G)={loss_gen.item():.3f} | Loss(D)={loss_disc.item():.3f} | Mel={log_mel:.3f} | KL={log_kl:.3f}",
                    hps.model_dir
                )
                if writer is not None:
                    scalar_dict = {
                        "loss/g/total": float(loss_gen_all.item()),
                        "loss/d/total": float(loss_disc.item()),
                        "loss/g/fm": float(loss_fm.item()),
                        "loss/g/mel": float(log_mel),
                        "loss/g/kl": float(log_kl),
                        "learning_rate": float(lr),
                    }
                    for k, v in scalar_dict.items():
                        writer.add_scalar(k, v, global_step)
            global_step += 1

        scheduler_g.step()
        scheduler_d.step()

        if epoch % hps.save_every_epoch == 0 and rank == 0:
            g_path = os.path.join(hps.model_dir, "G_{}.pth".format(global_step if hps.if_latest == 0 else 2333333))
            d_path = os.path.join(hps.model_dir, "D_{}.pth".format(global_step if hps.if_latest == 0 else 2333333))
            CheckpointHelper.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_path)
            CheckpointHelper.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_path)
            if str(hps.save_every_weights) == "1":
                ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
                saved_path = process_ckpt.savee(
                    ckpt, hps.sample_rate, hps.if_f0, f"{hps.name}_e{epoch}_s{global_step}", epoch, hps.version, hps
                )
                cmd_notify("SAVE", f"💾 [บันทึกโมเดลย่อยสำเร็จ (Saved)] Epoch {epoch}/{hps.train.epochs} -> {saved_path}", hps.model_dir)

        if rank == 0:
            rec_data = epoch_recorder.record()
            overall_pct = (epoch / hps.train.epochs) * 100.0 if hps.train.epochs > 0 else 0.0
            cmd_notify("EPOCH", f"🔄 [จบ Epoch {epoch}/{hps.train.epochs}] คืบหน้ารวม {overall_pct:.1f}% | {rec_data}", hps.model_dir)

        if epoch >= hps.train.epochs and rank == 0:
            ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
            final_path = process_ckpt.savee(ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps)
            cmd_notify("FINISH", "=" * 65, hps.model_dir)
            cmd_notify("FINISH", "🎉 การฝึกสอนโมเดล RVC เสร็จสิ้นสมบูรณ์ 100%! (Training Completed Successfully)", hps.model_dir)
            cmd_notify("FINISH", f"• ไฟล์โมเดลตัวสุดท้าย (Final Weights): {final_path}", hps.model_dir)
            cmd_notify("FINISH", "=" * 65, hps.model_dir)
            sleep(1)
            if use_ddp and dist.is_initialized():
                dist.destroy_process_group()
            sys.exit(0)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()