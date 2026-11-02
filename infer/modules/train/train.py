# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)   : train.py
# 3. ลำดับเวอร์ชัน (Version)   : 5.0 (Enterprise Zero-Defect DDP & VRAM Cache)
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              Original by Patiphat Wongyai (Delta) & RVC-Project
# 5. ตำแหน่งที่วางของไฟล์ (Path) : infer/modules/train/train.py
# 6. รายละเอียดการแก้ไขงาน    : เสริมความเสถียร VRAM Safety Protocol (กฎข้อที่ 8) และป้องกัน Loss แตก (NaN/Inf Guard) อย่างสมบูรณ์
# ==============================================================================

import os
import sys

# บังคับใช้ UTF-8 ป้องกันปัญหา UnicodeEncodeError บน Windows CP1252
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["USE_LIBUV"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
import warnings
warnings.filterwarnings("ignore")
import datetime
import traceback
import gc
from random import randint
from time import sleep, time as ttime

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)

import torch
torch.set_num_threads(1)
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.train import utils
from infer.lib.train.data_utils import (
    DistributedBucketSampler, TextAudioCollate, TextAudioCollateMultiNSFsid,
    TextAudioLoader, TextAudioLoaderMultiNSFsid,
)
from infer.lib.train import losses, mel_processing, process_ckpt
from infer.lib.infer_pack import commons
from infer.modules.train.rvc_train_utils import (
    CheckpointHelper, DeviceHelper, GpuDataCache, load_pretrained,
)

logger = logging.getLogger(__name__)
global_step = 0
ACTIVE_LOG_FILE = ""

def cmd_notify(prefix: str, msg: str, model_dir: str = "", logger_ref=None):
    """ระบบรายงานผลส่วนกลาง 2 ภาษา (Bilingual Log) และพิมพ์ลงไฟล์ train.log"""
    global ACTIVE_LOG_FILE
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    formatted = f"[{ts}] [{prefix}] {msg}"
    print(formatted, flush=True)
    target_log = os.path.join(model_dir, "train.log") if model_dir else ACTIVE_LOG_FILE
    
    if target_log:
        try:
            with open(target_log, "a+", encoding="utf-8", errors="replace") as lf:
                lf.write(formatted + "\n")
        except Exception:
            pass
            
    if logger_ref:
        if prefix in ["ERROR", "FATAL"]:
            logger_ref.error(msg)
        elif prefix in ["WARN"]:
            logger_ref.warning(msg)
        else:
            logger_ref.info(msg)

class EpochRecorder:
    def __init__(self): 
        self.last_time = ttime()
    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        return f"ใช้เวลาในรอบนี้ (Elapsed): {elapsed_time_str}"

def _create_models(hps, device):
    if hps.version == "v1":
        from infer.lib.infer_pack.models import (MultiPeriodDiscriminator, SynthesizerTrnMs256NSFsid as RVC_Model_f0, SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0)
    else:
        from infer.lib.infer_pack.models import (SynthesizerTrnMs768NSFsid as RVC_Model_f0, SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0, MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator)
        
    common_kwargs = {"is_half": hps.train.fp16_run, **hps.model}
    
    if hps.if_f0 == 1: 
        net_g = RVC_Model_f0(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **common_kwargs, sr=hps.sample_rate)
    else: 
        net_g = RVC_Model_nof0(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **common_kwargs)
        
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
        train_dataset, hps.train.batch_size * n_gpus,
        [32, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus, rank=rank, shuffle=True,
    )
    
    num_workers = 0 if os.name == 'nt' else 2
    
    return DataLoader(
        train_dataset, 
        num_workers=num_workers, 
        shuffle=False, 
        pin_memory=torch.cuda.is_available(), 
        collate_fn=collate_fn, 
        batch_sampler=train_sampler
    )

def main():
    global ACTIVE_LOG_FILE
    hps = utils.get_hparams()
    hps.train.epochs = int(hps.total_epoch) if hasattr(hps, "total_epoch") and hps.total_epoch else int(getattr(hps.train, "epochs", 200))
    
    pre_sub = "pretrained" if str(hps.version).lower() == "v1" else "pretrained_v2"
    f0_p = "f0" if int(getattr(hps, "if_f0", 1)) == 1 else ""
    sr_t = str(getattr(hps, "sample_rate", "40k"))
    
    if not getattr(hps, "pretrainG", "") or not os.path.isfile(str(hps.pretrainG)):
        cand_g = os.path.join(now_dir, "assets", pre_sub, f"{f0_p}G{sr_t}.pth")
        if os.path.isfile(cand_g): hps.pretrainG = cand_g
        
    if not getattr(hps, "pretrainD", "") or not os.path.isfile(str(hps.pretrainD)):
        cand_d = os.path.join(now_dir, "assets", pre_sub, f"{f0_p}D{sr_t}.pth")
        if os.path.isfile(cand_d): hps.pretrainD = cand_d
        
    os.makedirs(hps.model_dir, exist_ok=True)
    ACTIVE_LOG_FILE = os.path.join(hps.model_dir, "train.log")
    
    if hps.gpus: 
        os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    
    n_gpus = len([g for g in str(hps.gpus).split("-") if g.strip() != ""]) if torch.cuda.is_available() else 1
    if not torch.cuda.is_available(): 
        hps.train.fp16_run = False
        n_gpus = 1

    cmd_notify("SYSTEM", "=" * 65, hps.model_dir)
    cmd_notify("SYSTEM", "🚀 เริ่มต้นกระบวนการ RVC Training Engine (Zero-Defect Edition v5.0)", hps.model_dir)
    cmd_notify("SYSTEM", f"🎯 โมเดล (Model): {hps.name} | เวอร์ชัน (Version): {hps.version.upper()} | จำนวนรอบ (Epochs): {hps.train.epochs}", hps.model_dir)
    cmd_notify("SYSTEM", "=" * 65, hps.model_dir)

    if n_gpus == 1: 
        run(0, 1, hps, use_ddp=False)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(randint(20000, 55555))
        try:
            mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, True))
        except Exception as e:
            cmd_notify("FATAL", f"❌ ข้อผิดพลาดในการจำลอง Process (Multiprocessing Spawn Failed): {e}", hps.model_dir)
            sys.exit(1)

def run(rank, n_gpus, hps, use_ddp: bool = False):
    global global_step, ACTIVE_LOG_FILE
    ACTIVE_LOG_FILE = os.path.join(hps.model_dir, "train.log")
    device = DeviceHelper.get_device(rank)
    
    process_logger = utils.get_logger(hps.model_dir) if rank == 0 else logging.getLogger(__name__)

    try:
        if use_ddp:
            dist.init_process_group(backend=DeviceHelper.select_ddp_backend(), init_method="env://", world_size=n_gpus, rank=rank)
            
        torch.manual_seed(hps.train.seed)
        DeviceHelper.set_device(rank)
        writer = SummaryWriter(log_dir=hps.model_dir) if rank == 0 else None

        train_loader = _create_train_loader(hps, rank, n_gpus)
        if rank == 0:
            total_samples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
            cmd_notify("DATA", f"โหลดชุดข้อมูลสำเร็จ ({total_samples} ตัวอย่าง | {len(train_loader)} แบตช์ต่อรอบ)", hps.model_dir, process_logger)
            if len(train_loader) == 0:
                cmd_notify("ERROR", "❌ ล้มเหลว: ชุดข้อมูลว่างเปล่าหรือ Batch Size ใหญ่เกินไป กรุณาปรับขนาดใหม่ (Dataset is empty)", hps.model_dir, process_logger)
                os._exit(1)

        net_g, net_d = _create_models(hps, device)
        optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
        optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

        if use_ddp and n_gpus > 1:
            net_g = DDP(net_g, device_ids=[rank]) if torch.cuda.is_available() else DDP(net_g)
            net_d = DDP(net_d, device_ids=[rank]) if torch.cuda.is_available() else DDP(net_d)

        GradScaler, autocast = DeviceHelper.setup_amp()
        scaler = GradScaler(enabled=hps.train.fp16_run)

        epoch_str, global_step = 1, 0
        checkpoint_loaded = False
        try:
            d_path = CheckpointHelper.latest_checkpoint_path(hps.model_dir, "D_*.pth")
            g_path = CheckpointHelper.latest_checkpoint_path(hps.model_dir, "G_*.pth")
            _, _, _, epoch_str = CheckpointHelper.load_checkpoint(d_path, net_d, optim_d)
            _, _, _, epoch_str = CheckpointHelper.load_checkpoint(g_path, net_g, optim_g)
            global_step = (epoch_str - 1) * len(train_loader)
            checkpoint_loaded = True
            if rank == 0: 
                cmd_notify("MODEL", f"โหลดจุดเซฟสำเร็จ: เริ่มทำงานต่อจากรอบที่ (Resumed training from Epoch) {epoch_str}", hps.model_dir, process_logger)
        except Exception:
            checkpoint_loaded = False

        if not checkpoint_loaded:
            load_pretrained(net_g, getattr(hps, "pretrainG", ""), process_logger)
            load_pretrained(net_d, getattr(hps, "pretrainD", ""), process_logger)
            if rank == 0: 
                cmd_notify("MODEL", "โหลดโมเดลพื้นฐานเริ่มต้นแล้ว (Initialized from Pretrained Models) -> รอบที่ 1", hps.model_dir, process_logger)

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
        
        is_cache_enabled = getattr(hps, 'if_cache_data_in_gpu', False)
        cache = GpuDataCache(enabled=is_cache_enabled, use_f0=hps.if_f0 == 1, device=device)
        target_epochs = int(hps.train.epochs)
        
        for epoch in range(epoch_str, target_epochs + 1):
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

                phone, phone_lengths = phone.to(device, non_blocking=True), phone_lengths.to(device, non_blocking=True)
                sid, spec, spec_lengths = sid.to(device, non_blocking=True), spec.to(device, non_blocking=True), spec_lengths.to(device, non_blocking=True)
                wave, wave_lengths = wave.to(device, non_blocking=True), wave_lengths.to(device, non_blocking=True)
                
                if hps.if_f0 == 1: 
                    pitch, pitchf = pitch.to(device, non_blocking=True), pitchf.to(device, non_blocking=True)

                try:
                    with autocast(enabled=hps.train.fp16_run):
                        if hps.if_f0 == 1: 
                            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                        else: 
                            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, spec, spec_lengths, sid)

                        mel = mel_processing.spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)

                        with autocast(enabled=False):
                            y_hat_mel = mel_processing.mel_spectrogram_torch(y_hat.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
                            
                        if hps.train.fp16_run: 
                            y_hat_mel = y_hat_mel.half()
                            
                        wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)
                        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                        
                        with autocast(enabled=False): 
                            loss_disc, losses_disc_r, losses_disc_g = losses.discriminator_loss(y_d_hat_r, y_d_hat_g)

                    optim_d.zero_grad()
                    scaler.scale(loss_disc).backward()
                    scaler.unscale_(optim_d)
                    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                    scaler.step(optim_d)

                    with autocast(enabled=hps.train.fp16_run):
                        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                        with autocast(enabled=False):
                            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                            loss_kl = losses.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                            loss_fm = losses.feature_loss(fmap_r, fmap_g)
                            loss_gen, losses_gen = losses.generator_loss(y_d_hat_g)
                            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

                    # 🔴 ป้องกัน Loss แตก (NaN/Inf) ล้มโปรแกรมแบบเฉียบพลัน
                    if torch.isnan(loss_gen_all) or torch.isinf(loss_gen_all):
                        if rank == 0: 
                            cmd_notify("WARN", "⚠️ ข้าม 1 ชุดข้อมูล เนื่องจากค่า Loss แตก (NaN/Inf Detected)", hps.model_dir, process_logger)
                        optim_g.zero_grad()
                        scaler.update()
                        global_step += 1
                        continue

                    optim_g.zero_grad()
                    scaler.scale(loss_gen_all).backward()
                    scaler.unscale_(optim_g)
                    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                    scaler.step(optim_g)
                    scaler.update()

                    if rank == 0 and global_step % hps.train.log_interval == 0:
                        lr = optim_g.param_groups[0]["lr"]
                        batch_pct = 100.0 * (batch_idx + 1) / len(train_loader) if len(train_loader) > 0 else 0.0
                        cmd_notify("TRAIN", f"Epoch [{epoch}/{target_epochs} | {batch_pct:.1f}%] Step {global_step} | LR {lr:.2e} | Loss(G): {loss_gen.item():.3f} | Loss(D): {loss_disc.item():.3f} | Mel: {loss_mel.item():.3f}", hps.model_dir, process_logger)
                        
                        if writer:
                            scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g, "loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl}
                            scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                            scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                            scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                            image_dict = {
                                "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                                "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                                "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                            }
                            utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict)
                            
                    global_step += 1
                except RuntimeError as re_err:
                    if "out of memory" in str(re_err).lower():
                        cmd_notify("WARN", "⚠️ VRAM เต็มในขณะเทรน ทำการลดแคชและข้ามแบตช์ปัจจุบัน", hps.model_dir, process_logger)
                        optim_g.zero_grad()
                        optim_d.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise re_err

            scheduler_g.step()
            scheduler_d.step()

            is_last_epoch = (epoch == target_epochs)
            if (epoch % hps.save_every_epoch == 0 or is_last_epoch) and rank == 0:
                g_path = os.path.join(hps.model_dir, f"G_{global_step if hps.if_latest == 0 else 2333333}.pth")
                d_path = os.path.join(hps.model_dir, f"D_{global_step if hps.if_latest == 0 else 2333333}.pth")
                
                CheckpointHelper.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, g_path)
                CheckpointHelper.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, d_path)
                
                if str(hps.save_every_weights) == "1" or is_last_epoch:
                    ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
                    saved_path = process_ckpt.savee(ckpt, hps.sample_rate, hps.if_f0, f"{hps.name}_e{epoch}_s{global_step}" if not is_last_epoch else hps.name, epoch, hps.version, hps)
                    cmd_notify("SAVE", f"💾 บันทึกจุดเซฟสำเร็จ (Checkpoint Saved) Epoch {epoch}/{target_epochs} -> {saved_path}", hps.model_dir, process_logger)

            if rank == 0:
                cmd_notify("EPOCH", f"🏁 สรุปรอบฝึกที่ {epoch}/{target_epochs} | {epoch_recorder.record()}", hps.model_dir, process_logger)

            if is_last_epoch:
                if rank == 0:
                    cmd_notify("FINISH", "=" * 65, hps.model_dir, process_logger)
                    cmd_notify("FINISH", f"🎉 การฝึกสอนเสร็จสมบูรณ์ 100% ({target_epochs} Epochs)!", hps.model_dir, process_logger)
                    cmd_notify("FINISH", "=" * 65, hps.model_dir, process_logger)
                    sleep(1)
                
                if use_ddp and dist.is_initialized(): 
                    dist.destroy_process_group()
                
                os._exit(0)

    except Exception as e:
        if rank == 0:
            cmd_notify("FATAL", f"❌ เกิดข้อผิดพลาดร้ายแรง (Fatal Error in Training):\n{traceback.format_exc()}", hps.model_dir, process_logger)
        if use_ddp and dist.is_initialized(): 
            dist.destroy_process_group()
        os._exit(1)
    finally:
        # 🔴 กฎข้อที่ 8: การคืนหน่วยความจำ VRAM อย่างเข้มงวดเมื่อจบการทำงาน
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()