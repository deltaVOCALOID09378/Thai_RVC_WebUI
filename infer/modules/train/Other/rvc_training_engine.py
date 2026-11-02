# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : train.py
# ฟังก์ชันหลัก (Function)      : RVC Training Engine (Master Subprocess)
# เวอร์ชั่นโค้ด (Version)      : 4.5 (Zero-Defect & Robust Logging Edition)
# การแก้ไข (Modifications)    : 
#   - แก้ไขบั๊กชุดข้อมูลว่าง (Empty Dataloader) ที่ทำให้ Epoch ไม่แสดง
#   - เพิ่มระบบดักจับ Error จากโปรเซสย่อยและแสดงผลลัพธ์ลง train.log ทันที
#   - บังคับแสดงผล Log อย่างน้อย 1 ครั้งในทุกๆ Epoch เพื่อไม่ให้หน้า UI ค้าง
#   - ปรับลด num_workers เพื่อป้องกันคอขวดบน Windows
#   - เปลี่ยนรหัสการจบโปรแกรมให้เป็น Code: 0 เพื่อรายงานความสำเร็จให้ถูกต้อง
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
# ==============================================================================

import os
import sys
import logging
import datetime
import traceback
from random import randint, shuffle
from time import sleep, time as ttime

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.train import utils

# โหลดพารามิเตอร์การตั้งค่าการฝึกสอน
hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))

# การตรวจสอบและจำลองสภาพแวดล้อมสำหรับ Intel / AMD หรือ GPU พิเศษ
try:
    import intel_extension_for_pytorch as ipex
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.cuda.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

# คัดเลือกสถาปัตยกรรมโมเดล (V1 / V2) ตามผู้ใช้งานกำหนด
if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee

global_step = 0
logger = logging.getLogger(__name__)

class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"[{current_time}] | (⏱️ {elapsed_time_str})"


def main():
    """ตัวจัดการโปรเซสหลัก (Master Process) สำหรับควบคุม Worker ทั้งหมด"""
    n_gpus = torch.cuda.device_count()

    if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        n_gpus = 1
    if n_gpus < 1:
        print("[System Warning] ไม่พบการ์ดจอ Nvidia: ระบบกำลังสลับไปใช้ CPU (NO NVIDIA GPU DETECTED: falling back to CPU)")
        n_gpus = 1
        
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    
    children = []
    # เปิดการทำงานของโปรเซสย่อย (Workers)
    for i in range(n_gpus):
        subproc = mp.Process(target=run, args=(i, n_gpus, hps))
        children.append(subproc)
        subproc.start()

    # ตรวจจับข้อผิดพลาด (Failure Gate) - ป้องกันการแครชแบบเงียบๆ
    has_error = False
    for i in range(n_gpus):
        children[i].join()
        if children[i].exitcode != 0:
            has_error = True
            
    if has_error:
        print("❌ กระบวนการฝึกสอนถูกระงับเนื่องจากเกิดข้อผิดพลาดในโปรเซสย่อย (Training aborted due to subprocess error).")
        sys.exit(1)


def run(rank, n_gpus, hps):
    """กระบวนการฝึกสอนภายในของแต่ละโปรเซส (Worker Process)"""
    global global_step
    logger_instance = None
    writer = None
    writer_eval = None
    
    if rank == 0:
        logger_instance = utils.get_logger(hps.model_dir)
        logger_instance.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    try:
        dist.init_process_group(backend="gloo", init_method="env://", world_size=n_gpus, rank=rank)
        torch.manual_seed(hps.train.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        # เตรียมชุดข้อมูล
        if hps.if_f0 == 1:
            train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
        else:
            train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
            
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size * n_gpus,
            [100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        
        if hps.if_f0 == 1:
            collate_fn = TextAudioCollateMultiNSFsid()
        else:
            collate_fn = TextAudioCollate()
            
        # ควบคุม num_workers และ prefetch ให้เหมาะสม ป้องกันคอขวดบน Windows
        train_loader = DataLoader(
            train_dataset,
            num_workers=2, 
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=4,
        )

        # 🚧 ป้องกัน Zero Dataloader Bug ที่ทำให้ไม่เห็น Epoch
        if len(train_loader) == 0:
            err_msg = "❌ ล้มเหลว: ชุดข้อมูลมีขนาดเล็กกว่า Batch Size! กรุณาปรับลด Batch Size ลง หรือเพิ่มจำนวนข้อมูลเสียง (Dataset is too small for the given batch size)"
            if rank == 0 and logger_instance is not None:
                logger_instance.error(err_msg)
            raise ValueError(err_msg)

        # ตั้งค่าโมเดล G และ D
        if hps.if_f0 == 1:
            net_g = RVC_Model_f0(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
                is_half=hps.train.fp16_run,
                sr=hps.sample_rate,
            )
        else:
            net_g = RVC_Model_nof0(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
                is_half=hps.train.fp16_run,
            )
            
        if torch.cuda.is_available():
            net_g = net_g.cuda(rank)
            
        net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
        if torch.cuda.is_available():
            net_d = net_d.cuda(rank)
            
        optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
        optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            pass
        elif torch.cuda.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

        # ดึงน้ำหนักโมเดลกลับมาสานต่อ
        try:
            _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
            if rank == 0: logger_instance.info("[Checkpoint] โหลดโมเดล D สำเร็จ (Loaded D)")
            _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
            global_step = (epoch_str - 1) * len(train_loader)
        except Exception:
            epoch_str = 1
            global_step = 0
            if hps.pretrainG != "":
                if rank == 0: logger_instance.info(f"[Pretrain] โหลดโมเดลพื้นฐาน G (Loaded pretrained): {hps.pretrainG}")
                target_net = net_g.module if hasattr(net_g, "module") else net_g
                target_net.load_state_dict(torch.load(hps.pretrainG, map_location="cpu")["model"])
                
            if hps.pretrainD != "":
                if rank == 0: logger_instance.info(f"[Pretrain] โหลดโมเดลพื้นฐาน D (Loaded pretrained): {hps.pretrainD}")
                target_net_d = net_d.module if hasattr(net_d, "module") else net_d
                target_net_d.load_state_dict(torch.load(hps.pretrainD, map_location="cpu")["model"])

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
        scaler = GradScaler(enabled=hps.train.fp16_run)
        
        cache = []
        for epoch in range(epoch_str, hps.train.epochs + 1):
            if rank == 0:
                train_and_evaluate(
                    rank, epoch, hps, [net_g, net_d], [optim_g, optim_d],
                    [scheduler_g, scheduler_d], scaler, [train_loader, None],
                    logger_instance, [writer, writer_eval], cache
                )
            else:
                train_and_evaluate(
                    rank, epoch, hps, [net_g, net_d], [optim_g, optim_d],
                    [scheduler_g, scheduler_d], scaler, [train_loader, None],
                    None, None, cache
                )
            scheduler_g.step()
            scheduler_d.step()
            
    except Exception as e:
        # ระบบดักจับการแครชแบบเงียบ และเขียนข้อมูลความขัดข้องลงใน Logs ทันที
        if rank == 0 and logger_instance is not None:
            logger_instance.error(f"❌ เกิดข้อผิดพลาดร้ายแรงระหว่างการฝึกสอน (Fatal Error in Training):\n{traceback.format_exc()}")
        raise e


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger_ref, writers, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # ระบบจัดการข้อมูลแคช (Caching)
    if hps.if_cache_data_in_gpu == True:
        data_iterator = cache
        if not cache:
            for batch_idx, info in enumerate(train_loader):
                if hps.if_f0 == 1:
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info
                else:
                    phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
                    
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                    
                if hps.if_f0 == 1:
                    cache.append((batch_idx, (phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid)))
                else:
                    cache.append((batch_idx, (phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid)))
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        if hps.if_f0 == 1:
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
            
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
                
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # Discriminator Loss
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator Loss
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        # ระบบบังคับพิมพ์ Log อย่างน้อย 1 ครั้งต่อ Epoch ป้องกันอาการค้าง
        if rank == 0:
            if global_step % hps.train.log_interval == 0 or batch_idx == (len(train_loader) - 1):
                lr = optim_g.param_groups[0]["lr"]
                progress_pct = 100.0 * (batch_idx + 1) / len(train_loader)
                logger_ref.info(f"รอบฝึกสอน (Train Epoch): {epoch} [{progress_pct:.0f}%]")
                
                if loss_mel > 75: loss_mel = 75
                if loss_kl > 9: loss_kl = 9

                logger_ref.info(f"Losses => Disc: {loss_disc:.3f}, Gen: {loss_gen:.3f}, FM: {loss_fm:.3f}, Mel: {loss_mel:.3f}, KL: {loss_kl:.3f}")
                
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl,
                }
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

    # สรุปและบันทึกโมเดลหลังจบ Epoch
    if epoch % hps.save_every_epoch == 0 and rank == 0:
        save_idx = global_step if hps.if_latest == 0 else 2333333
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"G_{save_idx}.pth"))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"D_{save_idx}.pth"))
        
        if hps.save_every_weights == "1":
            target_net = net_g.module if hasattr(net_g, "module") else net_g
            ckpt = target_net.state_dict()
            logger_ref.info(
                f"[Save] กำลังบันทึกจุดตรวจสอบ (Saving ckpt) {hps.name}_e{epoch}:{savee(ckpt, hps.sample_rate, hps.if_f0, hps.name + f'_e{epoch}_s{global_step}', epoch, hps.version, hps)}"
            )

    if rank == 0:
        logger_ref.info(f"====> สรุปรอบฝึกสอน (Epoch Summary): {epoch} {epoch_recorder.record()}")
        
    if epoch >= hps.total_epoch and rank == 0:
        logger_ref.info("🎉 การฝึกสอนเสร็จสมบูรณ์ ระบบกำลังปิดตัวลง (Training is done. The program is closed.)")
        target_net = net_g.module if hasattr(net_g, "module") else net_g
        ckpt = target_net.state_dict()
        logger_ref.info(
            f"[Save] กำลังบันทึกจุดตรวจสอบสุดท้าย (Saving final ckpt):{savee(ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps)}"
        )
        sleep(1)
        os._exit(0) # แก้ไขจากการใช้ Code 2333333 เป็น Code 0 เพื่อให้โปรแกรมหลักสามารถอ่านสถานะความสำเร็จได้ถูกต้อง

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()