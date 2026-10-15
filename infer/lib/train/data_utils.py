# -*- coding: utf-8 -*-
# ==============================================================================
# วัน/เดือน/ปี: 12/10/2569 | 12/10/2026
# ชื่อไฟล์: data_utils.py
# ลำดับเวอร์ชัน: 1.3
# Made And Checked By DELTA SYNTH [ ปฎิภัทร วงศ์ใหญ่ ] & All Code Agentic AI Engine
# Original by "RVC-Project"
# ตำแหน่งที่วางไฟล์: infer/lib/train/data_utils.py
# หน้าที่: จัดการและโหลดชุดข้อมูลเสียงสำหรับการฝึกสอน (Data Utils & Loader)
# รายละเอียดการแก้ไขงาน: ซ่อมแซมโค้ดที่ตกหล่นในคลาส TextAudioLoader และ Collate ให้ทำงานได้สมบูรณ์
# ==============================================================================

import os
import traceback
import logging
import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed

logger = logging.getLogger(__name__)

# ==========================================
# DELTA SYNTH PoliteLogger Integration
# ==========================================
try:
    from utils import PoliteLogger
except ImportError:
    class PoliteLogger:
        @staticmethod
        def warning(th_msg, en_msg="", exc=None):
            msg = f"🟡 [WARNING] {th_msg} " + (f"({en_msg})" if en_msg else "")
            if exc: msg += f" | {exc}"
            logger.warning(msg)
            
        @staticmethod
        def error(th_msg, en_msg="", exc=None):
            msg = f"🔴 [ERROR] {th_msg} " + (f"({en_msg})" if en_msg else "")
            if exc: msg += f" | {exc}"
            logger.error(msg)

from infer.lib.train.mel_processing import spectrogram_torch
from infer.lib.train.utils import load_filepaths_and_text, load_wav_to_torch

class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """โหลดข้อมูลสำหรับโมเดลแบบมี Pitch (F0)"""
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, pitch, pitchf, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) <= self.max_text_len:
                if os.path.exists(audiopath):
                    audiopaths_and_text_new.append([audiopath, text, pitch, pitchf, dv])
                    lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)])

    def get_audio_text_pair(self, audiopath_and_text):
        file, phone, pitch, pitchf, dv = audiopath_and_text
        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size(0)
        len_spec = spec.size(-1)
        
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(self, phone, pitch, pitchf):
        phone = np.repeat(np.load(phone), 2, axis=0)
        pitch = np.load(pitch)
        pitchf = np.load(pitchf)
        n_num = min(phone.shape[0], 900)
        
        return (
            torch.FloatTensor(phone[:n_num, :]),
            torch.LongTensor(pitch[:n_num]),
            torch.FloatTensor(pitchf[:n_num])
        )

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Sample rate mismatch: {sampling_rate} != {self.sampling_rate}")
        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, weights_only=True)
            except Exception as e:
                PoliteLogger.warning(f"Failed to load spec file, regenerating: {spec_filename}", exc=str(e))
                spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False)
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
            
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextAudioCollateMultiNSFsid:
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)
        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len).zero_()
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len).zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1]).zero_()
        pitch_padded = torch.LongTensor(len(batch), max_phone_len).zero_()
        pitchf_padded = torch.FloatTensor(len(batch), max_phone_len).zero_()
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)
            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)
            pitch = row[3]
            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf = row[4]
            pitchf_padded[i, : pitchf.size(0)] = pitchf
            sid[i] = row[5]

        return phone_padded, phone_lengths, pitch_padded, pitchf_padded, spec_padded, spec_lengths, wave_padded, wave_lengths, sid

class TextAudioLoader(torch.utils.data.Dataset):
    """โหลดข้อมูลสำหรับโมเดลแบบไม่มี Pitch (No F0)"""
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) <= self.max_text_len:
                if os.path.exists(audiopath):
                    audiopaths_and_text_new.append([audiopath, text, dv])
                    lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)])

    def get_audio_text_pair(self, audiopath_and_text):
        file, phone, dv = audiopath_and_text
        phone = self.get_labels(phone)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size(0)
        len_spec = spec.size(-1)
        
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]

        return (spec, wav, phone, dv)

    def get_labels(self, phone):
        phone = np.repeat(np.load(phone), 2, axis=0)
        n_num = min(phone.shape[0], 900)
        return torch.FloatTensor(phone[:n_num, :])

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Sample rate mismatch: {sampling_rate} != {self.sampling_rate}")
        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, weights_only=True)
            except Exception as e:
                PoliteLogger.warning(f"Failed to load spec file, regenerating: {spec_filename}", exc=str(e))
                spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False)
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
            
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextAudioCollate:
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True)
        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len).zero_()
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len).zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1]).zero_()
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)
            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)
            sid[i] = row[3]

        return phone_padded, phone_lengths, spec_padded, spec_lengths, wave_padded, wave_lengths, sid

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum([t - t % self.batch_size for t in self.num_samples_per_bucket])
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i - 1)
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            t = (len_bucket + self.num_replicas - 1) // self.num_replicas * self.num_replicas
            num_samples_per_bucket.append(t)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
            ids_bucket = ids_bucket[self.rank::self.num_replicas]
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None: hi = len(self.boundaries)
        if hi > 0 and x > self.boundaries[hi - 1]: return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if x < self.boundaries[mid]: hi = mid
            else: lo = mid + 1
        return lo

    def __len__(self):
        return self.num_samples // self.batch_size