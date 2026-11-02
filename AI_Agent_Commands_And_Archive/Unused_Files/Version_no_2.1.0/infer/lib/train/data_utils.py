# -*- coding: utf-8 -*-
# ==============================================================================
# File: data_utils.py
# Path: infer/lib/train/data_utils.py
# Description: RVC WebUI Core Module / โมดูลหลัก RVC WebUI สำหรับจัดการชุดข้อมูลการฝึกสอน
# Version: 2.2.0 (Min-Frame Padding Guard, Safe Bucket Clamping & Disk Verification)
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by "RVC-Project"
# ==============================================================================

import os
import traceback
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

logger = logging.getLogger(__name__)

try:
    from utils import PoliteLogger
except ImportError:
    class PoliteLogger:
        @staticmethod
        def warning(th_msg, en_msg="", exc=None):
            msg = f"🟡 [WARNING] {th_msg} " + (f"({en_msg})" if en_msg else "")
            if exc:
                msg += f" | {exc}"
            logger.warning(msg)

        @staticmethod
        def error(th_msg, en_msg="", exc=None):
            msg = f"🔴 [ERROR] {th_msg} " + (f"({en_msg})" if en_msg else "")
            if exc:
                msg += f" | {exc}"
            logger.error(msg)

from infer.lib.train.mel_processing import spectrogram_torch
from infer.lib.train.utils import load_filepaths_and_text, load_wav_to_torch

MIN_SEGMENT_FRAMES = 44  # Guarantees >= segment_size // hop_length for 32k (40), 40k (32), and 48k (36)


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """Dataset loader for pitch-guided (F0) RVC v1/v2 training."""

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
        for row in self.audiopaths_and_text:
            if len(row) < 5:
                continue
            audiopath, text, pitch, pitchf, dv = row[0], row[1], row[2], row[3], row[4]
            if self.min_text_len <= len(text) <= self.max_text_len:
                if os.path.exists(audiopath) and os.path.exists(text) and os.path.exists(pitch) and os.path.exists(pitchf):
                    audiopaths_and_text_new.append([audiopath, text, pitch, pitchf, dv])
                    est_len = max(MIN_SEGMENT_FRAMES, os.path.getsize(audiopath) // (3 * self.hop_length))
                    lengths.append(est_len)
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)])

    def get_audio_text_pair(self, audiopath_and_text):
        file, phone, pitch, pitchf, dv = audiopath_and_text[:5]
        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        cur_frames = spec.size(-1)
        if cur_frames < MIN_SEGMENT_FRAMES:
            pad_frames = MIN_SEGMENT_FRAMES - cur_frames
            pad_wav = MIN_SEGMENT_FRAMES * self.hop_length - wav.size(-1)
            spec = F.pad(spec, (0, pad_frames))
            if pad_wav > 0:
                wav = F.pad(wav, (0, pad_wav))
            phone = F.pad(phone, (0, 0, 0, pad_frames))
            pitch = F.pad(pitch, (0, pad_frames))
            pitchf = F.pad(pitchf, (0, pad_frames))

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(self, phone, pitch, pitchf):
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        pitch = np.load(pitch)
        pitchf = np.load(pitchf)
        n_num = min(phone.shape[0], 900)
        phone = phone[:n_num, :]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]
        phone = torch.FloatTensor(phone)
        pitch = torch.LongTensor(pitch)
        pitchf = torch.FloatTensor(pitchf)
        return phone, pitch, pitchf

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            import librosa
            audio_np = librosa.resample(audio.numpy(), orig_sr=sampling_rate, target_sr=self.sampling_rate)
            audio = torch.FloatTensor(audio_np)
        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, map_location="cpu", weights_only=False)
                if spec.size(0) != (self.filter_length // 2 + 1):
                    raise ValueError("Spectrogram channel mismatch")
            except Exception:
                spec = spectrogram_torch(
                    audio_norm, self.filter_length, self.sampling_rate,
                    self.hop_length, self.win_length, center=False
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm, self.filter_length, self.sampling_rate,
                self.hop_length, self.win_length, center=False
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    """Zero-pads model inputs and targets for pitch-guided (F0) batches."""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )
        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1])
        pitch_padded = torch.LongTensor(len(batch), max_phone_len)
        pitchf_padded = torch.FloatTensor(len(batch), max_phone_len)
        phone_padded.zero_()
        pitch_padded.zero_()
        pitchf_padded.zero_()
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

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class TextAudioLoader(torch.utils.data.Dataset):
    """Dataset loader for pitchless (no-F0) RVC v1/v2 training."""

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
        for row in self.audiopaths_and_text:
            if len(row) < 2:
                continue
            audiopath = row[0]
            text = row[1]
            dv = row[2] if len(row) > 2 else 0
            if self.min_text_len <= len(text) <= self.max_text_len:
                if os.path.exists(audiopath) and os.path.exists(text):
                    audiopaths_and_text_new.append([audiopath, text, dv])
                    est_len = max(MIN_SEGMENT_FRAMES, os.path.getsize(audiopath) // (3 * self.hop_length))
                    lengths.append(est_len)
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)])

    def get_audio_text_pair(self, audiopath_and_text):
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        dv = audiopath_and_text[2] if len(audiopath_and_text) > 2 else 0
        phone = self.get_labels(phone)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]

        cur_frames = spec.size(-1)
        if cur_frames < MIN_SEGMENT_FRAMES:
            pad_frames = MIN_SEGMENT_FRAMES - cur_frames
            pad_wav = MIN_SEGMENT_FRAMES * self.hop_length - wav.size(-1)
            spec = F.pad(spec, (0, pad_frames))
            if pad_wav > 0:
                wav = F.pad(wav, (0, pad_wav))
            phone = F.pad(phone, (0, 0, 0, pad_frames))

        return (spec, wav, phone, dv)

    def get_labels(self, phone):
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        n_num = min(phone.shape[0], 900)
        phone = phone[:n_num, :]
        phone = torch.FloatTensor(phone)
        return phone

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            import librosa
            audio_np = librosa.resample(audio.numpy(), orig_sr=sampling_rate, target_sr=self.sampling_rate)
            audio = torch.FloatTensor(audio_np)
        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, map_location="cpu", weights_only=False)
                if spec.size(0) != (self.filter_length // 2 + 1):
                    raise ValueError("Spectrogram channel mismatch")
            except Exception:
                spec = spectrogram_torch(
                    audio_norm, self.filter_length, self.sampling_rate,
                    self.hop_length, self.win_length, center=False
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        else:
            spec = spectrogram_torch(
                audio_norm, self.filter_length, self.sampling_rate,
                self.hop_length, self.win_length, center=False
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets for pitchless (no-F0) batches."""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )
        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        spec_padded.zero_()
        wave_padded.zero_()
        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(len(batch), max_phone_len, batch[0][2].shape[1])
        phone_padded.zero_()
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
            sid[i] = row[3] if len(row) > 3 else 0

        return (
            phone_padded,
            phone_lengths,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintains similar input lengths in a batch while clamping out-of-range
    lengths into the nearest valid bucket so short datasets never yield 0 batches.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = max(1, int(batch_size))
        self.boundaries = list(boundaries)
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            # Clamp length into [boundaries[0] + 1, boundaries[-1]] so valid segments are never dropped
            clamped_len = max(self.boundaries[0] + 1, min(length, self.boundaries[-1]))
            idx_bucket = self._bisect(clamped_len)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
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
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]
                ]
                batches.append(batch)

        if self.shuffle and len(batches) > 0:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
