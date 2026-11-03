# ==============================================================================
# File Name    : utils.py
# Version      : v2.1.0-Release
# Target Path  : train/utils.py
# Function     : Unified Training Utilities, DataLoaders, Loss Functions & Samplers for RVC V2
# Credits      : Made And Checked By DELTA SYNTH & All Agentic AI
#                Original by RVC-Project / RVC Team
# ==============================================================================

import os
import sys
import math
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

# Ensure project root is present in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    Data loader supporting multi-speaker datasets, F0/pitch data,
    HuBERT features (256/768-dim), and GPU in-memory caching.
    """
    def __init__(self, training_files, hparams):
        self.audiopaths_and_text = self._load_filepaths_and_text(training_files)
        self.hparams = hparams
        self.max_wav_value = getattr(hparams, "max_wav_value", 32768.0)
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.unit_dim = 256 if getattr(hparams, "version", "v2") == "v1" else 768
        self.if_f0 = getattr(hparams, "if_f0", 1)
        self.if_cache_data_in_gpu = getattr(hparams, "if_cache_data_in_gpu", False)

        self._lengths = []
        for line in self.audiopaths_and_text:
            if len(line) >= 5:
                # Estimate length from phone feature or audio spec
                self._lengths.append(os.path.getsize(line[0]) // (2 * self.hop_length))
            else:
                self._lengths.append(1000)

    def _load_filepaths_and_text(self, filename, split="|"):
        with open(filename, encoding="utf-8") as f:
            lines = [line.strip().split(split) for line in f if line.strip()]
        return lines

    def get_audio_text_speaker_pair(self, audiopath_and_text):
        # Format: [gt_wav_path, phone_feat_path, f0_path, f0nsf_path, spk_id]
        wav_path = audiopath_and_text[0]
        phone_path = audiopath_and_text[1]
        f0_path = audiopath_and_text[2]
        f0nsf_path = audiopath_and_text[3]
        spk_id = int(audiopath_and_text[4])

        phone = np.load(phone_path)
        phone = np.repeat(phone, 2, axis=0)  # Align frame rates if necessary
        phone = torch.from_numpy(phone).float()

        if self.if_f0 == 1:
            pitch = np.load(f0_path)
            pitch = torch.from_numpy(pitch).long()
            pitchf = np.load(f0nsf_path)
            pitchf = torch.from_numpy(pitchf).float()
        else:
            pitch = torch.zeros(phone.size(0)).long()
            pitchf = torch.zeros(phone.size(0)).float()

        from scipy.io.wavfile import read
        sr, audio = read(wav_path)
        audio = torch.FloatTensor(audio.astype(np.float32))

        # Truncate to align length
        min_len = min(phone.size(0), pitch.size(0), audio.size(0) // self.hop_length)
        phone = phone[:min_len, :]
        pitch = pitch[:min_len]
        pitchf = pitchf[:min_len]
        audio = audio[: min_len * self.hop_length]

        spk_id = torch.LongTensor([spk_id])
        return (phone, pitch, pitchf, audio, spk_id)

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """
    Zero-pads audio waveforms, phoneme features, and F0 pitch curves
    to create clean and aligned training batches.
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # batch[i]: (phone, pitch, pitchf, audio, spk_id)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(0) for x in batch]), dim=0, descending=True
        )

        max_phone_len = max([x[0].size(0) for x in batch])
        max_audio_len = max([x[3].size(0) for x in batch])

        phone_dim = batch[0][0].size(1)
        batch_size = len(batch)

        phone_padded = torch.zeros(batch_size, max_phone_len, phone_dim)
        pitch_padded = torch.zeros(batch_size, max_phone_len, dtype=torch.long)
        pitchf_padded = torch.zeros(batch_size, max_phone_len, dtype=torch.float)
        audio_padded = torch.zeros(batch_size, 1, max_audio_len, dtype=torch.float)

        phone_lengths = torch.LongTensor(batch_size)
        audio_lengths = torch.LongTensor(batch_size)
        spk_ids = torch.LongTensor(batch_size)

        for i, idx in enumerate(ids_sorted_decreasing):
            phone, pitch, pitchf, audio, spk_id = batch[idx]

            phone_len = phone.size(0)
            audio_len = audio.size(0)

            phone_padded[i, :phone_len, :] = phone
            pitch_padded[i, :phone_len] = pitch
            pitchf_padded[i, :phone_len] = pitchf
            audio_padded[i, 0, :audio_len] = audio

            phone_lengths[i] = phone_len
            audio_lengths[i] = audio_len
            spk_ids[i] = spk_id[0]

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            audio_padded,
            audio_lengths,
            spk_ids,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintains balanced batches by binning similar audio sequence lengths together,
    improving training throughput and mitigating GPU memory spikes.
    """
    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset._lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets = self._create_buckets()
        self.total_size = sum(len(x) for x in self.buckets)

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for idx, length in enumerate(self.lengths):
            idx_bucket = self._find_bucket(length)
            buckets[idx_bucket].append(idx)
        return buckets

    def _find_bucket(self, length):
        for i, boundary in enumerate(self.boundaries):
            if length <= boundary:
                return i
        return len(self.boundaries)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        batches = []
        for bucket in self.buckets:
            if len(bucket) == 0:
                continue
            if self.shuffle:
                indices = torch.randperm(len(bucket), generator=g).tolist()
                bucket = [bucket[i] for i in indices]
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)

        if self.shuffle:
            indices = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in indices]

        batches = batches[self.rank :: self.num_replicas]
        return iter(batches)

    def __len__(self):
        return self.total_size // (self.batch_size * self.num_replicas)


# ----------------------------------------------------------------------
# Loss Computation & Mathematical Metrics
# ----------------------------------------------------------------------

def feature_loss(fmap_r, fmap_g):
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2.0


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0.0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        loss += l
        gen_losses.append(l.item())
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Computes Kullback-Leibler Divergence between predicted and posterior representations.
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss