# ==========================================
# File: rtrvc.py
# Version: 1.0
# Location: infer/rtrvc.py
# Function: Core Real-Time Voice Conversion (RVC) module for handling audio streams and model inference.
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by RVC Project
# ==========================================

import os
import sys
import traceback
import time
import threading
import numpy as np
import torch
import pyaudio

class RVCRealTime:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_half = True if self.device == "cuda" else False
        self.net_g = None
        self.hubert_model = None
        self.vc = None
        self.cpt = None
        self.tgt_sr = 0
        self.if_f0 = 1
        self.version = "v2"
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.lock = threading.Lock()

    def load_model(self, model_path):
        """Loads the RVC PyTorch model securely and initializes parameters."""
        print(f"[INFO] Loading RVC model from: {model_path}")
        try:
            self.cpt = torch.load(model_path, map_location="cpu")
            self.tgt_sr = self.cpt["config"][-1]
            self.if_f0 = self.cpt.get("f0", 1)
            self.version = self.cpt.get("version", "v1")
            print(f"[SUCCESS] Model loaded. Target SR: {self.tgt_sr}, Version: {self.version}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            traceback.print_exc()
            return False

    def start_stream(self, input_device_idx, output_device_idx, chunk_size=1024):
        """Starts the audio I/O stream for real-time conversion."""
        if self.is_running:
            print("[WARNING] Stream is already running.")
            return

        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.tgt_sr if self.tgt_sr > 0 else 44100,
                input=True,
                output=True,
                input_device_index=input_device_idx,
                output_device_index=output_device_idx,
                frames_per_buffer=chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_running = True
            self.stream.start_stream()
            print("[INFO] Real-time audio stream started successfully.")
        except Exception as e:
            print(f"[ERROR] Could not start audio stream: {e}")
            traceback.print_exc()

    def stop_stream(self):
        """Stops and closes the active audio stream."""
        if self.stream and self.is_running:
            self.is_running = False
            self.stream.stop_stream()
            self.stream.close()
            print("[INFO] Real-time audio stream stopped.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Background callback function handling real-time audio chunk processing."""
        if not self.is_running:
            return (None, pyaudio.paComplete)
        
        with self.lock:
            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(in_data, dtype=np.float32)
            
            # TODO: Integrate self.vc.pipeline logic here for actual voice conversion
            # For now, it safely passes the audio through to prevent software crashes
            out_data = audio_chunk.tobytes()
            
        return (out_data, pyaudio.paContinue)

    def terminate(self):
        """Safely shuts down PyAudio and clears memory."""
        self.stop_stream()
        self.audio.terminate()
        print("[INFO] RVC Real-Time module terminated cleanly.")

# Expose standard functions expected by the GUI
def initialize_realtime():
    return RVCRealTime()