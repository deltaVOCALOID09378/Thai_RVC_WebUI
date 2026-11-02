"""
╔══════════════════════════════════════════════════════════════╗
║          The Diff Singer Trainer UI  v1.2                    ║
║   Comprehensive Gradio-based DiffSinger Voice Bank Trainer   ║
║   Supports CPU (Intel/AMD) and NVIDIA GPU                    ║
║   Made And Checked By DELTA SYNTH & All Code Agentic AI      ║
║   Original by Patiphat Wongyai (Delta) & OpenVPI DiffSinger  ║
╚══════════════════════════════════════════════════════════════╝
"""
import gradio as gr
import os
import sys
import csv
import json
import yaml
import shutil
import signal
import platform
import subprocess
import threading
import pathlib
import time
import wave
from datetime import datetime
from collections import OrderedDict
import gc

# [Rule 2.3] บังคับใช้ UTF-8 สำหรับ Terminal Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Path Setup ───────────────────────────────────────────────
ROOT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DICTIONARIES_DIR = ROOT_DIR / "dictionaries"
CONFIGS_DIR = ROOT_DIR / "configs"
SCRIPTS_DIR = ROOT_DIR / "scripts"

os.environ['PYTHONPATH'] = str(ROOT_DIR)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ─── Global State ─────────────────────────────────────────────
_active_process = None
_process_lock = threading.Lock()
CANCEL_REQUESTED = False

# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS — Premium Dark Theme
# ═══════════════════════════════════════════════════════════════
CUSTOM_CSS = """
/* ── Global ── */
body, .gradio-container {
    max-width: 1400px !important;
    font-family: 'Leelawadee UI', 'Segoe UI', Tahoma, sans-serif !important;
    background-color: #1A1A1A !important;
    color: #F0F0F0 !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1A1A1A, #242424);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    border: 1px solid #333333;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.app-header h1 {
    color: #CC2200 !important;
    font-size: 2.2em;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #F0F0F0;
    font-size: 1.05em;
    margin: 0;
}

/* ── Tab Styling ── */
.tab-nav button {
    font-size: 0.95em !important;
    font-weight: 600 !important;
    padding: 12px 18px !important;
    border-radius: 10px 10px 0 0 !important;
    transition: all 0.3s ease !important;
}
.tab-nav button.selected {
    background-color: #CC2200 !important;
    color: white !important;
    border: none !important;
}

/* ── Status Badges ── */
.status-ok { color: #4ade80; font-weight: 600; }
.status-warn { color: #fbbf24; font-weight: 600; }
.status-err { color: #f87171; font-weight: 600; }

/* ── Cards ── */
.gr-box, div.gradio-container > div, .info-card {
    background-color: #242424 !important;
    border: 1px solid #333333 !important;
    border-radius: 12px;
}

/* ── Log Output ── */
.log-output textarea {
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace !important;
    font-size: 0.85em !important;
    line-height: 1.6 !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.primary-btn, .gr-button-primary {
    background-color: #CC2200 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
}
.primary-btn:hover, .gr-button-primary:hover {
    background-color: #FF4422 !important;
}
.primary-btn:active, .gr-button-primary:active {
    background-color: #991100 !important;
}
.danger-btn { background-color: #dc2626 !important; border: none !important; font-weight: 700 !important; color: white !important; }
.success-btn { background-color: #16a34a !important; border: none !important; font-weight: 700 !important; color: white !important; }

/* ── Section Headers ── */
.section-title {
    font-size: 1.1em;
    font-weight: 700;
    color: #F0F0F0;
    margin: 16px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #333333;
}
"""

# ═══════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def log_msg(msg, level="INFO"):
    icons = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "RUN": "🔄"}
    icon = icons.get(level, "📌")
    return f"[{get_timestamp()}] {icon} {msg}"

def safe_yaml_dump(data):
    return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)

def get_python_executable():
    """Get the Python executable, preferring the venv."""
    venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists(): return str(venv_python)
    venv_python_unix = ROOT_DIR / ".venv" / "bin" / "python"
    if venv_python_unix.exists(): return str(venv_python_unix)
    return sys.executable

# [Rule 8] VRAM Safety Protocol
def clear_vram_safe():
    try:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with contextlib.suppress(Exception): torch.cuda.ipc_collect()
    except Exception:
        pass

# [Rule 6] Windows Junk Sentinel
def purge_directory_junk(dir_path: str):
    if not os.path.exists(dir_path): return
    junk_files = {'desktop.ini', 'thumbs.db'}
    for root_d, _, files in os.walk(dir_path):
        for f in files:
            full_path = os.path.join(root_d, f)
            if f.lower() in junk_files or os.path.getsize(full_path) == 0:
                with contextlib.suppress(Exception):
                    os.chmod(full_path, 0o777)
                    os.remove(full_path)

def run_subprocess_streaming(cmd, cwd=None):
    """Run a subprocess and yield output lines in real-time."""
    global _active_process, CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT_DIR)
    env['PYTHONUNBUFFERED'] = '1'

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=cwd or str(ROOT_DIR), env=env, text=True, bufsize=1,
            encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        with _process_lock: _active_process = process

        output = ""
        for line in iter(process.stdout.readline, ''):
            if CANCEL_REQUESTED:
                stop_active_process()
                yield output + f"\n{log_msg('Process cancelled by user.', 'WARN')}\n"
                return
            if line:
                output += line
                yield output

        process.wait()
        ret = process.returncode
        if ret == 0: output += f"\n{log_msg('Process completed successfully!', 'OK')}\n"
        else: output += f"\n{log_msg(f'Process exited with code {ret}', 'ERROR')}\n"
        yield output
    except Exception as e:
        yield f"{log_msg(f'Error: {str(e)}', 'ERROR')}\n"
    finally:
        with _process_lock: _active_process = None
        clear_vram_safe()

# [Rule 9] Subprocess Tree Termination
def stop_active_process():
    global _active_process, CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    with _process_lock:
        if _active_process and _active_process.poll() is None:
            try:
                if os.name == 'nt':
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(_active_process.pid)], capture_output=True, timeout=5)
                else:
                    _active_process.terminate()
                _active_process.wait(timeout=5)
                return log_msg("Process stopped successfully.", "OK")
            except subprocess.TimeoutExpired:
                _active_process.kill()
                return log_msg("Process force-killed.", "WARN")
            except Exception as e:
                return log_msg(f"Error stopping process: {e}", "ERROR")
            finally:
                clear_vram_safe()
    return log_msg("No active process to stop.", "WARN")

def get_audio_info(filepath):
    try:
        ext = pathlib.Path(filepath).suffix.lower()
        if ext == '.wav':
            with wave.open(str(filepath), 'r') as wf:
                sr = wf.getframerate(); ch = wf.getnchannels(); frames = wf.getnframes(); dur = frames / sr
                return {"sample_rate": sr, "channels": ch, "duration": dur, "frames": frames}
        else:
            try:
                import librosa
                y, sr = librosa.load(str(filepath), sr=None, mono=False)
                ch = 1 if y.ndim == 1 else y.shape[0]
                dur = (len(y) if y.ndim == 1 else y.shape[1]) / sr
                return {"sample_rate": sr, "channels": ch, "duration": dur}
            except ImportError: return None
    except Exception: return None

# [Rule 10] Tkinter Topmost Isolation
def browse_folder(current_val=""):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        clean_p = str(current_val or "").strip().strip('"')
        init_dir = clean_p if (clean_p and os.path.isdir(clean_p)) else str(ROOT_DIR)
        selected = filedialog.askdirectory(initialdir=init_dir, title="เลือกโฟลเดอร์ (Select Folder)")
        root.destroy()
        if selected: return os.path.normpath(selected).replace("\\", "/")
    except Exception as e:
        print(f"Browse folder error: {e}")
    return current_val

def browse_file(current_val="", filetypes=None):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        clean_p = str(current_val or "").strip().strip('"')
        init_dir = os.path.dirname(clean_p) if (clean_p and os.path.isfile(clean_p)) else str(ROOT_DIR)
        selected = filedialog.askopenfilename(initialdir=init_dir, title="เลือกไฟล์ (Select File)", filetypes=filetypes or [("All files", "*.*")])
        root.destroy()
        if selected: return os.path.normpath(selected).replace("\\", "/")
    except Exception as e:
        print(f"Browse file error: {e}")
    return current_val

# ═══════════════════════════════════════════════════════════════
#  TAB 1: SYSTEM INFO & SETUP
# ═══════════════════════════════════════════════════════════════

def detect_system_info():
    info_lines = ["═" * 60, "   🖥️  SYSTEM INFORMATION  (ข้อมูลระบบ)", "═" * 60]
    info_lines.append(f"\n📌 OS: {platform.system()} {platform.release()} ({platform.machine()})")
    cpu_name = platform.processor() or "Unknown"
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        cpu_phys = psutil.cpu_count(logical=False)
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        info_lines.append(f"📌 CPU: {cpu_name}\n   Cores: {cpu_phys} physical / {cpu_count} logical\n📌 RAM: {ram_gb:.1f} GB total / {ram.available / (1024**3):.1f} GB available")
    except ImportError: info_lines.append(f"📌 CPU: {cpu_name}\n   (Install psutil for detailed info)")
    info_lines.append(f"\n📌 Python: {sys.version.split()[0]}\n   Executable: {get_python_executable()}")
    try:
        import torch
        info_lines.append(f"📌 PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info_lines.append(f"✅ CUDA: {torch.version.cuda}\n✅ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
            if torch.cuda.device_count() > 1: info_lines.append(f"   GPUs available: {torch.cuda.device_count()}")
        else: info_lines.append("⚠️ CUDA: Not available — จะใช้ CPU mode")
    except ImportError: info_lines.append("❌ PyTorch: Not installed!")

    info_lines.extend(["\n" + "─" * 40, "📦 Dependencies Status:"])
    deps = {'lightning': 'PyTorch Lightning', 'librosa': 'Librosa', 'numpy': 'NumPy', 'yaml': 'PyYAML', 'h5py': 'h5py', 'tensorboard': 'TensorBoard', 'onnx': 'ONNX'}
    for module, name in deps.items():
        try: __import__(module); info_lines.append(f"   ✅ {name}")
        except ImportError: info_lines.append(f"   ❌ {name} — not installed")

    info_lines.extend(["\n" + "═" * 60, "   ⚙️  AUTO-CONFIGURED SETTINGS  (ตั้งค่าอัตโนมัติ)", "═" * 60])
    for key, val in auto_detect_training_settings().items(): info_lines.append(f"   {key}: {val}")

    info_lines.extend(["\n" + "─" * 40, "📁 Pretrained Models:"])
    rmvpe_path = CHECKPOINTS_DIR / "rmvpe" / "model.pt"
    vocoder_found = any(d.is_dir() and 'nsf_hifigan' in d.name.lower() and (d / "model.ckpt").exists() for d in (CHECKPOINTS_DIR.iterdir() if CHECKPOINTS_DIR.exists() else []))
    info_lines.append("   ✅ Vocoder: Found" if vocoder_found else "   ⚠️ Vocoder: Not found — กรุณาดาวน์โหลด NSF-HiFiGAN checkpoint")
    info_lines.append("   ✅ RMVPE Pitch Extractor: Found" if rmvpe_path.exists() else "   ⚠️ RMVPE: Not found at checkpoints/rmvpe/model.pt")
    return "\n".join(info_lines)

def auto_detect_training_settings():
    settings = OrderedDict()
    try: import torch; has_cuda = torch.cuda.is_available()
    except ImportError: has_cuda = False

    if has_cuda:
        try: gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception: gpu_mem = 4.0
        settings["Accelerator"] = "auto (CUDA GPU)"
        settings["Precision"] = "16-mixed"
        settings["Strategy Backend"] = "nccl"
        if gpu_mem >= 12: settings["Max Batch Frames"] = "50000"; settings["Max Batch Size"] = "64"
        elif gpu_mem >= 8: settings["Max Batch Frames"] = "40000"; settings["Max Batch Size"] = "48"
        elif gpu_mem >= 6: settings["Max Batch Frames"] = "30000"; settings["Max Batch Size"] = "32"
        else: settings["Max Batch Frames"] = "20000"; settings["Max Batch Size"] = "16"
    else:
        settings["Accelerator"] = "cpu"; settings["Precision"] = "32 (CPU)"; settings["Strategy Backend"] = "gloo"
        settings["Max Batch Frames"] = "16000"; settings["Max Batch Size"] = "8"

    try: import psutil; settings["Num Workers"] = str(min(psutil.cpu_count(logical=False) or 2, 4))
    except ImportError: settings["Num Workers"] = "2"
    settings["Pitch Extractor"] = "rmvpe (default)"; settings["Sample Rate"] = "44100 Hz"; settings["Mel Bins"] = "128"; settings["Hop Size"] = "512"
    return settings

# ═══════════════════════════════════════════════════════════════
#  TAB 2: DATA MANAGER
# ═══════════════════════════════════════════════════════════════

def list_existing_projects():
    return [d.name for d in sorted(DATA_DIR.iterdir()) if d.is_dir() and (d / "raw").exists()] if DATA_DIR.exists() else []

def create_project(project_name, speaker_name, speaker_id, language):
    if not project_name or not project_name.strip(): return log_msg("กรุณาระบุชื่อโปรเจกต์", "ERROR"), gr.update()
    project_name = project_name.strip().replace(" ", "_"); speaker_name = speaker_name.strip() if speaker_name else project_name
    try: speaker_id = int(speaker_id)
    except (ValueError, TypeError): speaker_id = 0 # [Rule 1]
    
    project_dir = DATA_DIR / project_name
    if project_dir.exists(): return log_msg(f"โปรเจกต์ '{project_name}' มีอยู่แล้ว!", "WARN"), gr.update()
    
    try:
        (project_dir / "raw" / "wavs").mkdir(parents=True, exist_ok=True)
        (project_dir / "binary").mkdir(parents=True, exist_ok=True)
        (project_dir / "export").mkdir(parents=True, exist_ok=True)
        with open(project_dir / "raw" / "transcriptions.csv", 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['name', 'ph_seq', 'ph_dur'])
        meta = {"project_name": project_name, "speaker_name": speaker_name, "speaker_id": speaker_id, "language": language or "Custom-Language", "created_at": datetime.now().isoformat(), "sample_rate": 44100}
        with open(project_dir / "project_meta.json", 'w', encoding='utf-8') as f: json.dump(meta, f, ensure_ascii=False, indent=2)
        return log_msg(f"สร้างโปรเจกต์ '{project_name}' สำเร็จ!\n   Speaker: {speaker_name} (ID: {speaker_id})\n   Language: {meta['language']}", "OK"), gr.update(choices=list_existing_projects(), value=project_name)
    except Exception as e: return log_msg(f"ไม่สามารถสร้างโปรเจกต์: {e}", "ERROR"), gr.update()

def upload_audio_files(project_name, files):
    if not project_name: return log_msg("กรุณาเลือกโปรเจกต์ก่อน", "ERROR"), ""
    wavs_dir = DATA_DIR / project_name / "raw" / "wavs"
    if not wavs_dir.exists(): return log_msg(f"ไม่พบโฟลเดอร์โปรเจกต์ '{project_name}'", "ERROR"), ""
    if not files: return log_msg("ไม่มีไฟล์ที่เลือก", "WARN"), ""

    log = [log_msg(f"เริ่มอัปโหลด {len(files)} ไฟล์...", "RUN")]
    success_count = skip_count = error_count = 0
    for file_path in files:
        fname = pathlib.Path(file_path).name; ext = pathlib.Path(fname).suffix.lower()
        if ext not in ['.wav', '.flac']: log.append(log_msg(f"ข้าม '{fname}' — ไม่รองรับ format", "WARN")); skip_count += 1; continue
        dest = wavs_dir / fname
        if dest.exists(): log.append(log_msg(f"ข้าม '{fname}' — มีอยู่แล้ว", "WARN")); skip_count += 1; continue
        try:
            shutil.copy2(file_path, dest)
            info = get_audio_info(dest)
            if info and info["sample_rate"] != 44100: log.append(log_msg(f"'{fname}' — SR={info['sample_rate']}Hz (แนะนำ 44100Hz)", "WARN"))
            success_count += 1; log.append(log_msg(f"อัปโหลด '{fname}' สำเร็จ", "OK"))
        except Exception as e: error_count += 1; log.append(log_msg(f"ผิดพลาด '{fname}': {e}", "ERROR"))
    log.append(f"\n{log_msg(f'สรุป: สำเร็จ {success_count}, ข้าม {skip_count}, ผิดพลาด {error_count}', 'OK' if error_count == 0 else 'WARN')}")
    return "\n".join(log), get_project_stats(project_name)

def get_project_stats(project_name):
    if not project_name: return "ยังไม่ได้เลือกโปรเจกต์"
    wavs_dir = DATA_DIR / project_name / "raw" / "wavs"
    if not wavs_dir.exists(): return "ไม่พบโฟลเดอร์โปรเจกต์"
    audio_files = list(wavs_dir.glob("*.wav")) + list(wavs_dir.glob("*.flac"))
    total_duration = sum((get_audio_info(af) or {}).get("duration", 0) for af in audio_files)
    meta = json.load(open(DATA_DIR / project_name / "project_meta.json", 'r', encoding='utf-8')) if (DATA_DIR / project_name / "project_meta.json").exists() else {}
    return f"╔══════════════════════════════════════╗\n║   📊 สถิติโปรเจกต์ (Project Stats)    ║\n╠══════════════════════════════════════╣\n║  📁 Project: {project_name:<22} ║\n║  🎤 Speaker: {meta.get('speaker_name', 'N/A'):<22} ║\n║  🆔 Speaker ID: {str(meta.get('speaker_id', 0)):<19} ║\n║  🌐 Language: {meta.get('language', 'N/A'):<22} ║\n║  🎵 Audio Files: {str(len(audio_files)):<19} ║\n║  ⏱️ Total Duration: {total_duration/60:.1f} min ({total_duration:.0f}s) ║\n╚══════════════════════════════════════╝"

def remove_duplicates(project_name):
    if not project_name: return log_msg("กรุณาเลือกโปรเจกต์ก่อน", "ERROR")
    raw_dir = DATA_DIR / project_name / "raw"
    trans_path = raw_dir / "transcriptions.csv"
    if not trans_path.exists(): return log_msg("ไม่พบ transcriptions.csv", "WARN")

    log = [log_msg("เริ่มตรวจหาไฟล์เสียงที่ซ้ำกัน...", "RUN")]
    entries = []
    with open(trans_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f): entries.append(row)
    if not entries: return log_msg("ไม่มีข้อมูล transcription", "WARN")

    seen, duplicates = {}, []
    for entry in entries:
        key = entry.get('ph_seq', '').strip()
        if key in seen: duplicates.append(entry); log.append(log_msg(f"พบซ้ำ: '{entry['name']}'", "WARN"))
        else: seen[key] = entry
    if not duplicates: log.append(log_msg("ไม่พบไฟล์เสียงที่ซ้ำกัน ✨", "OK")); return "\n".join(log)

    dup_names = {d['name'] for d in duplicates}
    kept = [e for e in entries if e['name'] not in dup_names]
    with open(trans_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur']); writer.writeheader(); writer.writerows(kept)

    removed = 0
    for dn in dup_names:
        for ext in ['.wav', '.flac']:
            dup = raw_dir / "wavs" / f"{dn}{ext}"
            if dup.exists(): dup.unlink(); removed += 1; log.append(log_msg(f"ลบไฟล์: {dn}{ext}", "OK"))
    log.append(f"\n{log_msg(f'สรุป: ลบไฟล์ซ้ำ {removed} ไฟล์, เหลือ {len(kept)} รายการ', 'OK')}")
    return "\n".join(log)

def scan_phonemes_from_data(project_name):
    if not project_name: return log_msg("กรุณาเลือกโปรเจกต์ก่อน", "ERROR"), ""
    trans_path = DATA_DIR / project_name / "raw" / "transcriptions.csv"
    if not trans_path.exists(): return log_msg("ไม่พบ transcriptions.csv", "WARN"), ""
    phonemes = {'AP', 'SP'}
    with open(trans_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            for ph in row.get('ph_seq', '').strip().split(): phonemes.add(ph.split('/', 1)[1] if '/' in ph else ph)
    return log_msg(f"พบ {len(phonemes)} หน่วยเสียงในชุดข้อมูล", "OK"), "\n".join(f"{ph}\t{ph}" for ph in sorted(phonemes))

def save_custom_dictionary(project_name, dict_content):
    if not project_name or not dict_content: return log_msg("ข้อมูลไม่ครบ", "ERROR")
    try:
        with open(DATA_DIR / project_name / "custom_dictionary.txt", 'w', encoding='utf-8') as f: f.write(dict_content)
        return log_msg("บันทึก dictionary สำเร็จ", "OK")
    except Exception as e: return log_msg(f"ผิดพลาด: {e}", "ERROR")

def load_transcriptions(project_name): return open(DATA_DIR / project_name / "raw" / "transcriptions.csv", 'r', encoding='utf-8').read() if project_name and (DATA_DIR / project_name / "raw" / "transcriptions.csv").exists() else "name,ph_seq,ph_dur\n"
def save_transcriptions(project_name, csv_text):
    if not project_name: return log_msg("กรุณาเลือกโปรเจกต์ก่อน", "ERROR")
    try:
        with open(DATA_DIR / project_name / "raw" / "transcriptions.csv", 'w', encoding='utf-8', newline='') as f: f.write(csv_text)
        return log_msg(f"บันทึก transcription สำเร็จ ({max(0, len([l for l in csv_text.strip().split(chr(10)) if l.strip()]) - 1)} รายการ)", "OK")
    except Exception as e: return log_msg(f"ผิดพลาด: {e}", "ERROR")

# ═══════════════════════════════════════════════════════════════
#  TAB 3: CONFIG GENERATOR
# ═══════════════════════════════════════════════════════════════

def get_vocoder_ckpt_path():
    for d in (CHECKPOINTS_DIR.iterdir() if CHECKPOINTS_DIR.exists() else []):
        if d.is_dir() and 'nsf_hifigan' in d.name.lower() and (d / "model.ckpt").exists(): return f"checkpoints/{d.name}/model.ckpt"
    return "checkpoints/nsf_hifigan/model.ckpt"

def generate_config(project_name, model_type, max_updates, finetune_enabled, finetune_ckpt_path):
    if not project_name: return log_msg("กรุณาเลือกโปรเจกต์ก่อน", "ERROR"), ""
    meta_path = DATA_DIR / project_name / "project_meta.json"
    if not meta_path.exists(): return log_msg("ไม่พบข้อมูลโปรเจกต์", "ERROR"), ""
    meta = json.load(open(meta_path, 'r', encoding='utf-8'))
    
    settings = auto_detect_training_settings()
    has_cuda = "CUDA" in settings.get("Accelerator", "")
    max_updates = int(max_updates) if max_updates else 160000
    
    lang = meta.get("language", "Custom-Language")
    dict_path = "dictionaries/opencpop-extension.txt" if lang == "zh" else f"data/{project_name}/custom_dictionary.txt"
    wavs_dir = DATA_DIR / project_name / "raw" / "wavs"
    audio_files = sorted([f.stem for f in wavs_dir.glob("*.wav")] + [f.stem for f in wavs_dir.glob("*.flac")])
    test_prefixes = audio_files[:max(1, min(5, len(audio_files) // 10))] if audio_files else ["test"]

    config = OrderedDict()
    config['base_config'] = [f"configs/{model_type.lower()}.yaml"]
    config['dictionaries'] = {lang: dict_path}
    config['extra_phonemes'] = []
    config['merged_phoneme_groups'] = []
    config['datasets'] = [{'raw_data_dir': f'data/{project_name}/raw', 'speaker': meta.get("speaker_name", project_name), 'spk_id': meta.get("speaker_id", 0), 'language': lang, 'test_prefixes': test_prefixes}]
    config['binary_data_dir'] = f'data/{project_name}/binary'
    config['binarization_args'] = {'num_workers': int(settings.get("Num Workers", "2")), 'shuffle': True}
    config['pe'] = 'rmvpe'
    config['pe_ckpt'] = 'checkpoints/rmvpe/model.pt'
    config['hnsep'] = 'vr'
    config['hnsep_ckpt'] = 'checkpoints/vr/model.pt'
    
    if model_type == "Acoustic":
        config['vocoder'] = 'NsfHifiGAN'
        config['vocoder_ckpt'] = get_vocoder_ckpt_path()
        config['use_key_shift_embed'] = True; config['use_speed_embed'] = True
        config['augmentation_args'] = {'random_pitch_shifting': {'enabled': True, 'range': [-5.0, 5.0], 'scale': 0.75}, 'random_time_stretching': {'enabled': True, 'range': [0.5, 2.0], 'scale': 0.75}}
        config['diffusion_type'] = 'reflow'
        config['use_shallow_diffusion'] = True
        config['shallow_diffusion_args'] = {'train_aux_decoder': True, 'train_diffusion': True, 'val_gt_start': False, 'aux_decoder_arch': 'convnext', 'aux_decoder_args': {'num_channels': 512, 'num_layers': 6, 'kernel_size': 7, 'dropout_rate': 0.1}, 'aux_decoder_grad': 0.1}
        config['finetune_ignored_params'] = ['model.fs2.encoder.embed_tokens', 'model.fs2.txt_embed', 'model.fs2.spk_embed']
    else:
        config['predict_dur'] = True; config['predict_pitch'] = True
        config['dur_prediction_args'] = {'arch': 'fs2', 'hidden_size': 512, 'dropout': 0.1, 'num_layers': 5, 'kernel_size': 3, 'log_offset': 1.0, 'loss_type': 'mse', 'lambda_pdur_loss': 0.3, 'lambda_wdur_loss': 1.0, 'lambda_sdur_loss': 3.0}
        config['pitch_prediction_args'] = {'pitd_norm_min': -8.0, 'pitd_norm_max': 8.0, 'pitd_clip_min': -12.0, 'pitd_clip_max': 12.0, 'repeat_bins': 64, 'backbone_type': 'wavenet', 'backbone_args': {'num_layers': 20, 'num_channels': 256, 'dilation_cycle_length': 5}}
        config['diffusion_type'] = 'reflow'
        config['finetune_ignored_params'] = ['model.spk_embed', 'model.fs2.txt_embed', 'model.fs2.encoder.embed_tokens']

    config['max_batch_frames'] = int(settings.get("Max Batch Frames", "50000"))
    config['max_batch_size'] = int(settings.get("Max Batch Size", "64")) if model_type == "Acoustic" else 48
    config['max_updates'] = max_updates
    config['pl_trainer_devices'] = 'auto'
    if has_cuda: config['pl_trainer_precision'] = '16-mixed'
    else: config['pl_trainer_accelerator'] = 'cpu'; config['pl_trainer_precision'] = '32'; config['pl_trainer_strategy'] = {'name': 'auto', 'process_group_backend': 'gloo', 'find_unused_parameters': False}
    
    config['finetune_enabled'] = bool(finetune_enabled)
    config['finetune_ckpt_path'] = finetune_ckpt_path if finetune_enabled else None
    
    filename = f"config_{model_type.lower()}.yaml"
    return log_msg(f"Auto-generated {filename} สำหรับ '{project_name}'\n   Max Updates: {max_updates}\n   Device: {settings.get('Accelerator', 'auto')}", "OK"), safe_yaml_dump(config)

def save_config(project_name, model_type, yaml_text):
    if not project_name or not yaml_text: return log_msg("ข้อมูลไม่ครบ", "ERROR")
    filename = "config_acoustic.yaml" if model_type == "Acoustic" else "config_variance.yaml"
    try:
        with open(DATA_DIR / project_name / filename, 'w', encoding='utf-8') as f: f.write(yaml_text)
        return log_msg(f"บันทึก {filename} สำเร็จ", "OK")
    except Exception as e: return log_msg(f"ผิดพลาด: {e}", "ERROR")

# ═══════════════════════════════════════════════════════════════
#  TAB 4 & 5: PREPROCESSING & TRAINING
# ═══════════════════════════════════════════════════════════════

def run_binarize(project_name, model_type):
    if not project_name: yield log_msg("กรุณาเลือกโปรเจกต์ก่อน", "ERROR"); return
    cfg = DATA_DIR / project_name / ("config_acoustic.yaml" if model_type == "Acoustic" else "config_variance.yaml")
    if not cfg.exists(): yield log_msg("ไม่พบ config file", "ERROR"); return
    
    # [Rule 6]
    purge_directory_junk(str(DATA_DIR / project_name / "raw"))
    
    yield log_msg(f"เริ่ม Binarization สำหรับ '{project_name}' ({model_type})...", "RUN") + "\n"
    for out in run_subprocess_streaming([get_python_executable(), str(SCRIPTS_DIR / "binarize.py"), "--config", str(cfg)]): yield out

def run_training(project_name, model_type, exp_name, resume):
    if not project_name: yield log_msg("กรุณาเลือกโปรเจกต์ก่อน", "ERROR"); return
    exp_name = exp_name.strip() or f"{project_name}_{model_type.lower()}"
    cfg = DATA_DIR / project_name / ("config_acoustic.yaml" if model_type == "Acoustic" else "config_variance.yaml")
    if not cfg.exists(): yield log_msg("ไม่พบ config file", "ERROR"); return
    
    cmd = [get_python_executable(), str(SCRIPTS_DIR / "train.py"), "--config", str(cfg), "--exp_name", exp_name]
    if not resume: cmd.append("--reset")
    
    yield log_msg(f"เริ่ม {'Resume' if resume else 'New'} Training: '{exp_name}'", "RUN") + "\n"
    for out in run_subprocess_streaming(cmd): yield out

def list_checkpoints(exp_name):
    if not exp_name or not exp_name.strip(): return "ยังไม่ได้ระบุชื่อ Experiment"
    ckpt_dir = CHECKPOINTS_DIR / exp_name.strip()
    if not ckpt_dir.exists(): return f"ไม่พบ checkpoints สำหรับ '{exp_name}'"
    lines = [f"📁 Checkpoints for '{exp_name}':", ""]
    if (ckpt_dir / "config.yaml").exists(): lines.append("   ✅ config.yaml — saved")
    ckpts = sorted(ckpt_dir.glob("model_ckpt_steps_*.ckpt"), reverse=True)
    if not ckpts: lines.append("   (ยังไม่มี checkpoint)")
    for ckpt in ckpts: lines.append(f"   📌 Step {ckpt.stem.replace('model_ckpt_steps_', ''):>8}  |  {ckpt.stat().st_size / (1024*1024):>7.1f} MB  |  {datetime.fromtimestamp(ckpt.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)

def list_all_experiments():
    return sorted([d.name for d in (CHECKPOINTS_DIR.iterdir() if CHECKPOINTS_DIR.exists() else []) if d.is_dir() and d.name != '.gitkeep' and ((d / "config.yaml").exists() or list(d.glob("*.ckpt")))])

# ═══════════════════════════════════════════════════════════════
#  TAB 6: EXPORT & PACKAGE
# ═══════════════════════════════════════════════════════════════

def get_checkpoint_steps(exp_name):
    return [ckpt.stem.replace("model_ckpt_steps_", "") for ckpt in sorted((CHECKPOINTS_DIR / exp_name).glob("model_ckpt_steps_*.ckpt"), reverse=True)] if exp_name and (CHECKPOINTS_DIR / exp_name).exists() else []

def run_export(project_name, exp_name, export_type, ckpt_step):
    if not exp_name: yield log_msg("กรุณาระบุชื่อ Experiment", "ERROR"); return
    out_dir = str(DATA_DIR / project_name / "export") if project_name else None
    
    if export_type == "Vocoder (NSF-HiFiGAN)": cmd = [get_python_executable(), str(SCRIPTS_DIR / "export.py"), "nsf-hifigan", "--config", str(CONFIGS_DIR / "acoustic.yaml"), "--ckpt", str(ROOT_DIR / get_vocoder_ckpt_path())]
    else:
        cmd = [get_python_executable(), str(SCRIPTS_DIR / "export.py"), export_type.lower(), "--exp", exp_name]
        if ckpt_step: cmd.extend(["--ckpt", str(ckpt_step)])
    if out_dir: cmd.extend(["--out", out_dir])

    yield log_msg(f"เริ่ม Export {export_type} → ONNX", "RUN") + "\n"
    for out in run_subprocess_streaming(cmd): yield out

def package_for_openutau(project_name, exp_name, voicebank_name, author, description, version, web_link):
    if not project_name or not exp_name: return log_msg("กรุณาระบุโปรเจกต์และ Experiment", "ERROR")
    export_dir = DATA_DIR / project_name / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    log = [log_msg("เริ่มสร้างชุดคลังเสียง OpenUtau...", "RUN")]
    
    meta = json.load(open(DATA_DIR / project_name / "project_meta.json", 'r', encoding='utf-8')) if (DATA_DIR / project_name / "project_meta.json").exists() else {}
    char = OrderedDict({'name': voicebank_name or project_name, 'singer': meta.get('speaker_name', project_name), 'author': author or "DiffSinger Trainer UI", 'version': version or "1.0.0", 'description': description or "DiffSinger voice bank", 'web': web_link or '', 'singer_type': 'diffsinger', 'language': meta.get('language', 'Custom-Language'), 'image': 'avatar.png'})
    with open(export_dir / "character.yaml", 'w', encoding='utf-8') as f: yaml.dump(dict(char), f, allow_unicode=True, default_flow_style=False)
    log.append(log_msg("สร้าง character.yaml สำเร็จ", "OK"))
    
    dict_src = (DATA_DIR / project_name / "custom_dictionary.txt") if meta.get('language') == "Custom-Language" else (DICTIONARIES_DIR / "opencpop-extension.txt" if meta.get('language') == "zh" else None)
    if dict_src and dict_src.exists(): shutil.copy2(dict_src, export_dir / "dsdict.txt"); log.append(log_msg("คัดลอก dictionary", "OK"))
    
    if (ROOT_DIR / "artifacts" / exp_name).exists():
        for f in (ROOT_DIR / "artifacts" / exp_name).iterdir():
            if f.suffix in ['.onnx', '.yaml', '.json', '.txt', '.emb']: shutil.copy2(f, export_dir / f.name)
            
    onnx_files = list(export_dir.glob("*.onnx"))
    if onnx_files:
        log.append(log_msg(f"พบ ONNX models {len(onnx_files)} ไฟล์", "OK"))
        for of in onnx_files: log.append(f"   📦 {of.name} ({of.stat().st_size / (1024*1024):.1f} MB)")
    else: log.append(log_msg("ยังไม่มี ONNX models — กรุณา Export ONNX ก่อน", "WARN"))

    try:
        zip_path = DATA_DIR / project_name / f"{(voicebank_name or project_name).replace(' ', '_')}_OpenUtau"
        shutil.make_archive(str(zip_path), 'zip', str(export_dir))
        log.extend(["", log_msg("📦 สร้าง Package สำเร็จ!", "OK"), f"   ไฟล์: {zip_path}.zip", "   ✨ พร้อมใช้งานใน OpenUtau!"])
    except Exception as e: log.append(log_msg(f"ไม่สามารถสร้าง zip: {e}", "ERROR"))
    return "\n".join(log)

# ═══════════════════════════════════════════════════════════════
#  BUILD GRADIO UI
# ═══════════════════════════════════════════════════════════════
def build_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="The Diff Singer Trainer UI") as app:
        gr.HTML("""<div class="app-header"><h1>🎵 The Diff Singer Trainer UI</h1><p>ระบบเทรนคลังเสียง DiffSinger AI ครบวงจร — รองรับ CPU (Intel/AMD) & NVIDIA GPU</p></div>""")

        with gr.Tabs():
            with gr.Tab("🖥️ System Info"):
                gr.Markdown("### ข้อมูลระบบ & การตั้งค่าอัตโนมัติ")
                sys_refresh_btn = gr.Button("🔄 ตรวจจับระบบ (Detect System)", elem_classes=["primary-btn"])
                sys_info_box = gr.Textbox(label="System Information", lines=20, interactive=False, elem_classes=["log-output"])
                sys_refresh_btn.click(fn=detect_system_info, outputs=sys_info_box)

            with gr.Tab("📂 Data Manager"):
                with gr.Accordion("🆕 สร้างโปรเจกต์ใหม่ (New Project)", open=True):
                    with gr.Row():
                        new_proj_name = gr.Textbox(label="ชื่อโปรเจกต์", placeholder="my_voicebank", scale=2)
                        new_speaker_name = gr.Textbox(label="ชื่อ Speaker", placeholder="Singer01", scale=2)
                        new_speaker_id = gr.Number(label="Speaker ID", value=0, precision=0, scale=1)
                    with gr.Row():
                        new_language = gr.Dropdown(label="ภาษา", choices=["Custom-Language (อัตโนมัติจากข้อมูล)", "zh", "ja", "en", "th"], value="Custom-Language (อัตโนมัติจากข้อมูล)", scale=3)
                        create_proj_btn = gr.Button("✨ สร้างโปรเจกต์", elem_classes=["success-btn"], scale=1)
                data_log = gr.Textbox(label="📋 Log", lines=3, interactive=False, elem_classes=["log-output"])
                
                with gr.Row():
                    project_selector = gr.Dropdown(label="เลือกโปรเจกต์ (Select Project)", choices=list_existing_projects(), scale=3, allow_custom_value=True)
                    refresh_proj_btn = gr.Button("🔄", scale=0)
                project_stats_box = gr.Textbox(label="📊 สถิติ", lines=8, interactive=False, elem_classes=["log-output"])
                
                with gr.Accordion("🎵 อัปโหลดไฟล์เสียง", open=False):
                    audio_upload = gr.File(label="เลือกไฟล์เสียง (รองรับ .wav, .flac)", file_count="multiple", file_types=[".wav", ".flac"])
                    upload_btn = gr.Button("📤 อัปโหลดไฟล์เสียง", elem_classes=["primary-btn"])
                    upload_log = gr.Textbox(label="Log การอัปโหลด", lines=5, interactive=False, elem_classes=["log-output"])
                
                with gr.Accordion("📝 Transcription Editor", open=False):
                    trans_editor = gr.Textbox(label="transcriptions.csv", lines=10)
                    with gr.Row():
                        load_trans_btn = gr.Button("📂 โหลด Transcription")
                        save_trans_btn = gr.Button("💾 บันทึก Transcription", elem_classes=["success-btn"])
                    trans_log = gr.Textbox(label="Log", lines=2, interactive=False, elem_classes=["log-output"])
                
                with gr.Accordion("🔧 เครื่องมือจัดการข้อมูล", open=False):
                    with gr.Row():
                        dedup_btn = gr.Button("🔍 ลบไฟล์เสียงซ้ำ", elem_classes=["danger-btn"])
                        scan_ph_btn = gr.Button("📖 สแกนหน่วยเสียง")
                    dedup_log = gr.Textbox(label="Log ลบไฟล์ซ้ำ", lines=4, interactive=False, elem_classes=["log-output"])
                    dict_editor = gr.Textbox(label="Custom Dictionary Content", lines=8)
                    save_dict_btn = gr.Button("💾 บันทึก Custom Dictionary", elem_classes=["success-btn"])
                    dict_log = gr.Textbox(label="Log", lines=2, interactive=False, elem_classes=["log-output"])

                create_proj_btn.click(fn=lambda name, spk, sid, lang: create_project(name, spk, sid, lang.split()[0] if "Custom" not in lang else "Custom-Language"), inputs=[new_proj_name, new_speaker_name, new_speaker_id, new_language], outputs=[data_log, project_selector])
                refresh_proj_btn.click(fn=lambda: gr.update(choices=list_existing_projects()), outputs=project_selector)
                project_selector.change(fn=get_project_stats, inputs=project_selector, outputs=project_stats_box)
                upload_btn.click(fn=upload_audio_files, inputs=[project_selector, audio_upload], outputs=[upload_log, project_stats_box])
                load_trans_btn.click(fn=load_transcriptions, inputs=project_selector, outputs=trans_editor)
                save_trans_btn.click(fn=save_transcriptions, inputs=[project_selector, trans_editor], outputs=trans_log)
                dedup_btn.click(fn=remove_duplicates, inputs=project_selector, outputs=dedup_log)
                scan_ph_btn.click(fn=scan_phonemes_from_data, inputs=project_selector, outputs=[dict_log, dict_editor])
                save_dict_btn.click(fn=save_custom_dictionary, inputs=[project_selector, dict_editor], outputs=dict_log)

            with gr.Tab("⚙️ Config Generator"):
                with gr.Row():
                    cfg_project = gr.Dropdown(label="เลือกโปรเจกต์", choices=list_existing_projects(), scale=3, allow_custom_value=True)
                    cfg_model_type = gr.Radio(label="ประเภทโมเดล", choices=["Acoustic", "Variance"], value="Acoustic", scale=2)
                cfg_max_updates = gr.Number(label="🎯 Max Updates", value=160000, precision=0)
                with gr.Accordion("🔧 Finetune Mode", open=False):
                    cfg_finetune = gr.Checkbox(label="เปิดใช้ Finetune Mode", value=False)
                    cfg_finetune_ckpt = gr.Textbox(label="Finetune Checkpoint Path")
                with gr.Row():
                    gen_config_btn = gr.Button("🔄 Auto-Generate Config", elem_classes=["primary-btn"])
                    save_config_btn = gr.Button("💾 บันทึก Config", elem_classes=["success-btn"])
                cfg_log = gr.Textbox(label="📋 Log", lines=3, interactive=False, elem_classes=["log-output"])
                cfg_yaml_preview = gr.Code(label="Config YAML", language="yaml", lines=20)
                
                gen_config_btn.click(fn=generate_config, inputs=[cfg_project, cfg_model_type, cfg_max_updates, cfg_finetune, cfg_finetune_ckpt], outputs=[cfg_log, cfg_yaml_preview])
                save_config_btn.click(fn=save_config, inputs=[cfg_project, cfg_model_type, cfg_yaml_preview], outputs=cfg_log)
                cfg_project.focus(fn=lambda: gr.update(choices=list_existing_projects()), outputs=cfg_project)

            with gr.Tab("🔄 Preprocessing & Training"):
                gr.Markdown("### 1. Binarization")
                with gr.Row():
                    bin_project = gr.Dropdown(label="เลือกโปรเจกต์", choices=list_existing_projects(), scale=3, allow_custom_value=True)
                    bin_model_type = gr.Radio(label="ประเภทโมเดล", choices=["Acoustic", "Variance"], value="Acoustic", scale=2)
                with gr.Row():
                    bin_start_btn = gr.Button("▶️ เริ่ม Binarization", elem_classes=["primary-btn"], scale=1)
                    bin_stop_btn = gr.Button("⏹️ หยุด", elem_classes=["danger-btn"], scale=0)
                bin_log = gr.Textbox(label="📋 Real-time Log", lines=10, interactive=False, elem_classes=["log-output"])
                bin_start_btn.click(fn=run_binarize, inputs=[bin_project, bin_model_type], outputs=bin_log)
                bin_stop_btn.click(fn=stop_active_process, outputs=bin_log)
                
                gr.Markdown("### 2. Training")
                train_exp_name = gr.Textbox(label="ชื่อ Experiment", placeholder="my_voicebank_acoustic")
                with gr.Row():
                    train_start_btn = gr.Button("▶️ เริ่มเทรน (New)", elem_classes=["primary-btn"])
                    train_resume_btn = gr.Button("⏩ Resume", elem_classes=["success-btn"])
                    train_stop_btn = gr.Button("⏹️ หยุด (Stop)", elem_classes=["danger-btn"])
                train_log = gr.Textbox(label="📋 Training Log", lines=15, interactive=False, elem_classes=["log-output"])
                train_start_btn.click(fn=lambda proj, mt, exp: run_training(proj, mt, exp, False), inputs=[bin_project, bin_model_type, train_exp_name], outputs=train_log)
                train_resume_btn.click(fn=lambda proj, mt, exp: run_training(proj, mt, exp, True), inputs=[bin_project, bin_model_type, train_exp_name], outputs=train_log)
                train_stop_btn.click(fn=stop_active_process, outputs=train_log)

            with gr.Tab("📦 Export & Package"):
                with gr.Accordion("⚡ Export ONNX Models", open=True):
                    with gr.Row():
                        exp_project = gr.Dropdown(label="เลือกโปรเจกต์", choices=list_existing_projects(), scale=2, allow_custom_value=True)
                        exp_exp_name = gr.Dropdown(label="เลือก Experiment", choices=list_all_experiments(), scale=2, allow_custom_value=True)
                    with gr.Row():
                        exp_type = gr.Radio(label="ประเภท Export", choices=["Acoustic", "Variance", "Vocoder (NSF-HiFiGAN)"], value="Acoustic", scale=3)
                        exp_ckpt_step = gr.Dropdown(label="Checkpoint Step", choices=[], allow_custom_value=True, scale=1)
                    with gr.Row():
                        export_btn = gr.Button("📤 Export ONNX", elem_classes=["primary-btn"])
                        export_stop_btn = gr.Button("⏹️ หยุด", elem_classes=["danger-btn"])
                    export_log = gr.Textbox(label="📋 Export Log", lines=10, interactive=False, elem_classes=["log-output"])
                
                with gr.Accordion("🎤 Voice Bank Info", open=True):
                    with gr.Row():
                        vb_name = gr.Textbox(label="ชื่อ Voice Bank", scale=2)
                        vb_author = gr.Textbox(label="ผู้สร้าง (Author)", scale=2)
                    with gr.Row():
                        vb_version = gr.Textbox(label="เวอร์ชัน", value="1.0.0", scale=1)
                        vb_web = gr.Textbox(label="Website", scale=2)
                    vb_desc = gr.Textbox(label="คำอธิบาย (Description)", lines=2)
                    package_btn = gr.Button("📦 สร้างชุดคลังเสียง OpenUtau", elem_classes=["success-btn"])
                    package_log = gr.Textbox(label="📋 Package Log", lines=8, interactive=False, elem_classes=["log-output"])

                exp_exp_name.change(fn=lambda exp: gr.update(choices=get_checkpoint_steps(exp), value=get_checkpoint_steps(exp)[0] if get_checkpoint_steps(exp) else None), inputs=exp_exp_name, outputs=exp_ckpt_step)
                exp_exp_name.focus(fn=lambda: gr.update(choices=list_all_experiments()), outputs=exp_exp_name)
                export_btn.click(fn=run_export, inputs=[exp_project, exp_exp_name, exp_type, exp_ckpt_step], outputs=export_log)
                export_stop_btn.click(fn=stop_active_process, outputs=export_log)
                package_btn.click(fn=package_for_openutau, inputs=[exp_project, exp_exp_name, vb_name, vb_author, vb_desc, vb_version, vb_web], outputs=package_log)

    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, inbrowser=True)