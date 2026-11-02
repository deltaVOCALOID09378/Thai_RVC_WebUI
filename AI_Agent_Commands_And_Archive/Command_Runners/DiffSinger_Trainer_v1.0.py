# -*- coding: utf-8 -*-
# ==========================================
# Made And Checked By DELTA SYNTH & Staff
# Original by DELTA SYNTH & Gemini AI
# Version: 1.2
# File: DiffSinger_Trainer_v1.2.py
# ==========================================
"""
DiffSinger_Trainer_v1.2.py (Version 1.2.0 - Architecture Hardened Edition)
ตัวช่วยเทรนโมเดล DiffSinger (Acoustic / Variance) พร้อมตรวจจับ GPU และตั้งค่าอัตโนมัติ
DiffSinger training helper (Acoustic / Variance) with GPU detection and auto-configuration

หมายเหตุ / Notes:
- สคริปต์นี้เป็นไฟล์แยกอิสระ (Standalone) ไม่แก้ไขไฟล์ระบบเดิม
- ไม่แก้ไฟล์ .yaml ใช้การส่งค่าผ่าน --hparams แทนเพื่อความปลอดภัยของไฟล์ต้นฉบับ
- วิธีใช้ / Usage: ลากโฟลเดอร์ DiffSinger (openvpi) มาวางบน Start DiffSinger Trainer.bat
"""
import os
import sys
import queue
import threading
import subprocess
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

APP_TITLE = "DELTA SYNTH: DiffSinger Trainer v1.2 | ตัวช่วยเทรน DiffSinger"
CREDIT = "Made And Checked By DELTA SYNTH & Staff\nOriginal by DELTA SYNTH & Gemini AI"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

def detect_gpus():
    """Return list of (index, name, vram_mb) using nvidia-smi with a 5-second timeout / ตรวจจับ GPU ด้วย nvidia-smi"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, creationflags=NO_WINDOW, stderr=subprocess.DEVNULL, timeout=5)
        gpus = []
        for line in out.strip().splitlines():
            idx, name, mem = [p.strip() for p in line.split(",")]
            gpus.append((int(idx), name, int(float(mem))))
        return gpus
    except Exception:
        return []

def auto_settings(vram_mb):
    """Pick batch settings by VRAM capacity (in MB) / เลือกค่าตามขนาด VRAM"""
    if vram_mb <= 0:
        return 8000, 8, "32-true"
    gb = vram_mb / 1024
    if gb <= 6.5:
        return 15000, 16, "16-mixed"
    if gb <= 8.5:
        return 25000, 24, "16-mixed"
    if gb <= 12.5:
        return 40000, 32, "16-mixed"
    if gb <= 16.5:
        return 50000, 48, "16-mixed"
    return 80000, 64, "16-mixed"

def find_python(root):
    """Prefer virtual environment python.exe inside the DiffSinger folder based on DELTA SYNTH Standards"""
    # ลำดับความสำคัญ 1: โฟลเดอร์สภาพแวดล้อมภายในโปรเจกต์
    for sub in ("runtime", "venv", ".venv", "env"):
        cand_rt = os.path.join(root, sub, "python.exe")
        if os.path.isfile(cand_rt):
            return cand_rt
        cand_script = os.path.join(root, sub, "Scripts", "python.exe")
        if os.path.isfile(cand_script):
            return cand_script
            
    # ลำดับความสำคัญ 2: Path มาตรฐานของระบบ RVC DELTA SYNTH
    fallback = r"A:\RVC1006AMD_Intel1\runtime\python.exe"
    if os.path.isfile(fallback):
        return fallback
        
    return sys.executable

class TrainerApp(tk.Tk):
    """Main Tkinter GUI application for DiffSinger binarization, training, TensorBoard, and ONNX export."""
    def __init__(self, initial_root=""):
        """Initialize the DiffSinger Trainer GUI state, variables, and event loop callbacks."""
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("920x720")
        self.minsize(820, 620)
        self.proc = None
        self.tb_proc = None
        self._busy = False
        self._stop_requested = False
        self.log_q = queue.Queue()
        self.gpus = detect_gpus()

        self.v_root = tk.StringVar(value=initial_root)
        self.v_python = tk.StringVar(value=sys.executable)
        self.v_type = tk.StringVar(value="acoustic")
        self.v_cfg = tk.StringVar()
        self.v_exp = tk.StringVar(value="my_voice")
        self.v_gpu = tk.StringVar()
        self.v_frames = tk.StringVar()
        self.v_bsize = tk.StringVar()
        self.v_prec = tk.StringVar()
        self.v_extra = tk.StringVar()
        self.v_reset = tk.BooleanVar(value=False)

        self._build()
        self._init_gpu()
        if initial_root:
            self.on_root_changed()
        self.after(100, self._drain_log)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build(self):
        """Construct and layout all Tkinter widgets for project paths, hardware settings, and controls."""
        pad = {"padx": 6, "pady": 4}
        top = ttk.LabelFrame(self, text="โปรเจกต์ / Project Configuration")
        top.pack(fill="x", padx=8, pady=6)
        top.columnconfigure(1, weight=1)

        rows = [
            ("โฟลเดอร์ DiffSinger / DiffSinger Directory", self.v_root, self.pick_root),
            ("สภาพแวดล้อมไพทอน / Python (venv)", self.v_python, self.pick_python),
            ("ไฟล์คอนฟิก / Config .yaml", self.v_cfg, self.pick_cfg),
        ]
        for r, (label, var, cmd) in enumerate(rows):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", **pad)
            ent = ttk.Entry(top, textvariable=var)
            ent.grid(row=r, column=1, sticky="ew", **pad)
            if r == 0:
                ent.bind("<FocusOut>", lambda e: self.on_root_changed())
                ent.bind("<Return>", lambda e: self.on_root_changed())
            ttk.Button(top, text="เลือก / Browse", command=cmd).grid(row=r, column=2, **pad)

        ttk.Label(top, text="ชนิดโมเดล / Model Type").grid(row=3, column=0, sticky="w", **pad)
        cb = ttk.Combobox(top, textvariable=self.v_type, values=["acoustic", "variance"],
                          state="readonly", width=14)
        cb.grid(row=3, column=1, sticky="w", **pad)
        cb.bind("<<ComboboxSelected>>", lambda e: self.default_cfg(preserve_existing=True))

        ttk.Label(top, text="ชื่อการทดลอง / Experiment Name").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.v_exp).grid(row=4, column=1, sticky="ew", **pad)

        mid = ttk.LabelFrame(self, text="การตั้งค่าฮาร์ดแวร์ / Hardware & Auto Settings")
        mid.pack(fill="x", padx=8, pady=6)
        mid.columnconfigure(1, weight=1)
        
        ttk.Label(mid, text="ระบบการ์ดจอ / GPU").grid(row=0, column=0, sticky="w", **pad)
        self.cb_gpu = ttk.Combobox(mid, textvariable=self.v_gpu, state="readonly")
        self.cb_gpu.grid(row=0, column=1, sticky="ew", **pad)
        self.cb_gpu.bind("<<ComboboxSelected>>", lambda e: self.apply_auto())
        ttk.Button(mid, text="ตั้งค่าอัตโนมัติ / Auto", command=self.apply_auto).grid(row=0, column=2, **pad)

        grid = ttk.Frame(mid)
        grid.grid(row=1, column=0, columnspan=3, sticky="w")
        for c, (label, var, w) in enumerate([
            ("max_batch_frames", self.v_frames, 10),
            ("max_batch_size", self.v_bsize, 8),
            ("precision", self.v_prec, 10)]):
            ttk.Label(grid, text=label).grid(row=0, column=c * 2, **pad)
            ttk.Entry(grid, textvariable=var, width=w).grid(row=0, column=c * 2 + 1, **pad)

        ttk.Label(mid, text="ค่าเพิ่มเติม / Extra --hparams (key=value)").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(mid, textvariable=self.v_extra).grid(row=2, column=1, columnspan=2, sticky="ew", **pad)
        
        ttk.Checkbutton(mid, text="เริ่มใหม่ (--reset) / Reset training state",
                        variable=self.v_reset).grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=4)
        self.b_bin = ttk.Button(btns, text="1) เตรียมข้อมูล / Binarize", command=self.do_binarize)
        self.b_train = ttk.Button(btns, text="2) เริ่มเทรน / Start Training", command=self.do_train)
        self.b_stop = ttk.Button(btns, text="หยุดการทำงาน / Stop", command=self.do_stop)
        self.b_tb = ttk.Button(btns, text="เปิด TensorBoard", command=self.do_tensorboard)
        self.b_exp = ttk.Button(btns, text="3) ส่งออก ONNX / Export ONNX", command=self.do_export)
        
        for b in (self.b_bin, self.b_train, self.b_stop, self.b_tb, self.b_exp):
            b.pack(side="left", padx=4)

        self.status = tk.StringVar(value="พร้อมใช้งาน / System Ready")
        ttk.Label(self, textvariable=self.status, anchor="w", foreground="#CC2200", font=("Leelawadee UI", 10, "bold")).pack(fill="x", padx=10)
        
        self.log = scrolledtext.ScrolledText(self, height=18, state="disabled",
                                             bg="#1A1A1A", fg="#F0F0F0", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=8, pady=6)
        ttk.Label(self, text=CREDIT, foreground="#666").pack(pady=(0, 4))

    def _init_gpu(self):
        """Populate the GPU selector dropdown and apply automatic VRAM-based settings."""
        if self.gpus:
            vals = [f"{i}: {n} ({m} MB)" for i, n, m in self.gpus]
        else:
            vals = ["CPU (ไม่พบ GPU NVIDIA / NVIDIA GPU not found)"]
        self.cb_gpu["values"] = vals
        self.cb_gpu.current(0)
        self.apply_auto()

    def apply_auto(self):
        """Calculate and apply recommended batch frames, batch size, and precision for the selected GPU."""
        vram = 0
        sel = self.cb_gpu.current()
        if self.gpus and 0 <= sel < len(self.gpus):
            vram = self.gpus[sel][2]
        frames, bsize, prec = auto_settings(vram)
        self.v_frames.set(str(frames))
        self.v_bsize.set(str(bsize))
        self.v_prec.set(prec)
        self.say(f"[AUTO SETTINGS] VRAM={vram} MB -> frames={frames}, batch={bsize}, precision={prec}")

    def on_root_changed(self):
        """Refresh Python executable path and default configuration when the root folder changes."""
        root = self.v_root.get().strip().strip('"')
        self.v_root.set(root)
        if os.path.isdir(root):
            self.v_python.set(find_python(root))
            self.default_cfg(preserve_existing=True)

    def default_cfg(self, preserve_existing=True):
        """Set default config path only if no valid custom config is currently selected."""
        current_cfg = self.v_cfg.get().strip()
        root = self.v_root.get().strip()
        default_acoustic = os.path.join(root, "configs", "acoustic.yaml")
        default_variance = os.path.join(root, "configs", "variance.yaml")
        if preserve_existing and current_cfg and os.path.isfile(current_cfg):
            if os.path.normcase(current_cfg) not in (
                os.path.normcase(default_acoustic),
                os.path.normcase(default_variance),
            ):
                return
        cand = os.path.join(root, "configs", f"{self.v_type.get()}.yaml")
        if os.path.isfile(cand):
            self.v_cfg.set(cand)

    def pick_root(self):
        d = filedialog.askdirectory(title="เลือกโฟลเดอร์ DiffSinger / Select DiffSinger Directory")
        if d:
            self.v_root.set(d)
            self.on_root_changed()

    def pick_python(self):
        f = filedialog.askopenfilename(title="Select Python executable", filetypes=[("Python EXE", "*.exe")])
        if f:
            self.v_python.set(f)

    def pick_cfg(self):
        f = filedialog.askopenfilename(title="Select Configuration YAML", filetypes=[("YAML Files", "*.yaml *.yml")],
                                       initialdir=self.v_root.get() or None)
        if f:
            self.v_cfg.set(f)

    def say(self, text):
        self.log_q.put(text + "\n")
        print(text, flush=True)

    def _drain_log(self):
        try:
            while True:
                chunk = self.log_q.get_nowait()
                self.log.configure(state="normal")
                self.log.insert("end", chunk)
                self.log.see("end")
                self.log.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _validate(self, need_cfg=True):
        root = self.v_root.get().strip()
        if not os.path.isdir(os.path.join(root, "scripts")):
            messagebox.showerror("Error", "ไม่พบโฟลเดอร์ scripts ใน DiffSinger\nDirectory 'scripts' not found in DiffSinger folder.")
            return None
        if need_cfg and not os.path.isfile(self.v_cfg.get()):
            messagebox.showerror("Error", "ไม่พบไฟล์คอนฟิก\nConfiguration file not found.")
            return None
        if not self.v_exp.get().strip():
            messagebox.showerror("Error", "กรุณาตั้งชื่อการทดลอง\nPlease set an experiment name.")
            return None
        return root

    def _env(self, root):
        env = os.environ.copy()
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        sel = self.cb_gpu.current()
        if self.gpus and 0 <= sel < len(self.gpus):
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpus[sel][0])
        return env

    def run_cmd(self, cmd, root, label):
        if self._busy or (self.proc and self.proc.poll() is None):
            messagebox.showwarning("Busy", "ระบบกำลังประมวลผลงานอื่นอยู่\nA task is already running in the background.")
            return
        self._busy = True
        self._stop_requested = False
        self.say(f"\n=== {label} ===\n> {' '.join(cmd)}\n")
        self.status.set(f"กำลังทำงาน / Running: {label}")

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=root, env=self._env(root), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", creationflags=NO_WINDOW)
                for line in self.proc.stdout:
                    self.say(line.rstrip())
                code = self.proc.wait()
                if self._stop_requested:
                    self.say(f"=== {label} : หยุดโดยผู้ใช้ / Terminated by user (Exit Code {code}) ===")
                    self.status.set(f"หยุดแล้ว / Stopped: {label}")
                elif code == 0:
                    self.say(f"=== {label} : เสร็จสิ้น / Task Completed (Exit Code 0) ===")
                    self.status.set(f"เสร็จสิ้น / Done: {label}")
                else:
                    self.say(f"=== {label} : ล้มเหลว / Task Failed (Exit Code {code}) ===")
                    self.status.set(f"ผิดพลาด / Failed: {label} (Code {code})")
            except Exception as e:
                self.say(f"[CRITICAL ERROR] {e}")
                self.status.set("ผิดพลาดระบบ / System Error")
            finally:
                self._busy = False

        threading.Thread(target=worker, daemon=True).start()

    def do_binarize(self):
        root = self._validate()
        if root:
            cmd = [self.v_python.get(), "scripts/binarize.py", "--config", self.v_cfg.get()]
            if self.v_extra.get().strip():
                cmd.extend(["--hparams", self.v_extra.get().strip()])
            self.run_cmd(cmd, root, "Binarize / เตรียมข้อมูลการเทรน")

    def do_train(self):
        root = self._validate()
        if not root:
            return
        hp = [f"max_batch_frames={self.v_frames.get().strip()}",
              f"max_batch_size={self.v_bsize.get().strip()}",
              f"pl_trainer_precision={self.v_prec.get().strip()}"]
        if self.v_extra.get().strip():
            hp.append(self.v_extra.get().strip())
        cmd = [self.v_python.get(), "scripts/train.py", "--config", self.v_cfg.get(),
               "--exp_name", self.v_exp.get().strip(), "--hparams", ",".join(hp)]
        if self.v_reset.get():
            cmd.append("--reset")
        self.run_cmd(cmd, root, "Training / กระบวนการเทรนโมเดล")

    def do_export(self):
        root = self._validate(need_cfg=False)
        if root:
            self.run_cmd([self.v_python.get(), "scripts/export.py", self.v_type.get(),
                          "--exp", self.v_exp.get().strip()], root, "Export ONNX / ส่งออกไฟล์โมเดล ONNX")

    def do_stop(self):
        if self.proc and self.proc.poll() is None:
            self._stop_requested = True
            subprocess.call(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                            creationflags=NO_WINDOW, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
            self.say("[STOP] บังคับหยุดการทำงานสำเร็จ / Process terminated.")
            self.status.set("หยุดการทำงาน / Execution Stopped")

    def do_tensorboard(self):
        root = self._validate(need_cfg=False)
        if not root:
            return
        logdir = os.path.join(root, "checkpoints", self.v_exp.get().strip())
        if self.tb_proc and self.tb_proc.poll() is None:
            webbrowser.open("http://localhost:6006")
            return
        try:
            self.tb_proc = subprocess.Popen(
                [self.v_python.get(), "-m", "tensorboard.main", "--logdir", logdir, "--port", "6006"],
                cwd=root, creationflags=NO_WINDOW, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            self.say(f"[TB] TensorBoard Started: http://localhost:6006 (logdir={logdir})")

            def _open_if_alive():
                if self.tb_proc and self.tb_proc.poll() is None:
                    webbrowser.open("http://localhost:6006")
                else:
                    self.say("[ERROR] TensorBoard ปิดตัวกระทันหัน / TensorBoard exited prematurely. Check if installed.")

            self.after(3000, _open_if_alive)
        except Exception as e:
            self.say(f"[ERROR] TensorBoard Failed: {e}")

    def on_close(self):
        for p in (self.proc, self.tb_proc):
            if p and p.poll() is None:
                subprocess.call(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                                creationflags=NO_WINDOW, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        self.destroy()

if __name__ == "__main__":
    start_root = sys.argv[1].strip('"') if len(sys.argv) > 1 else ""
    TrainerApp(start_root).mainloop()