# -*- coding: utf-8 -*-
"""
DiffSinger_Trainer_v1.0.py
ตัวช่วยเทรนโมเดล DiffSinger (Acoustic / Variance) พร้อมตรวจจับ GPU และตั้งค่าอัตโนมัติ
DiffSinger training helper (Acoustic / Variance) with GPU detection and auto-configuration

Made And Checked By DELTA SYNTH & Gemini AI

หมายเหตุ / Notes:
- สคริปต์นี้เป็นไฟล์ใหม่แยกอิสระ ไม่แก้ไขไฟล์เดิมใน Thai_RVC_WebUI_v1
  This is a standalone new file; it does not modify existing Thai_RVC_WebUI_v1 files.
- ไม่แก้ไฟล์ .yaml ใช้การส่งค่าผ่าน --hparams แทน
  YAML files are never edited; overrides are passed through --hparams.
- วิธีใช้ / Usage: ลากโฟลเดอร์ DiffSinger (openvpi) มาวางบน Run_DiffSinger_Trainer_v1.0.bat
  Drag your DiffSinger (openvpi) folder onto Run_DiffSinger_Trainer_v1.0.bat
"""
import os
import sys
import queue
import threading
import subprocess
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

APP_TITLE = "DiffSinger Trainer v1.0 | ตัวช่วยเทรน DiffSinger"
CREDIT = "Made And Checked By DELTA SYNTH & Gemini AI"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------- helpers ---
def detect_gpus():
    """Return list of (index, name, vram_mb) using nvidia-smi / ตรวจจับ GPU ด้วย nvidia-smi"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, creationflags=NO_WINDOW, stderr=subprocess.DEVNULL)
        gpus = []
        for line in out.strip().splitlines():
            idx, name, mem = [p.strip() for p in line.split(",")]
            gpus.append((int(idx), name, int(float(mem))))
        return gpus
    except Exception:
        return []


def auto_settings(vram_mb):
    """Pick batch settings by VRAM / เลือกค่าตาม VRAM"""
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
    """Prefer venv inside the DiffSinger folder / ใช้ venv ในโฟลเดอร์ DiffSinger ก่อน"""
    for sub in ("venv", ".venv", "env"):
        cand = os.path.join(root, sub, "Scripts", "python.exe")
        if os.path.isfile(cand):
            return cand
    return sys.executable


# -------------------------------------------------------------------- GUI ---
class TrainerApp(tk.Tk):
    def __init__(self, initial_root=""):
        """Initialize the trainer window, hardware settings, and log polling."""
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("920x720")
        self.minsize(820, 620)
        self.proc = None
        self.tb_proc = None
        self.log_q = queue.Queue()
        self.gpus = detect_gpus()

        self.v_root = tk.StringVar(value=initial_root)
        self.v_python = tk.StringVar()
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

    # ---- UI
    def _build(self):
        """Create project inputs, hardware controls, task buttons, and the log view."""
        pad = {"padx": 6, "pady": 4}
        top = ttk.LabelFrame(self, text="โปรเจกต์ / Project")
        top.pack(fill="x", padx=8, pady=6)
        top.columnconfigure(1, weight=1)

        rows = [
            ("โฟลเดอร์ DiffSinger / DiffSinger folder", self.v_root, self.pick_root),
            ("Python (venv)", self.v_python, self.pick_python),
            ("ไฟล์คอนฟิก / Config .yaml", self.v_cfg, self.pick_cfg),
        ]
        for r, (label, var, cmd) in enumerate(rows):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(top, textvariable=var).grid(row=r, column=1, sticky="ew", **pad)
            ttk.Button(top, text="เลือก / Browse", command=cmd).grid(row=r, column=2, **pad)

        ttk.Label(top, text="ชนิดโมเดล / Model type").grid(row=3, column=0, sticky="w", **pad)
        cb = ttk.Combobox(top, textvariable=self.v_type, values=["acoustic", "variance"],
                          state="readonly", width=14)
        cb.grid(row=3, column=1, sticky="w", **pad)
        cb.bind("<<ComboboxSelected>>", lambda e: self.default_cfg())

        ttk.Label(top, text="ชื่อการทดลอง / Experiment name").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.v_exp).grid(row=4, column=1, sticky="ew", **pad)

        mid = ttk.LabelFrame(self, text="การตั้งค่าฮาร์ดแวร์ / Hardware & Auto settings")
        mid.pack(fill="x", padx=8, pady=6)
        mid.columnconfigure(1, weight=1)
        ttk.Label(mid, text="GPU").grid(row=0, column=0, sticky="w", **pad)
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

        ttk.Label(mid, text="ค่าเพิ่มเติม / Extra --hparams (key=value,key=value)").grid(
            row=2, column=0, sticky="w", **pad)
        ttk.Entry(mid, textvariable=self.v_extra).grid(row=2, column=1, columnspan=2, sticky="ew", **pad)
        ttk.Checkbutton(mid, text="เริ่มใหม่ (--reset) / Reset training state",
                        variable=self.v_reset).grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=4)
        self.b_bin = ttk.Button(btns, text="1) เตรียมข้อมูล / Binarize",
                                command=self.do_binarize)
        self.b_train = ttk.Button(btns, text="2) เริ่มเทรน / Start Training", command=self.do_train)
        self.b_stop = ttk.Button(btns, text="หยุด / Stop", command=self.do_stop)
        self.b_tb = ttk.Button(btns, text="TensorBoard", command=self.do_tensorboard)
        self.b_exp = ttk.Button(btns, text="3) ส่งออก ONNX / Export ONNX", command=self.do_export)
        for b in (self.b_bin, self.b_train, self.b_stop, self.b_tb, self.b_exp):
            b.pack(side="left", padx=4)

        self.status = tk.StringVar(value="พร้อม / Ready")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=10)
        self.log = scrolledtext.ScrolledText(self, height=18, state="disabled",
                                             bg="#111", fg="#ddd", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=8, pady=6)
        ttk.Label(self, text=CREDIT, foreground="#666").pack(pady=(0, 4))

    # ---- init / events
    def _init_gpu(self):
        """Populate the GPU selector and apply defaults for its first entry."""
        if self.gpus:
            vals = [f"{i}: {n} ({m} MB)" for i, n, m in self.gpus]
        else:
            vals = ["CPU (ไม่พบ GPU NVIDIA / No NVIDIA GPU)"]
        self.cb_gpu["values"] = vals
        self.cb_gpu.current(0)
        self.apply_auto()

    def apply_auto(self):
        """Set and log batch and precision defaults for the selected GPU or CPU."""
        vram = 0
        sel = self.cb_gpu.current()
        if self.gpus and 0 <= sel < len(self.gpus):
            vram = self.gpus[sel][2]
        frames, bsize, prec = auto_settings(vram)
        self.v_frames.set(str(frames))
        self.v_bsize.set(str(bsize))
        self.v_prec.set(prec)
        self.say(f"[AUTO] VRAM={vram} MB -> frames={frames}, batch={bsize}, precision={prec}")

    def on_root_changed(self):
        """Normalize the project path and select its Python and config if valid."""
        root = self.v_root.get().strip().strip('"')
        self.v_root.set(root)
        if os.path.isdir(root):
            self.v_python.set(find_python(root))
            self.default_cfg()

    def default_cfg(self):
        """Select the model type's default config when it exists in the project."""
        root = self.v_root.get()
        cand = os.path.join(root, "configs", f"{self.v_type.get()}.yaml")
        if os.path.isfile(cand):
            self.v_cfg.set(cand)

    def pick_root(self):
        """Choose a project folder and refresh its Python and config paths."""
        d = filedialog.askdirectory(title="เลือกโฟลเดอร์ DiffSinger / Select DiffSinger folder")
        if d:
            self.v_root.set(d)
            self.on_root_changed()

    def pick_python(self):
        """Choose a Python executable and update the field unless cancelled."""
        f = filedialog.askopenfilename(title="Python executable", filetypes=[("python.exe", "*.exe")])
        if f:
            self.v_python.set(f)

    def pick_cfg(self):
        """Choose a YAML config from a file dialog opened at the project folder."""
        f = filedialog.askopenfilename(title="Config", filetypes=[("YAML", "*.yaml *.yml")],
                                       initialdir=self.v_root.get() or None)
        if f:
            self.v_cfg.set(f)

    # ---- logging
    def say(self, text):
        """Queue a message for the GUI log and print it to stdout immediately."""
        self.log_q.put(text + "\n")
        print(text, flush=True)

    def _drain_log(self):
        """Append queued messages to the log view and schedule the next poll."""
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

    # ---- process control
    def _validate(self, need_cfg=True):
        """Return the project path, or show an error and return None if invalid.

        Require a scripts directory and experiment name, plus an existing
        config file when need_cfg is True.
        """
        root = self.v_root.get().strip()
        if not os.path.isdir(os.path.join(root, "scripts")):
            messagebox.showerror("Error", "ไม่พบโฟลเดอร์ scripts ใน DiffSinger\n"
                                          "Folder 'scripts' not found in DiffSinger folder.")
            return None
        if need_cfg and not os.path.isfile(self.v_cfg.get()):
            messagebox.showerror("Error", "ไม่พบไฟล์คอนฟิก\nConfig file not found.")
            return None
        if not self.v_exp.get().strip():
            messagebox.showerror("Error", "กรุณาตั้งชื่อการทดลอง\nPlease set an experiment name.")
            return None
        return root

    def _env(self, root):
        """Copy the environment with project imports, UTF-8, and any selected GPU."""
        env = os.environ.copy()
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        sel = self.cb_gpu.current()
        if self.gpus and 0 <= sel < len(self.gpus):
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpus[sel][0])
        return env

    def run_cmd(self, cmd, root, label):
        """Run a command in a daemon thread with output and status updates.

        Use root as the working directory and label for progress messages;
        show a warning and return if a tracked task is still running.
        """
        if self.proc and self.proc.poll() is None:
            messagebox.showwarning("Busy", "มีงานกำลังทำงานอยู่\nA task is already running.")
            return
        self.say(f"\n=== {label} ===\n> {' '.join(cmd)}\n")
        self.status.set(f"กำลังทำงาน / Running: {label}")

        def worker():
            """Launch the subprocess, stream its output, and report exit or error."""
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=root, env=self._env(root), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", creationflags=NO_WINDOW)
                for line in self.proc.stdout:
                    self.say(line.rstrip())
                code = self.proc.wait()
                self.say(f"=== {label} finished / เสร็จสิ้น (exit code {code}) ===")
                self.status.set(f"เสร็จสิ้น / Done: {label} (code {code})")
            except Exception as e:
                self.say(f"[ERROR] {e}")
                self.status.set("ผิดพลาด / Error")

        threading.Thread(target=worker, daemon=True).start()

    def do_binarize(self):
        """Validate the project and run data binarization with the chosen config."""
        root = self._validate()
        if root:
            self.run_cmd([self.v_python.get(), "scripts/binarize.py", "--config", self.v_cfg.get()],
                         root, "Binarize / เตรียมข้อมูล")

    def do_train(self):
        """Validate and launch training with batch, precision, and reset settings."""
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
        self.run_cmd(cmd, root, "Training / เทรนโมเดล")

    def do_export(self):
        """Validate the project and launch ONNX export for the chosen experiment."""
        root = self._validate(need_cfg=False)
        if root:
            self.run_cmd([self.v_python.get(), "scripts/export.py", self.v_type.get(),
                          "--exp", self.v_exp.get().strip()], root, "Export ONNX / ส่งออก ONNX")

    def do_stop(self):
        """Force-stop the tracked task's Windows process tree and update status."""
        if self.proc and self.proc.poll() is None:
            subprocess.call(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                            creationflags=NO_WINDOW, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
            self.say("[STOP] หยุดการทำงานแล้ว / Process stopped.")
            self.status.set("หยุดแล้ว / Stopped")

    def do_tensorboard(self):
        """Open an existing TensorBoard process or start one for the experiment."""
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
            self.say(f"[TB] TensorBoard: http://localhost:6006  (logdir={logdir})")
            self.after(3000, lambda: webbrowser.open("http://localhost:6006"))
        except Exception as e:
            self.say(f"[ERROR] TensorBoard: {e}")

    def on_close(self):
        """Kill active task and TensorBoard process trees, then destroy the window."""
        for p in (self.proc, self.tb_proc):
            if p and p.poll() is None:
                subprocess.call(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                                creationflags=NO_WINDOW, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        self.destroy()


if __name__ == "__main__":
    start_root = sys.argv[1].strip('"') if len(sys.argv) > 1 else ""
    TrainerApp(start_root).mainloop()
