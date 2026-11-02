# ==============================================================================
# ชื่อไฟล์ (File Name)       : unified_vocal_studio_core.py
# เวอร์ชั่นโค้ด (Version)     : v5.1 (Ultimate Unified Edition)
# ตำแหน่งจัดเก็บ (Location)  : วางไว้ในแฟ้มหลักของโปรแกรม (Root Directory) หรือในแฟ้ม src/
# หน้าที่ (Role/Function)    : ศูนย์กลางการจัดการที่อยู่ไฟล์โมเดลตามเวอร์ชั่น, ระบบแปลงหน่วยเสียง
#                            ภาษาไทย (DiffSinger/Arpasing/VCCV), และระบบปรับแต่งเสียงร้องอัตโนมัติ
# เครดิตผู้สร้าง (Credits)   : Made And Checked By DELTA SYNTH & All Agentic AI
#                            Original by RVC Project & OpenUtau / DiffSinger Community
# ==============================================================================

import os
import sys
import json
import math
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ==============================================================================
# ฐานข้อมูลการแปลงหน่วยเสียงภาษาไทย (Thai Phonemizer Databases)
# ==============================================================================
PHONEME_STANDARDS: Dict[str, Dict[str, Dict[str, str]]] = {
    "DiffSinger_AI": {
        "consonants": {
            "ก": "kk", "ค": "k", "ง": "ng", "จ": "j", "ช": "ch", "ซ": "s",
            "ด": "d", "ต": "tt", "ท": "t", "น": "n", "บ": "b", "ป": "pp",
            "พ": "p", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l",
            "ว": "w", "ฮ": "h", "อ": "-"
        },
        "vowels": {
            "อา": "a", "อี": "i", "อู": "u", "เอ": "e", "โอ": "o", "แอ": "A",
            "ออ": "O", "เอาะ": "O", "เออ": "E", "เอือ": "Ua", "อือ": "U",
            "เอีย": "ia", "อัว": "ua", "อัย": "I", "ไอ": "I", "ใอ": "I",
            "อาย": "I", "เอา": "au", "อาว": "au", "อำ": "am", "อัม": "am", "อาม": "am"
        },
        "endings": {
            "น": "n", "ม": "M", "ย": "Y", "ว": "W", "ง": "ng", "ก": "K", "บ": "B", "ด": "D"
        }
    },
    "Arpasing": {
        "consonants": {
            "ก": "k", "ค": "kh", "ง": "g", "จ": "j", "ช": "ch", "ซ": "s",
            "ด": "d", "ต": "t", "ท": "th", "น": "n", "บ": "b", "ป": "p",
            "พ": "ph", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l",
            "ว": "w", "ฮ": "h", "อ": "-"
        },
        "vowels": {
            "อา": "ah", "อี": "ih", "อู": "uh", "เอ": "eh", "โอ": "oh", "แอ": "ae",
            "ออ": "ao", "เออ": "er", "เอือ": "eu", "อือ": "ue", "เอีย": "ia",
            "อัว": "ua", "ไอ": "ay", "อัย": "ay", "ใอ": "ay", "อาย": "ay",
            "เอา": "aw", "อาว": "aw", "อำ": "am", "อาม": "am", "อัม": "am"
        },
        "endings": {
            "น": "n", "ม": "m", "ย": "y", "ว": "w", "ง": "g", "ก": "k", "บ": "b", "ด": "d"
        }
    },
    "VCCV": {
        "consonants": {
            "ก": "k", "ค": "kh", "ง": "g", "จ": "j", "ช": "ch", "ซ": "s",
            "ด": "d", "ต": "t", "ท": "th", "น": "n", "บ": "b", "ป": "p",
            "พ": "ph", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l",
            "ว": "w", "ฮ": "h", "อ": "-"
        },
        "vowels": {
            "อา": "a", "อี": "i", "อู": "u", "เอ": "e", "โอ": "o", "แอ": "@",
            "ออ": "Q", "เอาะ": "Q", "เออ": "3", "เอือ": "6", "อือ": "1",
            "เอีย": "ia", "อัว": "ua", "อัย": "I", "ไอ": "I", "ใอ": "I",
            "อาย": "I", "เอา": "aw", "อาว": "aw", "อำ": "am", "อัม": "am", "อาม": "am"
        },
        "endings": {
            "น": "n", "ม": "m", "ย": "y", "ว": "w", "ง": "g", "ก": "k", "บ": "b", "ด": "d"
        }
    }
}


# ==============================================================================
# ระบบจัดการเส้นทางไฟล์โมเดลอัจฉริยะ (Smart Model Path & Version Resolver)
# ==============================================================================
class ModelVersionManager:
    """
    ระบบตรวจสอบและจัดระเบียบที่อยู่ไฟล์โมเดลตามเวอร์ชั่นและประเภทของการประมวลผล
    (Model Directory & Version Routing Engine)
    """
    VERSION_DIRECTORIES: Dict[str, Dict[str, str]] = {
        "RVC_v1": {
            "weights": "assets/weights/v1",
            "indices": "assets/indices/v1",
            "configs": "configs/v1"
        },
        "RVC_v2": {
            "weights": "assets/weights/v2",
            "indices": "assets/indices/v2",
            "configs": "configs/v2"
        },
        "DiffSinger_v1": {
            "weights": "checkpoints/diffsinger_v1",
            "acoustic": "checkpoints/diffsinger_v1/acoustic",
            "variance": "checkpoints/diffsinger_v1/variance",
            "dictionaries": "dictionaries/diffsinger_v1"
        },
        "DiffSinger_v2": {
            "weights": "checkpoints/diffsinger_v2",
            "acoustic": "checkpoints/diffsinger_v2/acoustic",
            "variance": "checkpoints/diffsinger_v2/variance",
            "dictionaries": "dictionaries/diffsinger_v2"
        }
    }

    def __init__(self, base_root: Optional[str] = None) -> None:
        self.base_root = Path(base_root) if base_root else Path.cwd()
        self.registry_file = self.base_root / "model_registry_config.json"
        self.ensure_directory_structure()

    def ensure_directory_structure(self) -> None:
        """สร้างโครงสร้างแฟ้มสำหรับจัดเก็บโมเดลทุกเวอร์ชั่นโดยอัตโนมัติ (Initialize Directories)"""
        for ver_key, subdirs in self.VERSION_DIRECTORIES.items():
            for folder_type, rel_path in subdirs.items():
                target_dir = self.base_root / rel_path
                target_dir.mkdir(parents=True, exist_ok=True)

    def detect_model_version(self, file_path: Path) -> str:
        """
        ตรวจสอบเวอร์ชั่นของโมเดลจากชื่อไฟล์ นามสกุล หรือโครงสร้างภายในเบื้องต้น
        (Auto-detect Model Version from File Attributes)
        """
        name_lower = file_path.name.lower()
        ext = file_path.suffix.lower()

        if ext in [".ckpt", ".onnx", ".ds"]:
            if "v1" in name_lower:
                return "DiffSinger_v1"
            return "DiffSinger_v2"

        if ext in [".pth", ".index"]:
            if "v1" in name_lower or "32k" in name_lower:
                return "RVC_v1"
            return "RVC_v2"

        return "RVC_v2"

    def route_and_organize_model(
        self,
        source_path_str: str,
        forced_version: str = "Auto",
        copy_files: bool = True
    ) -> Dict[str, Any]:
        """
        จัดวางไฟล์หรือแฟ้มโมเดลไปยังตำแหน่งที่ถูกต้องตามเวอร์ชั่นที่ระบุหรือตรวจพบ
        (Route and Organize Model Files into Version-Specific Directories)
        """
        src_path = Path(source_path_str.strip('"').strip("'")).resolve()
        if not src_path.exists():
            return {
                "status": "error",
                "message_th": f"ไม่พบไฟล์หรือแฟ้มต้นทาง: {src_path}",
                "message_en": f"Source file or directory not found: {src_path}",
                "routed_files": []
            }

        files_to_process: List[Path] = []
        if src_path.is_file():
            files_to_process.append(src_path)
        else:
            for ext in ("*.pth", "*.index", "*.ckpt", "*.onnx", "*.json", "*.yaml", "*.txt"):
                files_to_process.extend(src_path.rglob(ext))

        routed_records: List[Dict[str, str]] = []

        for item in files_to_process:
            version_tag = (
                self.detect_model_version(item)
                if forced_version == "Auto"
                else forced_version
            )
            if version_tag not in self.VERSION_DIRECTORIES:
                version_tag = "RVC_v2"

            ver_map = self.VERSION_DIRECTORIES[version_tag]
            ext = item.suffix.lower()

            if ext == ".pth":
                dest_sub = ver_map.get("weights", "assets/weights/v2")
            elif ext == ".index":
                dest_sub = ver_map.get("indices", "assets/indices/v2")
            elif ext in [".ckpt", ".onnx"]:
                dest_sub = ver_map.get("acoustic", ver_map.get("weights", "checkpoints/diffsinger_v2"))
            elif ext in [".json", ".yaml"]:
                dest_sub = ver_map.get("configs", ver_map.get("weights", "configs/v2"))
            else:
                dest_sub = ver_map.get("dictionaries", "dictionaries/diffsinger_v2")

            dest_dir = self.base_root / dest_sub
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / item.name

            if copy_files and item.resolve() != dest_file.resolve():
                shutil.copy2(item, dest_file)
                final_location = str(dest_file)
            else:
                final_location = str(item)

            routed_records.append({
                "file_name": item.name,
                "version": version_tag,
                "source": str(item),
                "destination": final_location
            })

        self.save_registry(routed_records)
        return {
            "status": "success",
            "message_th": f"จัดระเบียบและลงทะเบียนไฟล์โมเดลสำเร็จจำนวน {len(routed_records)} รายการ",
            "message_en": f"Successfully routed and registered {len(routed_records)} model file(s).",
            "routed_files": routed_records
        }

    def save_registry(self, new_records: List[Dict[str, str]]) -> None:
        """บันทึกฐานข้อมูลที่อยู่ไฟล์โมเดลลงในไฟล์ JSON (Save Model Path Registry)"""
        existing_data: List[Dict[str, str]] = []
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = []

        record_map = {rec["file_name"]: rec for rec in existing_data}
        for rec in new_records:
            record_map[rec["file_name"]] = rec

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(list(record_map.values()), f, ensure_ascii=False, indent=4)


# ==============================================================================
# ระบบปรับแต่งเสียงร้องและหน่วยเสียงอัตโนมัติ (Vocal Tuning & Phoneme Engine)
# ==============================================================================
class AutomatedVocalTuningEngine:
    """
    ระบบขยายความยาวพยัญชนะเพื่อให้ร้องเพลงชัดเจน และสร้างเส้นโค้งระดับเสียงพร้อมลูกคออัตโนมัติ
    (Consonant Expansion & Automated Pitch/Vibrato Generator)
    """
    def __init__(
        self,
        consonant_expansion_ratio: float = 1.35,
        vibrato_rate_hz: float = 6.2,
        vibrato_depth_cents: float = 35.0
    ) -> None:
        self.consonant_expansion_ratio = max(1.0, min(consonant_expansion_ratio, 2.5))
        self.vibrato_rate_hz = vibrato_rate_hz
        self.vibrato_depth_cents = vibrato_depth_cents

    def convert_syllable(
        self,
        consonant_th: str,
        vowel_th: str,
        ending_th: str = "",
        standard: str = "DiffSinger_AI"
    ) -> Dict[str, Any]:
        """
        แปลงพยางค์ภาษาไทยเป็นหน่วยเสียงตามมาตรฐานที่เลือก พร้อมคำนวณความยาวพยัญชนะที่ขยายแล้ว
        (Convert Thai Syllable & Apply Consonant Expansion)
        """
        db = PHONEME_STANDARDS.get(standard, PHONEME_STANDARDS["DiffSinger_AI"])
        c_mapped = db["consonants"].get(consonant_th, "-")
        v_mapped = db["vowels"].get(vowel_th, "a")
        e_mapped = db["endings"].get(ending_th, "") if ending_th else ""

        base_consonant_ms = 65.0
        expanded_consonant_ms = round(base_consonant_ms * self.consonant_expansion_ratio, 2)

        phoneme_seq = [p for p in (c_mapped, v_mapped, e_mapped) if p]
        return {
            "standard": standard,
            "phonemes": " ".join(phoneme_seq),
            "consonant_duration_ms": expanded_consonant_ms,
            "expansion_applied": f"{int((self.consonant_expansion_ratio - 1.0) * 100)}%"
        }

    def generate_pitch_and_vibrato_curve(
        self,
        base_midi_note: int = 60,
        duration_sec: float = 1.0,
        sample_rate_hz: int = 100
    ) -> List[float]:
        """
        สร้างเส้นโค้งระดับเสียงและลูกคอความถี่สูงอย่างเป็นธรรมชาติ (Generate Pitch & Fast-Paced Vibrato Curve)
        """
        total_frames = max(1, int(duration_sec * sample_rate_hz))
        pitch_curve: List[float] = []
        onset_Portion = 0.25

        for idx in range(total_frames):
            t = idx / float(sample_rate_hz)
            progress = idx / float(total_frames)

            # สร้างการเอื้อนช่วงต้นโน้ตอย่างนุ่มนวล (Natural Vocal Inflection Glide)
            if progress < 0.12:
                inflection_cents = -40.0 * (1.0 - (progress / 0.12))
            else:
                inflection_cents = 0.0

            # สร้างลูกคอความถี่สูงในช่วงท้ายของโน้ต (Fast-Paced Vibrato Modulation)
            if progress >= onset_Portion:
                vib_env = min(1.0, (progress - onset_Portion) / 0.35)
                vibrato_cents = (
                    self.vibrato_depth_cents
                    * vib_env
                    * math.sin(2.0 * math.pi * self.vibrato_rate_hz * t)
                )
            else:
                vibrato_cents = 0.0

            current_midi_pitch = base_midi_note + ((inflection_cents + vibrato_cents) / 100.0)
            pitch_curve.append(round(current_midi_pitch, 4))

        return pitch_curve


# ==============================================================================
# ส่วนติดต่อผู้ใช้งานแบบสองภาษา (Bilingual Graphical User Interface - TH/EN)
# ==============================================================================
class UnifiedVocalStudioApp:
    def __init__(self, root: tk.Tk, initial_target_path: Optional[str] = None) -> None:
        self.root = root
        self.root.title("Unified Vocal Studio v5.1 - ระบบบริหารจัดการโมเดลและปรับแต่งเสียงร้องสมบูรณ์แบบ")
        self.root.geometry("920x680")

        self.model_manager = ModelVersionManager()
        self.tuning_engine = AutomatedVocalTuningEngine()

        self._build_ui()

        if initial_target_path:
            self.path_entry_var.set(initial_target_path)
            self.execute_model_routing()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ส่วนหัวโปรแกรม (Header Section)
        header_lbl = ttk.Label(
            main_frame,
            text="Unified Vocal Studio & Smart Model Path Resolver (v5.1)\nระบบจัดการที่อยู่ไฟล์โมเดลตามเวอร์ชั่น และปรับแต่งเสียงร้องอัจฉริยะ (TH / EN)",
            font=("Segoe UI", 12, "bold"),
            justify=tk.CENTER
        )
        header_lbl.pack(pady=(0, 12))

        # ส่วนที่ 1: ระบบจัดการที่อยู่ไฟล์โมเดล (Model Path & Version Routing)
        route_group = ttk.LabelFrame(
            main_frame,
            text=" 1. จัดระเบียบที่อยู่ไฟล์โมเดลตามเวอร์ชั่น (Model Path & Version Configuration) ",
            padding=10
        )
        route_group.pack(fill=tk.X, pady=6)

        ttk.Label(route_group, text="ที่อยู่ไฟล์/แฟ้ม (Target Path / Code Input):").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.path_entry_var = tk.StringVar()
        path_entry = ttk.Entry(route_group, textvariable=self.path_entry_var, width=65)
        path_entry.grid(row=0, column=1, padx=6, pady=4, sticky=tk.EW)

        browse_btn = ttk.Button(route_group, text="เลือกแฟ้ม/ไฟล์ (Browse)", command=self.browse_target)
        browse_btn.grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(route_group, text="เวอร์ชั่นโมเดล (Model Version):").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.version_var = tk.StringVar(value="Auto")
        version_combo = ttk.Combobox(
            route_group,
            textvariable=self.version_var,
            values=["Auto", "RVC_v1", "RVC_v2", "DiffSinger_v1", "DiffSinger_v2"],
            state="readonly",
            width=25
        )
        version_combo.grid(row=1, column=1, sticky=tk.W, padx=6, pady=4)

        self.copy_mode_var = tk.BooleanVar(value=True)
        copy_chk = ttk.Checkbutton(
            route_group,
            text="คัดลอกไฟล์เข้าสู่โครงสร้างแฟ้มมาตรฐานอัตโนมัติ (Copy to Standard Version Folders)",
            variable=self.copy_mode_var
        )
        copy_chk.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=4)

        route_btn = ttk.Button(
            route_group,
            text="ดำเนินการจัดที่อยู่โมเดล (Execute Routing)",
            command=self.execute_model_routing
        )
        route_btn.grid(row=2, column=2, padx=4, pady=4, sticky=tk.EW)
        route_group.columnconfigure(1, weight=1)

        # ส่วนที่ 2: ระบบแปลงหน่วยเสียงและปรับแต่งเสียงร้อง (Phonemizer & Vocal Tuning)
        tune_group = ttk.LabelFrame(
            main_frame,
            text=" 2. ระบบแปลงหน่วยเสียงภาษาไทยและปรับแต่งเสียงร้อง (Thai Phonemizer & Automated Vocal Tuning) ",
            padding=10
        )
        tune_group.pack(fill=tk.X, pady=6)

        ttk.Label(tune_group, text="มาตรฐานหน่วยเสียง (Phoneme Standard):").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.standard_var = tk.StringVar(value="DiffSinger_AI")
        std_combo = ttk.Combobox(
            tune_group,
            textvariable=self.standard_var,
            values=["DiffSinger_AI", "Arpasing", "VCCV"],
            state="readonly",
            width=25
        )
        std_combo.grid(row=0, column=1, sticky=tk.W, padx=6, pady=4)

        ttk.Label(tune_group, text="อัตราขยายพยัญชนะ (Consonant Expansion):").grid(row=0, column=2, sticky=tk.W, padx=6, pady=4)
        self.expansion_var = tk.DoubleVar(value=1.35)
        exp_spin = ttk.Spinbox(tune_group, from_=1.0, to=2.5, increment=0.05, textvariable=self.expansion_var, width=10)
        exp_spin.grid(row=0, column=3, sticky=tk.W, pady=4)

        ttk.Label(tune_group, text="ทดสอบพยางค์ (พยัญชนะ | สระ | ตัวสะกด):").grid(row=1, column=0, sticky=tk.W, pady=4)
        syl_frame = ttk.Frame(tune_group)
        syl_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=6, pady=4)

        self.c_var = tk.StringVar(value="ก")
        self.v_var = tk.StringVar(value="อา")
        self.e_var = tk.StringVar(value="น")
        ttk.Entry(syl_frame, textvariable=self.c_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Entry(syl_frame, textvariable=self.v_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Entry(syl_frame, textvariable=self.e_var, width=8).pack(side=tk.LEFT, padx=2)

        test_tune_btn = ttk.Button(
            tune_group,
            text="ประมวลผลหน่วยเสียงและลูกคอ (Synthesize Phoneme & Pitch)",
            command=self.execute_phoneme_and_tuning
        )
        test_tune_btn.grid(row=1, column=3, padx=4, pady=4, sticky=tk.EW)

        # ส่วนที่ 3: หน้าต่างแสดงผลการทำงาน (Bilingual System Console Log)
        log_group = ttk.LabelFrame(
            main_frame,
            text=" 3. รายงานผลการประมวลผลของระบบ (System Execution Log - TH/EN) ",
            padding=10
        )
        log_group.pack(fill=tk.BOTH, expand=True, pady=6)

        self.log_text = tk.Text(log_group, height=14, font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        credit_lbl = ttk.Label(
            main_frame,
            text="Made And Checked By DELTA SYNTH & All Agentic AI | Original by RVC & DiffSinger Community",
            font=("Segoe UI", 9, "italic")
        )
        credit_lbl.pack(pady=(6, 0))

    def log_message(self, msg_th: str, msg_en: str) -> None:
        self.log_text.insert(tk.END, f"[TH] {msg_th}\n[EN] {msg_en}\n" + ("-" * 78) + "\n")
        self.log_text.see(tk.END)

    def browse_target(self) -> None:
        selected_dir = filedialog.askdirectory(title="เลือกแฟ้มโมเดลเป้าหมาย (Select Target Model Directory)")
        if selected_dir:
            self.path_entry_var.set(selected_dir)

    def execute_model_routing(self) -> None:
        target_path = self.path_entry_var.get().strip()
        if not target_path:
            messagebox.showwarning(
                "แจ้งเตือน / Warning",
                "กรุณาระบุที่อยู่ไฟล์หรือแฟ้มโมเดลก่อนดำเนินการ\nPlease enter a valid model file or folder path."
            )
            return

        result = self.model_manager.route_and_organize_model(
            source_path_str=target_path,
            forced_version=self.version_var.get(),
            copy_files=self.copy_mode_var.get()
        )
        self.log_message(result["message_th"], result["message_en"])
        for item in result.get("routed_files", []):
            self.log_text.insert(
                tk.END,
                f"  -> [{item['version']}] {item['file_name']} => {item['destination']}\n"
            )
        self.log_text.see(tk.END)

    def execute_phoneme_and_tuning(self) -> None:
        self.tuning_engine.consonant_expansion_ratio = float(self.expansion_var.get())
        res = self.tuning_engine.convert_syllable(
            consonant_th=self.c_var.get().strip(),
            vowel_th=self.v_var.get().strip(),
            ending_th=self.e_var.get().strip(),
            standard=self.standard_var.get()
        )
        pitch_preview = self.tuning_engine.generate_pitch_and_vibrato_curve(
            base_midi_note=60, duration_sec=0.5, sample_rate_hz=20
        )
        msg_th = (
            f"มาตรฐาน: {res['standard']} | หน่วยเสียง: [{res['phonemes']}] | "
            f"ความยาวพยัญชนะ: {res['consonant_duration_ms']} ms (ขยายเพิ่ม {res['expansion_applied']})"
        )
        msg_en = (
            f"Standard: {res['standard']} | Phonemes: [{res['phonemes']}] | "
            f"Consonant Duration: {res['consonant_duration_ms']} ms (Expanded {res['expansion_applied']})\n"
            f"Automated Pitch & Vibrato Curve Sample: {pitch_preview[:8]}..."
        )
        self.log_message(msg_th, msg_en)


# ==============================================================================
# จุดเริ่มต้นการทำงานรองรับทั้งโหมด CLI และ GUI (Dual CLI & GUI Entry Point)
# ==============================================================================
def main() -> None:
    if len(sys.argv) > 1:
        # โหมดรับค่าจากไฟล์ Batch หรือ Command Line (CLI Batch Processing Mode)
        target_input = sys.argv[1]
        version_override = sys.argv[2] if len(sys.argv) > 2 else "Auto"

        manager = ModelVersionManager()
        outcome = manager.route_and_organize_model(
            source_path_str=target_input,
            forced_version=version_override,
            copy_files=True
        )
        print("=" * 75)
        print(f"[TH] {outcome['message_th']}")
        print(f"[EN] {outcome['message_en']}")
        for rec in outcome.get("routed_files", []):
            print(f"  * [{rec['version']}] {rec['file_name']} -> {rec['destination']}")
        print("=" * 75)
    else:
        # โหมดเปิดหน้าต่างโปรแกรมกราฟิก (Interactive GUI Mode)
        root = tk.Tk()
        app = UnifiedVocalStudioApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()