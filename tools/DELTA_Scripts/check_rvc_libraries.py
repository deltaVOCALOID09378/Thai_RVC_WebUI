# ==============================================================================
# File: check_rvc_libraries.py
# Path: check_rvc_libraries.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.2
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
"""
RVC WebUI library status checker for Windows and embedded Python runtimes.

Original by Patiphat Wongyai (Delta) & RVC Project

The checker is read-only. It does not install, upgrade, or remove packages.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class CheckResult:
    category: str
    name: str
    status: str
    detail: str


IMPORT_ALIASES: dict[str, str] = {
    "faiss-cpu": "faiss",
    "scikit-learn": "sklearn",
    "python-dotenv": "dotenv",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "opencv-python": "cv2",
    "ffmpeg-python": "ffmpeg",
    "jinja2": "jinja2",
    "markdown": "markdown",
    "cython": "Cython",
    "praat-parselmouth": "parselmouth",
    "werkzeug": "werkzeug",
    "uc-micro-py": "uc_micro",
    "absl-py": "absl",
    "hydra-core": "hydra",
}

CORE_IMPORTS: tuple[tuple[str, str], ...] = (
    (
        "fairseq core",
        "import fairseq; from fairseq.data.dictionary import Dictionary; "
        "from fairseq.modules import grad_multiply",
    ),
    ("torch", "import torch"),
    ("torch_directml", "import torch_directml"),
    ("gradio", "import gradio"),
    ("onnxruntime", "import onnxruntime"),
    ("audio stack", "import numpy, scipy, librosa, soundfile"),
    ("RVC support stack", "import faiss, sklearn, numba, llvmlite"),
)


def normalize_windows_path(value: str | os.PathLike[str]) -> Path:
    """Normalize quotes, environment variables, and trailing separators safely."""
    text = os.path.expandvars(os.fspath(value)).strip()
    text = text.strip('"').strip("'")
    text = text.rstrip("\\/")
    if not text:
        text = os.curdir
    return Path(text).expanduser().resolve()


def resolve_project_root(argv: list[str]) -> Path:
    if len(argv) > 1 and argv[1].strip():
        return normalize_windows_path(argv[1])
    return Path(__file__).resolve().parent


def parse_requirement_name(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    requirement_name = re.split(r"[<>=!~;\s\[]", stripped, maxsplit=1)[0].strip()
    marker = stripped.split(";", 1)[1].strip().lower() if ";" in stripped else ""
    current_platform = platform.system().lower()
    if "sys_platform == 'darwin'" in marker and current_platform != "darwin":
        return None
    if "sys_platform != 'darwin'" in marker and current_platform == "darwin":
        return None
    if not requirement_name or requirement_name.lower() == "aria2":
        return None
    import_name = IMPORT_ALIASES.get(
        requirement_name.lower(), requirement_name.replace("-", "_")
    )
    return requirement_name, import_name


def package_status(distribution: str, import_name: str) -> CheckResult:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return CheckResult("library", distribution, "FAIL", "Distribution is not installed")
    except Exception as exc:
        return CheckResult("library", distribution, "FAIL", f"Metadata error: {exc}")

    try:
        import_available = importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return CheckResult("library", distribution, "FAIL", f"Import probe failed: {exc}")
    if not import_available:
        return CheckResult(
            "library",
            distribution,
            "FAIL",
            f"Version {version}; module not found: {import_name}",
        )
    return CheckResult(
        "library",
        distribution,
        "PASS",
        f"Version {version}; module: {import_name}",
    )


def command_status(label: str, command: list[str], timeout: int = 120) -> CheckResult:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CheckResult("runtime", label, "FAIL", f"Command not found: {exc}")
    except subprocess.TimeoutExpired:
        return CheckResult("runtime", label, "FAIL", f"Command timed out after {timeout}s")
    except OSError as exc:
        return CheckResult("runtime", label, "FAIL", f"Command error: {exc}")

    output = (completed.stdout + completed.stderr).strip().replace("\n", " | ")
    if completed.returncode == 0:
        status = "PASS"
    elif label == "pip check":
        status = "WARN"
    else:
        status = "FAIL"
    return CheckResult("runtime", label, status, output or f"Exit code: {completed.returncode}")


def probe_core_import(label: str, code: str) -> CheckResult:
    return_code, output = run_python(code)
    return CheckResult(
        "core import",
        label,
        "PASS" if return_code == 0 else "FAIL",
        output or f"Exit code: {return_code}",
    )


def run_python(code: str) -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip().replace("\n", " | ")
    return completed.returncode, output


def collect_results() -> list[CheckResult]:
    results: list[CheckResult] = []
    requirements = PROJECT_ROOT / "requirements.txt"
    program_py = PROJECT_ROOT / "Program.py"
    webui = PROJECT_ROOT / "webui.py"
    runtime_python = PROJECT_ROOT / "runtime" / "python.exe"

    for label, path, required in (
        ("project root", PROJECT_ROOT, True),
        ("requirements.txt", requirements, True),
        ("Program.py", program_py, True),
        ("webui.py", webui, True),
        ("runtime\\python.exe", runtime_python, False),
    ):
        exists = path.exists()
        results.append(
            CheckResult(
                "project file",
                label,
                "PASS" if exists else ("FAIL" if required else "WARN"),
                str(path) if exists else f"Missing: {path}",
            )
        )

    results.append(command_status("Python version", [sys.executable, "--version"], 30))
    results.append(command_status("pip version", [sys.executable, "-m", "pip", "--version"], 30))

    if requirements.exists():
        seen: set[str] = set()
        for line in requirements.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = parse_requirement_name(line)
            if parsed is None:
                continue
            distribution, import_name = parsed
            key = distribution.lower().replace("_", "-")
            if key in seen:
                continue
            seen.add(key)
            results.append(package_status(distribution, import_name))

    for label, code in CORE_IMPORTS:
        results.append(probe_core_import(label, code))

    cuda_code = "import torch; print(f'cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}')"
    results.append(probe_core_import("PyTorch CUDA probe", cuda_code))
    results.append(command_status("pip check", [sys.executable, "-m", "pip", "check"], 120))
    return results


def write_reports(results: Iterable[CheckResult]) -> tuple[Path, Path, str, int, int]:
    result_list = list(results)
    failures = sum(item.status == "FAIL" for item in result_list)
    warnings = sum(item.status == "WARN" for item in result_list)
    overall = "FAIL" if failures else ("WARN" if warnings else "PASS")
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    markdown_path = log_dir / f"RVC_Library_Status_{stamp}.md"
    json_path = log_dir / f"RVC_Library_Status_{stamp}.json"

    payload = {
        "generated": generated,
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "overall": overall,
        "failures": failures,
        "warnings": warnings,
        "checks": [asdict(item) for item in result_list],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# RVC WebUI Library Status Report",
        "",
        f"- Generated: `{generated}`",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- Python: `{sys.executable}`",
        f"- Overall: **{overall}**",
        f"- Failures: **{failures}**",
        f"- Warnings: **{warnings}**",
        "",
        "| Category | Check | Status | Detail |",
        "|---|---|---|---|",
    ]
    for item in result_list:
        detail = item.detail.replace("|", "\\|")
        lines.append(f"| {item.category} | {item.name} | **{item.status}** | {detail} |")
    lines.extend(
        [
            "",
            f"Raw JSON report: `{json_path}`",
            "",
            "Original by Patiphat Wongyai (Delta) & RVC Project",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path, json_path, overall, failures, warnings


def main() -> int:
    global PROJECT_ROOT
    PROJECT_ROOT = resolve_project_root(sys.argv)
    print("DELTA SYNTH - RVC WebUI Library Status Checker")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print()
    results = collect_results()
    markdown_path, json_path, overall, failures, warnings = write_reports(results)
    for item in results:
        print(f"[{item.status}] {item.category}: {item.name} - {item.detail}")
    print()
    print(f"Overall: {overall}; failures={failures}; warnings={warnings}")
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
