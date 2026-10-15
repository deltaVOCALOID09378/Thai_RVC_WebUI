# ==============================================================================
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by Patiphat Wongyai (Delta) & RVC Project
# File: configs/config.py | Version: 1.6
# Location: configs/config.py
# Description: Hardware and Runtime Environment Configuration Manager
# ==============================================================================
import argparse
import os
import sys
import json
import shutil
from multiprocessing import cpu_count
import torch
import logging

logger = logging.getLogger(__name__)

try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        ipex_init()
except Exception:
    pass

version_config_list = [
    "v1/32k.json", "v1/40k.json", "v1/48k.json",
    "v2/48k.json", "v2/32k.json",
]


def load_dotenv_safely(env_path: str = ".env") -> None:
    """
    Safely load environment variables from .env file with UTF-8 encoding.
    Adheres to python-development-python-configuration standard.
    """
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception as e:
            logger.warning(f"Could not load {env_path}: {e}")

    # Set guaranteed sensible defaults for system paths & network safety
    os.environ.setdefault("weight_root", "assets/weights")
    os.environ.setdefault("weight_uvr5_root", "assets/uvr5_weights")
    os.environ.setdefault("index_root", "logs")
    os.environ.setdefault("outside_index_root", "assets/indices")
    os.environ.setdefault("rmvpe_root", "assets/rmvpe")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if wrapper.instance is None:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance
    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        # Load typed settings and environment variables
        load_dotenv_safely()

        # Default fallback mode
        self.device = "cpu"
        self.is_half = False
        self.use_jit = False
        self.n_cpu = 0
        self.gpu_name = "CPU (Default Fallback Mode)"
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.dml,
        ) = self.arg_parse()
        self.instead = ""
        self.preprocess_per = 3.7

        # Safe assignment with guaranteed fallback
        dev_res = self.device_config()
        if dev_res is None or not isinstance(dev_res, (tuple, list)) or len(dev_res) != 4:
            logger.warning("[DELTA SYNTH] device_config returned invalid output. Applying default safe config.")
            self.x_pad, self.x_query, self.x_center, self.x_max = (1, 6, 38, 41)
        else:
            self.x_pad, self.x_query, self.x_center, self.x_max = dev_res

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            p = f"configs/inuse/{config_file}"
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                if os.path.exists(f"configs/{config_file}"):
                    shutil.copy(f"configs/{config_file}", p)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    d[config_file] = json.load(f)
        return d

    @staticmethod
    def arg_parse() -> tuple:
        exe = os.path.join(os.getcwd(), "runtime", "python.exe") if os.path.exists(os.path.join(os.getcwd(), "runtime", "python.exe")) else (sys.executable or "python")
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=9378, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument("--noparallel", action="store_true", help="Disable parallel processing")
        parser.add_argument("--noautoopen", action="store_true", help="Do not open in browser automatically")
        parser.add_argument("--dml", action="store_true", help="torch_dml")
        cmd_opts, _ = parser.parse_known_args()
        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 9378
        return (cmd_opts.pycmd, cmd_opts.port, cmd_opts.colab, cmd_opts.noparallel, cmd_opts.noautoopen, cmd_opts.dml)

    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @staticmethod
    def has_xpu() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def use_fp32_config(self):
        for config_file in version_config_list:
            try:
                if config_file in self.json_config and "train" in self.json_config[config_file]:
                    self.json_config[config_file]["train"]["fp16_run"] = False
                p = f"configs/inuse/{config_file}"
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        strr = f.read().replace("true", "false")
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(strr)
            except Exception:
                pass
        self.preprocess_per = 3.0
        logger.info("Overwritten preprocess_per to %d (FP32 Mode)" % (self.preprocess_per))

    def device_config(self) -> tuple:
        fallback_device = "cpu"
        self.device = fallback_device
        self.is_half = False
        self.use_fp32_config()
        self.gpu_name = "CPU Core (Default)"

        try:
            if self.dml:
                logger.info("DirectML requested: Attempting to hook DirectML device...")
                import_dml_success = False
                try:
                    import torch_directml
                    self.device = torch_directml.device(torch_directml.default_device())
                    self.gpu_name = "DirectML GPU"
                    self.is_half = False
                    import_dml_success = True
                except Exception as exc:
                    logger.warning(f"Failed to initialize torch_directml: {exc}")
                if not import_dml_success:
                    self.device = "cpu"
                    self.is_half = False
            elif torch.cuda.is_available():
                if self.has_xpu():
                    self.device = self.instead = "xpu:0"
                    self.is_half = True
                else:
                    self.device = "cuda:0"
                    self.is_half = True
                i_device = int(self.device.split(":")[-1]) if ":" in self.device else 0
                self.gpu_name = torch.cuda.get_device_name(i_device)
                self.gpu_mem = int(torch.cuda.get_device_properties(i_device).total_memory / 1024 / 1024 / 1024 + 0.4)

                if any(x in self.gpu_name.upper() for x in ["P40", "P10", "1060", "1070", "1080"]) or ("16" in self.gpu_name and "V100" not in self.gpu_name.upper()):
                    logger.info(f"Found Legacy GPU {self.gpu_name}, forcing FP32.")
                    self.is_half = False
                    self.use_fp32_config()
                if self.gpu_mem is not None and self.gpu_mem <= 4:
                    self.preprocess_per = 3.0
            elif self.has_mps():
                logger.info("Apple MPS detected.")
                self.device = self.instead = "mps"
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("No GPU detected, using CPU mode.")
                self.device = self.instead = "cpu"
                self.is_half = False
                self.use_fp32_config()
        except Exception as e:
            logger.error(f"Error in hardware detection: {e}. Falling back to CPU safely.")
            self.device = fallback_device
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32

        logger.info(f"Hardware Initialized: {self.device} ({self.gpu_name}) | Half-Precision: {self.is_half}")
        return (x_pad, x_query, x_center, x_max)