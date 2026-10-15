# ==============================================================================
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by Patiphat Wongyai (Delta) & RVC Project
# File: i18n/i18n.py | Version: 1.4
# Description: RVC WebUI Internationalization (i18n) Module
# ==============================================================================
import json
import locale
import os
from pathlib import Path


def load_language_list(language):
    locale_dir = Path(__file__).resolve().parent / "locale"
    file_path = locale_dir / f"{language}.json"
    if not file_path.exists():
        file_path = locale_dir / "th_TH.json"
        if not file_path.exists():
            file_path = locale_dir / "en_US.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            try:
                language = locale.getlocale()[0]
            except Exception:
                language = "th_TH"
        if not language:
            language = "th_TH"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return f"Use Language: {self.language}"
