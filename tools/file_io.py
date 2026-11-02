# ==============================================================================
# File: file_io.py
# Path: tools\file_io.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
def read_text(path, errors="strict", newline=None):
    last_error = None
    for encoding in (None, "utf8", "gbk"):
        try:
            kwargs = {"errors": "strict", "newline": newline}
            if encoding is not None:
                kwargs["encoding"] = encoding
            with open(path, "r", **kwargs) as file:
                return file.read()
        except UnicodeDecodeError as error:
            last_error = error
    if errors != "strict":
        with open(path, "r", encoding="gbk", errors=errors, newline=newline) as file:
            return file.read()
    raise last_error
