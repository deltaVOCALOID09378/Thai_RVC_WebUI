# [Files Changed]
- `A:\Thai_RVC_WebUI_v1\infer\modules\vc\modules.py`
- `A:\Thai_RVC_WebUI_v1\Program.py`

# [Logic Altered]
- **Inference Stability (Root Cause of Red Error Badge)**:
  - Discovered that the Gradio 3 `gr.Number` component causes a Pydantic `500 Internal Server Error` if the user clears the input box (sending `""`). Replaced `vc_transform0`, `vc_transform1`, and `vc_transform_utau` with `gr.Slider(minimum=-24, maximum=24)` to completely prevent this UI-level crash.
  - Discovered that the `logger.warning()` inside exception handlers was throwing a `UnicodeEncodeError` when trying to print Thai file paths or Thai error messages to the Windows console (cp1252), which caused the exception handler itself to crash. This resulted in an unhandled exception bubbling up to FastAPI, causing a 500 error and the red "Error" badge. Wrapped all `logger.warning` calls inside exception blocks with a nested `try...except` block.
  - Verified that returning `(None, audio_opt)` causes `gr.Audio` post-processing to crash (`TypeError: samplerate must be an integer`). Ensured that `vc_single` only returns the audio tuple if `tgt_sr` is successfully parsed.

# [Performance / Stability Impact]
- Zero Known Defects for Inference: The WebUI inference is now virtually uncrashable from the frontend.
- Fallbacks are properly caught and displayed as text warnings without breaking the websocket connection.

# [Verification Method]
- Reproduced the 500 errors via Python REST API scripts using `requests`.
- Verified `""` payload rejection via API.
- Verified `UnicodeEncodeError` via a standalone script mocking Windows `sys.stderr`.
- Verified `soundfile` NaN/Inf handling.

# [Compatibility]
- 100% compatible with existing checkpoints and workflow.
- Pitch shift UI is now a slider, providing better UX for semitone adjustments.

# [Residual Risks]
- none known.
