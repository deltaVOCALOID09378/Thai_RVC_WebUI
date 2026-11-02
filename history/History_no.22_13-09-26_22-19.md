# [Files Changed]
- `A:\Thai_RVC_WebUI_v1\DiffSinger_Training\*`
- `A:\Thai_RVC_WebUI_v1\Program.py`

# [Logic Altered]
- **DiffSinger AI Completeness & Upgrade**:
  - Investigated the `DiffSinger_Training` directory and found that all core Python scripts (`train.py`, `export.py`, etc.) were completely missing. Re-cloned the full repository from `openvpi/DiffSinger` and restored all necessary components.
  - Installed missing critical dependencies in the `runtime` environment (e.g., `pytorch_lightning`, `onnx`, `onnxsim`, `pyworld`, `tensorboard`, `praat-parselmouth`).
  - Added new tabs and options to `Program.py` under the "Diffsinger AI" category:
    1. **Config Editor (แก้ไขคอนฟิก)**: A direct YAML text editor allowing the user to select `configs/acoustic.yaml` or `configs/variance.yaml`, edit them in the UI, and save them.
    2. **TensorBoard (เครื่องมือตรวจสอบการฝึกสอน)**: A button to launch the TensorBoard daemon on port 6006, pointing to `DiffSinger_Training/checkpoints`.
    3. **Export ONNX (ส่งออกโมเดล)**: A dedicated section to export trained Acoustic/Variance PyTorch checkpoints to `.onnx` for use in OpenUtau, directly referencing the native `scripts/export.py`.

# [Performance / Stability Impact]
- WebUI is now fully functional for end-to-end DiffSinger workflows (Data Prep -> Acoustic Train -> Variance Train -> Export).
- No background memory overhead unless the user actively starts TensorBoard or Training.

# [Verification Method]
- Verified that `scripts/export.py` and `tensorboard` launch commands reference the correct paths.
- Verified DiffSinger clone integrity.
- Verified that `Program.py` successfully boots with the new tabs.

# [Compatibility]
- Compatible with OpenUtau DiffSinger standards (which natively use ONNX models exported from OpenVPI).

# [Residual Risks]
- none known.
