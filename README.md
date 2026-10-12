# DELTA SYNTH - Smart AI Voice Studio

**Made And Checked By DELTA SYNTH & All Code Agentic AI Engine**
*Original by RVC-Project*

## 🎯 Project Architecture & File Mapping

The environment utilizes embedded Python execution securely constrained to `runtime\python.exe` and strictly operates on port `9378`. 

```text
📁 DELTA_RVC_Workspace/
 ├── 📄 Program.py                 (Main Core Execution File - Merged UI/API logic)
 ├── 📄 Start-The-Program.bat      (Launcher with Drag-and-Drop functionality)
 ├── 📄 Install-File.bat           (Environment Auto Installer)
 ├── 📄 requirements.txt           (Dependency lists for Pip Engine)
 ├── 📁 runtime/                   (Embedded Python Directory)
 │   └── 📄 python.exe             (Strict Interpreter Path)
 ├── 📁 DiffSinger_Training/       (Diffsinger Processing Directory)
 └── 📁 For OpenUtau VB Making/    (Voicebank Renderer Directory)
     └── 📁 A Voicebank File/
         ├── 📁 EN
         ├── 📁 TH
         └── 📁 JP