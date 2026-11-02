# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 30/10/2569 (30 October 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)   : F0Predictor.py
# 3. ลำดับเวอร์ชั่นของไฟล์ (Version): 1.2
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              [ Original by "RVC-Project" ]
# 5. ตำแหน่งที่วางของไฟล์ (Path) : infer/lib/infer_pack/modules/F0Predictor/F0Predictor.py
# 6. รายละเอียดการแก้ไขงาน    : ปรับสมดุลโครงสร้างคลาสพื้นฐาน (Base Class) และจัดระเบียบ Docstring
# ==============================================================================

class F0Predictor(object):
    def compute_f0(self, wav, p_len):
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length]
        """
        pass

    def compute_f0_uv(self, wav, p_len):
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        """
        pass