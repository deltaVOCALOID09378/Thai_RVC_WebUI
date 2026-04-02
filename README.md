<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
เฟรมเวิร์กการแปลงเสียง (Voice Conversion) ที่ใช้งานง่ายและพัฒนาต่อยอดมาจาก VITS<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>
  
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**บันทึกการอัปเดต (Changelog)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_EN.md) | [**คำถามที่พบบ่อย (FAQ)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) 

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

</div>

> รับชม [วิดีโอสาธิตการใช้งาน (Demo Video)](https://www.bilibili.com/video/BV1pm4y1z7Gm/) ได้ที่นี่!

<table>
   <tr>
		<td align="center">WebUI สำหรับการฝึกสอน (Training) และอนุมาน (Inference)</td>
		<td align="center">GUI สำหรับการแปลงเสียงแบบเรียลไทม์ (Real-time)</td>
	</tr>
  <tr>
		<td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/092e5c12-0d49-4168-a590-0b0ef6a4f630"></td>
    <td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/730b4114-8805-44a1-ab1a-04668f3c30a6"></td>
	</tr>
	<tr>
		<td align="center">go-web.bat</td>
		<td align="center">go-realtime-gui.bat</td>
	</tr>
  <tr>
    <td align="center">คุณสามารถเลือกใช้งานโหมดต่างๆ ได้อย่างอิสระตามความต้องการ</td>
		<td align="center">เราสามารถทำความหน่วง (Latency) แบบ end-to-end ได้ที่ 170ms และหากใช้อุปกรณ์ I/O ที่รองรับ ASIO จะสามารถลดความหน่วงลงเหลือเพียง 90ms ได้ (แต่ทั้งนี้ขึ้นอยู่กับการรองรับของไดรเวอร์ฮาร์ดแวร์ด้วย)</td>
	</tr>
</table>

> ชุดข้อมูลที่ใช้สำหรับโมเดลพรีเทรน (Pre-training model) ใช้เสียงคุณภาพสูงความยาวเกือบ 50 ชั่วโมงจากชุดข้อมูลโอเพนซอร์ส VCTK

> เราจะมีการเพิ่มชุดข้อมูลเพลงคุณภาพสูงที่มีลิขสิทธิ์ถูกต้องลงในชุดฝึกสอน (Training-set) อย่างต่อเนื่อง เพื่อให้คุณสามารถนำไปใช้งานได้โดยไม่ต้องกังวลเรื่องการละเมิดลิขสิทธิ์

> โปรดติดตามโมเดลพรีเทรนพื้นฐานของ RVCv3 ที่กำลังจะมาถึง ซึ่งจะมีพารามิเตอร์ขนาดใหญ่ขึ้น, ข้อมูลฝึกสอนมากขึ้น, ผลลัพธ์ที่ดีกว่าเดิม, ความเร็วในการอนุมานคงที่ และใช้ข้อมูลในการฝึกสอนน้อยลง

## คุณสมบัติเด่น (Features):
+ ลดปัญหาเสียงหลุดโทน (Tone leakage) โดยการแทนที่ฟีเจอร์เสียงต้นฉบับด้วยฟีเจอร์จากชุดข้อมูลฝึกสอนโดยใช้การดึงข้อมูลแบบ Top1
+ การฝึกสอน (Training) ทำได้ง่ายและรวดเร็ว แม้จะใช้การ์ดจอที่สเปคไม่สูงมาก
+ รองรับการฝึกสอนด้วยข้อมูลจำนวนน้อย (แนะนำให้ใช้เสียงพูดที่มีสัญญาณรบกวนต่ำ ความยาวตั้งแต่ 10 นาทีขึ้นไป)
+ สามารถผสานโมเดล (Model fusion) เพื่อเปลี่ยนลักษณะน้ำเสียงได้ (ผ่านแท็บ ckpt processing -> ckpt merge)
+ หน้าต่างใช้งาน (WebUI) ที่เข้าใจง่ายและใช้งานสะดวก
+ มาพร้อมโมเดล UVR5 สำหรับแยกเสียงร้องและเสียงดนตรีได้อย่างรวดเร็ว
+ ใช้อัลกอริทึมสกัดระดับเสียงร้องโทนสูง [InterSpeech2023-RMVPE](#Credits) เพื่อป้องกันปัญหาเสียงหาย/เสียงใบ้ ให้ผลลัพธ์ที่ดีเยี่ยม (อย่างเห็นได้ชัด) และทำงานเร็วกว่าพร้อมกินทรัพยากรน้อยกว่า Crepe_full
+ รองรับการเร่งความเร็วผ่านการ์ดจอ AMD และ Intel
+ รองรับการเร่งความเร็วผ่านการ์ดจอ Intel ARC ด้วยเทคโนโลยี IPEX

## การเตรียมสภาพแวดล้อม (Preparing the environment)
คำสั่งต่อไปนี้จำเป็นต้องรันผ่าน Python 3.8 ขึ้นไป

(Windows/Linux)
เริ่มต้นด้วยการติดตั้งแพ็กเกจหลักผ่าน pip:
```bash
# ติดตั้งแพ็กเกจหลักที่เกี่ยวข้องกับ PyTorch (ข้ามได้หากติดตั้งแล้ว)
# อ้างอิง: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

# สำหรับ Windows + การ์ดจอสถาปัตยกรรม Nvidia Ampere (RTX30xx) คุณต้องระบุเวอร์ชัน CUDA ที่ตรงกับ PyTorch ตามคำแนะนำจาก [https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/21](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/21)
#pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)

# สำหรับ Linux + การ์ดจอ AMD ให้ใช้ PyTorch เวอร์ชันต่อไปนี้:
#pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/rocm5.4.2](https://download.pytorch.org/whl/rocm5.4.2)
