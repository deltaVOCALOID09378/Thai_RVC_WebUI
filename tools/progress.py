# -*- coding: utf-8 -*-
# ==============================================================================
# File: progress.py
# Path: tools\progress.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 2.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
# Description: Studio Batch Progress Tracking and Status Formatter (Thai & English)

import math
from typing import List, Optional


def should_report(index: int, total: int, max_updates: int = 12) -> bool:
    """
    คำนวณจังหวะการอัปเดตสถานะเพื่อป้องกัน UI ค้างหรือแล็ก (Throttle progress updates)
    """
    if total <= 0:
        return False
    if total <= max_updates:
        return True
    interval = max(1, math.ceil(total / max_updates))
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def batch_status(
    title: str,
    current: int,
    total: int,
    success: int,
    failed: int,
    latest: str = "",
    failures: Optional[List[str]] = None,
) -> str:
    """
    จัดรูปแบบข้อความรายงานสถานะแบบเรียลไทม์ พร้อมอีโมจิและการแจกแจงรายละเอียด
    (Format real-time batch progress report with emojis and detailed metrics)
    """
    if total <= 0:
        state = "⏳ รอข้อมูลนำเข้า (Waiting for Input)"
    elif current >= total:
        state = "🎉 ดำเนินการเสร็จสมบูรณ์ 100% (Completed)"
    else:
        pct = (current / total * 100.0) if total > 0 else 0.0
        state = f"🚀 กำลังประมวลผล (In Progress: {pct:.1f}%)"

    lines = [
        f"【📌 {title}】",
        f"📊 สถานะ (Status): {state}",
        f"📈 ความคืบหน้า (Progress): {current}/{total} | ✅ สำเร็จ (Success): {success} | ❌ ล้มเหลว (Failed): {failed}",
    ]
    if latest:
        lines.append(f"📄 รายการล่าสุด (Current Item): {latest}")
    if failures:
        lines.append("⚠️ รายการที่พบข้อผิดพลาด (Failure Logs):")
        lines.extend(failures[-10:])
        if len(failures) > 10:
            lines.append(f"... แสดงเฉพาะ {min(10, len(failures))} รายการล่าสุด")
    return "\n".join(lines)
