// ==============================================================================
// Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
// Original by Patiphat Wongyai (Delta) & RVC Project
// File: Models/StudioModels.cs | Version: 1.0
// Description: [TH] โครงสร้างข้อมูลสำหรับพารามิเตอร์การทำงานใน Thai RVC Studio
// ==============================================================================
using System;
using System.Collections.Generic;

namespace ThaiRVCStudio.Models;

public class ConversionParameters
{
    public string ModelName { get; set; } = string.Empty;
    public string AudioInputPath { get; set; } = string.Empty;
    public int PitchShift { get; set; } = 0;
    public string F0Method { get; set; } = "rmvpe";
    public string IndexPath { get; set; } = string.Empty;
    public double IndexRate { get; set; } = 0.75;
    public int FilterRadius { get; set; } = 3;
    public int ResampleRate { get; set; } = 0;
    public double RmsMixRate { get; set; } = 0.25;
    public double ProtectRate { get; set; } = 0.33;
}

public class UVR5Parameters
{
    public string ModelName { get; set; } = "HP2-all-vocals";
    public string InputPath { get; set; } = string.Empty;
    public string VocalOutputDir { get; set; } = @"B:\Render file to UTAU";
    public string InstrumentalOutputDir { get; set; } = @"B:\Render file to UTAU";
    public int Aggressiveness { get; set; } = 10;
    public string OutputFormat { get; set; } = "wav";
}

public class TrainingParameters
{
    public string ExpName { get; set; } = "My_Voice_Model";
    public string TargetSampleRate { get; set; } = "40k";
    public string Version { get; set; } = "v2";
    public bool UseF0 { get; set; } = true;
    public string DatasetDir { get; set; } = @"B:\For OpenUtau VB Making";
    public int BatchSize { get; set; } = 8;
    public int TotalEpochs { get; set; } = 20;
    public int SaveEveryEpoch { get; set; } = 5;
    public string GpuIds { get; set; } = "0";
}

public enum LogLevel
{
    Info,
    Success,
    Warning,
    Error
}

public class LogItem
{
    public DateTime Timestamp { get; set; } = DateTime.Now;
    public LogLevel Level { get; set; } = LogLevel.Info;
    public string Message { get; set; } = string.Empty;

    public string DisplayText => $"[{Timestamp:HH:mm:ss}] {Message}";
}
