// ==============================================================================
// Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
// Original by Patiphat Wongyai (Delta) & RVC Project
// File: Services/AudioScannerService.cs | Version: 1.0
// Description: [TH] บริการสแกนและค้นหาโมเดลเสียง อินเด็กซ์ และไฟล์ระบบ
// ==============================================================================
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ThaiRVCStudio.Services;

public class AudioScannerService
{
    private readonly string _projectRoot;

    public AudioScannerService(string? projectRoot = null)
    {
        _projectRoot = projectRoot ?? Directory.GetCurrentDirectory();
        // If running from bin/Debug/net10.0-windows, resolve up to project root if needed
        if (!Directory.Exists(Path.Combine(_projectRoot, "assets")) &&
            Directory.Exists(Path.Combine(_projectRoot, "..", "..", "..", "..", "assets")))
        {
            _projectRoot = Path.GetFullPath(Path.Combine(_projectRoot, "..", "..", "..", ".."));
        }
    }

    public string ProjectRoot => _projectRoot;

    public List<string> GetAvailableRvcModels()
    {
        var models = new List<string>();
        string weightsDir = Path.Combine(_projectRoot, "assets", "weights");
        if (Directory.Exists(weightsDir))
        {
            foreach (var file in Directory.GetFiles(weightsDir, "*.pth", SearchOption.TopDirectoryOnly))
            {
                models.Add(Path.GetFileName(file));
            }
        }
        return models.OrderBy(m => m).ToList();
    }

    public List<string> GetAvailableIndices()
    {
        var indices = new List<string>();
        
        string outsideIndices = Path.Combine(_projectRoot, "assets", "indices");
        if (Directory.Exists(outsideIndices))
        {
            foreach (var file in Directory.GetFiles(outsideIndices, "*.index", SearchOption.AllDirectories))
            {
                if (!file.Contains("trained"))
                    indices.Add(file);
            }
        }

        string logsDir = Path.Combine(_projectRoot, "logs");
        if (Directory.Exists(logsDir))
        {
            foreach (var file in Directory.GetFiles(logsDir, "*.index", SearchOption.AllDirectories))
            {
                if (!file.Contains("trained"))
                    indices.Add(file);
            }
        }

        return indices.Distinct().OrderBy(i => Path.GetFileName(i)).ToList();
    }

    public List<string> GetAvailableUvr5Models()
    {
        var models = new List<string>();
        string uvr5Dir = Path.Combine(_projectRoot, "assets", "uvr5_weights");
        if (Directory.Exists(uvr5Dir))
        {
            foreach (var file in Directory.GetFiles(uvr5Dir, "*.pth", SearchOption.TopDirectoryOnly))
            {
                models.Add(Path.GetFileNameWithoutExtension(file));
            }
            foreach (var file in Directory.GetFiles(uvr5Dir, "*.onnx", SearchOption.TopDirectoryOnly))
            {
                models.Add(Path.GetFileNameWithoutExtension(file));
            }
        }
        if (models.Count == 0)
        {
            models.AddRange(new[] { "HP2-all-vocals", "HP3-all-vocals", "HP5-only-vocals", "onnx_dereverb_By_FoxJoy" });
        }
        return models.Distinct().OrderBy(m => m).ToList();
    }
}
