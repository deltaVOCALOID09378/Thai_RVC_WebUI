// ==============================================================================
// Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
// Original by Patiphat Wongyai (Delta) & RVC Project
// File: Services/PythonEngineService.cs | Version: 1.0
// Description: [TH] บริการรันและควบคุมการทำงานของ Python Engine แบบอะซิงโครนัส
//              [EN] Asynchronous Python Engine Process Controller & Streamer
// ==============================================================================
using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ThaiRVCStudio.Services;

public class PythonEngineService : IDisposable
{
    private readonly string _pythonExe;
    private readonly string _projectRoot;
    private Process? _serverProcess;
    private Process? _activeTaskProcess;
    private readonly HttpClient _httpClient;
    private bool _disposed;

    public event Action<string, bool>? OutputReceived; // (message, isError)
    public event Action<bool>? ServerStateChanged;      // (isOnline)

    public PythonEngineService(string projectRoot)
    {
        _projectRoot = projectRoot;
        string candidateRuntime = Path.Combine(_projectRoot, "runtime", "python.exe");
        _pythonExe = File.Exists(candidateRuntime) ? candidateRuntime : "python";
        _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
    }

    public string PythonPath => _pythonExe;
    public bool IsServerRunning => _serverProcess != null && !_serverProcess.HasExited;

    public async Task<bool> CheckServerHealthAsync(int port = 9378)
    {
        try
        {
            var res = await _httpClient.GetAsync($"http://127.0.0.1:{port}/");
            return res.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public void StartWebUIServer(int port = 9378)
    {
        if (IsServerRunning)
            return;

        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = $"Program.py --port {port}",
            WorkingDirectory = _projectRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };

        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0";

        _serverProcess = new Process { StartInfo = psi, EnableRaisingEvents = true };

        _serverProcess.OutputDataReceived += (s, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
                OutputReceived?.Invoke(e.Data, false);
        };

        _serverProcess.ErrorDataReceived += (s, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
                OutputReceived?.Invoke(e.Data, true);
        };

        _serverProcess.Exited += (s, e) =>
        {
            ServerStateChanged?.Invoke(false);
            OutputReceived?.Invoke("🛑 [SYSTEM] เซิร์ฟเวอร์ Python WebUI สิ้นสุดการทำงานแล้ว (WebUI Process Exited)", false);
        };

        _serverProcess.Start();
        _serverProcess.BeginOutputReadLine();
        _serverProcess.BeginErrorReadLine();

        ServerStateChanged?.Invoke(true);
        OutputReceived?.Invoke($"🚀 [SYSTEM] เริ่มต้นการทำงาน WebUI Engine บนพอร์ต {port} เรียบร้อยแล้ว", false);
    }

    public void StopWebUIServer()
    {
        if (_serverProcess != null && !_serverProcess.HasExited)
        {
            try
            {
                _serverProcess.Kill(entireProcessTree: true);
                _serverProcess.Dispose();
            }
            catch { }
            finally
            {
                _serverProcess = null;
                ServerStateChanged?.Invoke(false);
                OutputReceived?.Invoke("⏹️ [SYSTEM] หยุดการทำงานของเซิร์ฟเวอร์เรียบร้อยแล้ว (Server Stopped)", false);
            }
        }
    }

    public async Task<int> RunScriptAsync(string relativeScriptPath, string arguments, CancellationToken cancellationToken = default)
    {
        string fullScriptPath = Path.Combine(_projectRoot, relativeScriptPath);
        if (!File.Exists(fullScriptPath))
        {
            OutputReceived?.Invoke($"🔴 [ERROR] ไม่พบไฟล์สคริปต์ (Script Not Found): {fullScriptPath}", true);
            return -1;
        }

        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            Arguments = $"\"{fullScriptPath}\" {arguments}",
            WorkingDirectory = _projectRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };

        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

        using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _activeTaskProcess = process;

        process.OutputDataReceived += (s, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
                OutputReceived?.Invoke(e.Data, false);
        };

        process.ErrorDataReceived += (s, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
                OutputReceived?.Invoke(e.Data, true);
        };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        using (cancellationToken.Register(() =>
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                    OutputReceived?.Invoke("🛑 [CANCEL] คำสั่งถูกยกเลิกโดยผู้ใช้ (Process Terminated by User)", true);
                }
            }
            catch { }
        }))
        {
            await process.WaitForExitAsync(cancellationToken);
            _activeTaskProcess = null;
            return process.ExitCode;
        }
    }

    public void CancelActiveTask()
    {
        if (_activeTaskProcess != null && !_activeTaskProcess.HasExited)
        {
            try
            {
                _activeTaskProcess.Kill(entireProcessTree: true);
            }
            catch { }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            StopWebUIServer();
            CancelActiveTask();
            _httpClient.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
