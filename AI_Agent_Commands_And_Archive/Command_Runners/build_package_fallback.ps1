# ==============================================================================
# File Name   : build_package_fallback.ps1
# Version     : v1.0
# Location    : AI_Agent_Commands_And_Archive\Command_Runners\build_package_fallback.ps1
# Function    : High-speed PowerShell Clean Distribution Packager (Fallback Engine)
# Credit      : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#               Original by DELTA SYNTH
# ==============================================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [Parameter(Mandatory = $true)]
    [string]$TargetZip
)

$ErrorActionPreference = "Stop"

try {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem

    $skips = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    @(
        'runtime', 'venv', '_local_env', '.git', '.github', '.claude', 
        '.cursor', '.hermes', '.gemini', '.tmp.driveupload', 'Published', 
        'TEMP', 'logs', 'weights', 'pretrained', 'pretrained_v2', 
        'uvr5_weights', 'rmvpe', 'DiffSinger_Training', 'Export File', 
        'For OpenUtau VB Making', 'RVCRealtimeVST', '__pycache__'
    ) | ForEach-Object { $skips.Add($_) | Out-Null }

    $skipExts = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    @('.pyc', '.zip', '.tmp', '.log') | ForEach-Object { $skipExts.Add($_) | Out-Null }

    $targetParent = [System.IO.Path]::GetDirectoryName($TargetZip)
    if (-not [System.IO.Directory]::Exists($targetParent)) {
        [System.IO.Directory]::CreateDirectory($targetParent) | Out-Null
    }
    if ([System.IO.File]::Exists($TargetZip)) {
        [System.IO.File]::Delete($TargetZip)
    }

    $fileList = [System.Collections.Generic.List[string]]::new()
    function ScanDir($dir) {
        foreach ($entry in [System.IO.Directory]::EnumerateFileSystemEntries($dir)) {
            $name = [System.IO.Path]::GetFileName($entry)
            if ([System.IO.Directory]::Exists($entry)) {
                if (-not $skips.Contains($name)) { ScanDir $entry }
            } else {
                $ext = [System.IO.Path]::GetExtension($entry)
                if (-not $skipExts.Contains($ext) -and $entry -ne $TargetZip) {
                    $fileList.Add($entry)
                }
            }
        }
    }

    Write-Host "[INFO] Scanning workspace directory structure..."
    ScanDir $SourceDir
    Write-Host "[INFO] Discovered $($fileList.Count) clean distribution files."

    Write-Host "[INFO] Compressing archive entries..."
    $zipArchive = [System.IO.Compression.ZipFile]::Open($TargetZip, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($f in $fileList) {
            $rel = $f.Substring($SourceDir.Length).TrimStart('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zipArchive, 
                $f, 
                $rel, 
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $zipArchive.Dispose()
    }

    $zipSizeMB = [math]::Round(((New-Object System.IO.FileInfo($TargetZip)).Length / 1MB), 2)
    Write-Host "[SUCCESS] Package archive generated: $TargetZip ($zipSizeMB MB)"
    exit 0
}
catch {
    Write-Error "[FATAL] Fallback packaging failed: $_"
    exit 1
}
