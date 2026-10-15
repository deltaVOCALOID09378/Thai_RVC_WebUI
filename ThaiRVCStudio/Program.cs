// ==============================================================================
// Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
// Original by Patiphat Wongyai (Delta) & RVC Project
// File: Program.cs | Version: 1.0
// Description: [TH] จุดเริ่มต้นการทำงานหลักของโปรแกรม Desktop C# Thai RVC Studio
//              [EN] Main Entry Point for Thai RVC Studio C# WPF Application
// ==============================================================================
using System;
using System.Windows;

namespace ThaiRVCStudio;

public static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        var app = new App();
        app.InitializeComponent();
        app.Run(new MainWindow());
    }
}
