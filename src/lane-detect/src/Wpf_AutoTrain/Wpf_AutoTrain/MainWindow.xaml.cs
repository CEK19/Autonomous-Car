using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Threading;
using System.Diagnostics;
using System.IO;

namespace Wpf_AutoTrain
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public Process process;
        public MainWindow()
        {
            InitializeComponent();
        }
        public void start_pythonFile()
        {
            label.Content = "Starting...";

            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = "python";
            startInfo.Arguments = "d:/NDT/PY_ws/DCLV/src/auto_train.py";
            startInfo.UseShellExecute = false;
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardInput = true;
            startInfo.CreateNoWindow = true;

            process = new Process();
            process.StartInfo = startInfo;
            process.OutputDataReceived += new DataReceivedEventHandler(process_OutputDataReceived);
            process.Start();
            process.BeginOutputReadLine();
        }

        private void process_OutputDataReceived(object sender, DataReceivedEventArgs e)
        {
            if (e.Data != null)
            {
                string output = e.Data;
                if (output.Trim() == "DONE")
                {
                    Thread.Sleep(10000);
                    this.Dispatcher.Invoke(() =>
                    {
                        int num = int.Parse(runLoop.Content.ToString());
                        num += 1;
                        runLoop.Content = num.ToString();
                        start_pythonFile();
                    });
                }
                else
                {
                    
                    this.Dispatcher.Invoke(() =>
                    {
                        label.Content = output.Trim();
                        if (label.Content.ToString().Contains("Epoch"))
                        {
                            epoch.Content = output.Trim();
                        }
                    });
                    
                }
            }
        }

        private void button_Click(object sender, RoutedEventArgs e)
        {
            start_pythonFile();
        }
    }
}
