using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;
using System.IO;

// Version 2

namespace Wpf_EboxVisionSimulation
{
    public class RunPy
    {
        public string backendLocation;
        public string backendName;
        public string imageName;
        public string visionData;
        public Func<string, int> triggerDoneFunction;
        public Process process;





    }
}


