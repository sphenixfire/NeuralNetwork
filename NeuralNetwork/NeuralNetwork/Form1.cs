using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            // for testing
            Random rnd = new Random(DateTime.Now.Millisecond);
            // Setup the network
            var neuralnet = new NeuralNetwork(5, 5, 5, 0.4);
            // get input vector/matrix from network in proper size
            NeuralInput input = neuralnet.getNeuralInputContainer();
            // Fill with testdata
            for(int i=0;i < input.nodecount; i++)
            {
                input.setNode(i, ((double)rnd.Next(1, 999)/1000.0));
            }

        }
    }
}
