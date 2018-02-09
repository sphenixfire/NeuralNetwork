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
            var neuralnet = new NeuralNetwork(5, 4, 3, 0.3);
            // get input vector/matrix from network in proper size
            NeuralInOutput input = neuralnet.getNeuralInputContainer();
            // Fill with testdata
            for(int i=0;i < input.nodecount; i++)
            {
                input.setNode(i, ((double)rnd.Next(1, 999)/1000.0));
            }
            // Query the network
            NeuralInOutput output = neuralnet.queryNetwork(input);
        }
    }
}
