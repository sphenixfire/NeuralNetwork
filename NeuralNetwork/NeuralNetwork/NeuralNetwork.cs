using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;


namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        // Node counts in input-, hidden-, outputlayer
        private int _inodescount;
        private int _hnodescount;
        private int _onodescount;
        // Learnrate of the network
        private double _learnrate;
        // Weight matrix between i-nput and h-idden layer and between h-idden and o-utputlayer
        private Matrix<double> _wih;
        private Matrix<double> _who;
        

        /// <summary>
        /// Initializes the Neural Network.
        /// </summary>
        /// <param name="inodescount">Number of Inputnodes</param>
        /// <param name="hnodescount">Number of Hiddennodes</param>
        /// <param name="onodescount">Number of Outputnodes</param>
        /// <param name="learnreate">Learnrate of the network</param>
        public NeuralNetwork(int inodescount, int hnodescount, int onodescount, double learnreate)
        {
            this._inodescount = inodescount;
            this._hnodescount = hnodescount;
            this._onodescount = onodescount;
            this._learnrate = learnreate;
            
            // Generates probability values based on normal distribution
            Normal norgen = Normal.WithMeanStdDev(0.0, 1 / Math.Sqrt(this._hnodescount), new Random(DateTime.Now.Millisecond * DateTime.Now.Minute));

            // Initialize weight with normal distribution, center 0, stddev 1/sqrt(x) 
            this._wih = Matrix<double>.Build.Dense(
                this._hnodescount,
                this._inodescount
                );
            for(int i=0;i < this._wih.RowCount; i++)
            {
                for (int j = 0; j < this._wih.ColumnCount; j++)
                {
                    this._wih[i, j] = norgen.Sample();
                }
            }
            this._who = Matrix<double>.Build.Dense(
                this._onodescount,
                this._hnodescount
                );
            for (int i = 0; i < this._who.RowCount; i++)
            {
                for (int j = 0; j < this._who.ColumnCount; j++)
                {
                    this._who[i, j] = norgen.Sample();
                }
            }
        }

        /// <summary>
        /// Creates an Input-List based on neural network nodecount
        /// </summary>
        /// <returns>Arraylist in proper length</returns>
        public NeuralInOutput getNeuralInputContainer()
        {
            return new NeuralInOutput(this._inodescount);
        }

        /// <summary>
        /// Creates an output list based on the network by the network.
        /// </summary>
        /// <returns>The empty output list.</returns>
        protected NeuralInOutput getNeuralOutput()
        {
            return new NeuralInOutput(this._onodescount);
        }

        /// <summary>
        /// Applies activation function to a node layer.
        /// </summary>
        /// <param name="input">Input of the layer.</param>
        /// <returns>Output of the layer.</returns>
        protected Matrix<double> activateNeurons(Matrix<double> input)
        {
            for(int i=0;i < input.RowCount; i++)
            {
                input[i, 0] = NeuralMath.getSigmoid(input[i, 0]);
            }
            return input;
        }

        /// <summary>
        /// Queries the network with an input.
        /// </summary>
        /// <param name="input">The input for the network.</param>
        /// <returns>The output of the network.</returns>
        public NeuralInOutput queryNetwork(NeuralInOutput input)
        {
            // Converts container to Matrix
            Matrix<double> inputnodes = input.getNodelist();
            // Calculates the input of the hidden layer and applies the activation functions which results in output of hidden layer.
            Matrix<double> hiddennodesout = this.activateNeurons(this._wih.Multiply(inputnodes));
            // Calculates the input of the output layer and applies the activation functions which results in output of output layer.
            Matrix<double> outputnodesout = this.activateNeurons(this._who.Multiply(hiddennodesout));

            return new NeuralInOutput(outputnodesout);
        }
    }
}