using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
            
            // Initialize weight with normal distribution, center 0, stddev 1/sqrt(x) 
            this._wih = Matrix<double>.Build.Dense(
                this._hnodescount,
                this._inodescount,
                Normal.WithMeanStdDev(0.0, 1 / Math.Sqrt(this._hnodescount)).Sample()
                );
            this._who = Matrix<double>.Build.Dense(
                this._onodescount,
                this._hnodescount,
                Normal.WithMeanStdDev(0.0, 1 / Math.Sqrt(this._onodescount)).Sample()
                );
        }

        /// <summary>
        /// Activation function of nodes.
        /// </summary>
        /// <param name="input">Input value of node (sum of all weighted inputs).</param>
        /// <returns>Output of the neuron.</returns>
        protected double activateNeuron(double input)
        {
            return NeuralMath.getSigmoid(input);
        }
    }
}
