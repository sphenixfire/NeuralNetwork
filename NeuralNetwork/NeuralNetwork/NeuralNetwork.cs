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
        private int _inodescount;
        private int _hnodescount;
        private int _onodescount;
        private double _learnrate;
        private MathNet.Numerics.LinearAlgebra.Matrix<double> _wih;
        private Matrix _who;
        

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


    }
}
