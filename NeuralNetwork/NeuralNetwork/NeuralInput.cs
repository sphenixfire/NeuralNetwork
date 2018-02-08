using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// Represents an input container for the neural network which has to be creates after the network has been set up.
    /// </summary>
    public class NeuralInput
    {
        private Matrix<double> _inputlist;

        /// <summary>
        /// Allows to create the inputlist only by Neural-Network function
        /// </summary>
        /// <param name="nodecount">Nomber of nodes to create</param>
        internal NeuralInput(int nodecount)
        {
            this._inputlist = Matrix<double>.Build.Dense(nodecount, 1, 0.01);
        }

        /// <summary>
        /// Sets an input node value.
        /// </summary>
        /// <param name="nodeid">The position in the input vector.</param>
        /// <param name="val">The value to set.</param>
        public void setNode(int nodeid, double val)
        {
            this._inputlist[nodeid, 1] = val;
        }

        /// <summary>
        /// Returns the size of the input vector.
        /// </summary>
        public int nodecount
        {
            get
            {
                return this._inputlist.RowCount;
            }
        }
    }
}
