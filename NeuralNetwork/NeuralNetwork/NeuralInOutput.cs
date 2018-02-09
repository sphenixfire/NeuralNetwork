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
    public class NeuralInOutput
    {
        /// <summary>
        /// Nodes as vector (matrix of 1 column, several rows).
        /// </summary>
        private Matrix<double> _nodelist;

        /// <summary>
        /// Allows to create the InOutput only by Neural-Network function.
        /// </summary>
        /// <param name="nodecount">Nomber of nodes to create</param>
        internal NeuralInOutput(int nodecount)
        {
            this._nodelist = Matrix<double>.Build.Dense(nodecount, 1, 0.01);
        }

        /// <summary>
        /// Allows to create the InOutput only by the Neural-Network.
        /// </summary>
        /// <param name="nodelist"></param>
        internal NeuralInOutput(Matrix<double> nodelist)
        {
            this.setNodelist(nodelist);
        }

        /// <summary>
        /// Returns the node list for use by neural network.
        /// </summary>
        /// <returns>The nodelist.</returns>
        internal protected Matrix<double> getNodelist()
        {
            return this._nodelist;
        }

        /// <summary>
        /// Sets the node list by neural network.
        /// </summary>
        /// <param name="nodelist"></param>
        protected void setNodelist(Matrix<double> nodelist)
        {
            this._nodelist = nodelist;
        }
        
        /// <summary>
        /// Sets an node value.
        /// </summary>
        /// <param name="nodeid">The position in the vector.</param>
        /// <param name="val">The value to set.</param>
        public void setNode(int nodeid, double val)
        {
            this._nodelist[nodeid, 0] = val;
        }

        /// <summary>
        /// Gets an node value.
        /// </summary>
        /// <param name="nodeid">Id of the node</param>
        /// <returns>Node value</returns>
        public double getNode(int nodeid)
        {
            return this._nodelist[nodeid, 0];
        }


        /// <summary>
        /// Returns the size of the vector.
        /// </summary>
        public int nodecount
        {
            get
            {
                return this._nodelist.RowCount;
            }
        }
    }
}
