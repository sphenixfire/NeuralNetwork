using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralMath
    {
        /// <summary>
        /// Returns sigmoid value for activaton function.
        /// </summary>
        /// <param name="x">Input (x-value) for sigmoid.</param>
        /// <returns>y-value of sigmoid</returns>
        public static double getSigmoid(double x)
        {
             return 1 / (1 + Math.Exp(-x));
        }

        /// <summary>
        /// Returns sidmoid derivate (d/dx sigmoid).
        /// </summary>
        /// <param name="x">Input (x-value) for derivatesigmoid.</param>
        /// <returns>y-value</returns>
        public static double getSigmoidDerivate(double x)
        {
            double s = NeuralMath.getSigmoid(x);
            return s * (1.0 - s);
        }
    }
}
