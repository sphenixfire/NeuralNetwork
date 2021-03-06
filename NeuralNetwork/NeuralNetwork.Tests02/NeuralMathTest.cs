// <copyright file="NeuralMathTest.cs">Copyright ©  2018</copyright>
using System;
using Microsoft.Pex.Framework;
using Microsoft.Pex.Framework.Validation;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;

namespace NeuralNetwork.Tests
{
    /// <summary>This class contains parameterized unit tests for NeuralMath</summary>
    [PexClass(typeof(NeuralMath))]
    [PexAllowedExceptionFromTypeUnderTest(typeof(InvalidOperationException))]
    [PexAllowedExceptionFromTypeUnderTest(typeof(ArgumentException), AcceptExceptionSubtypes = true)]
    [TestClass]
    public partial class NeuralMathTest
    {
        /// <summary>Test stub for getSigmoidDerivate(Double)</summary>
        [PexMethod]
        public double getSigmoidDerivateTest(double x)
        {
            double result = NeuralMath.getSigmoidDerivate(x);
            return result;
            // TODO: add assertions to method NeuralMathTest.getSigmoidDerivateTest(Double)
        }

        /// <summary>Test stub for getSigmoid(Double)</summary>
        [PexMethod]
        public double getSigmoidTest(double x)
        {
            double result = NeuralMath.getSigmoid(x);
            return result;
            // TODO: add assertions to method NeuralMathTest.getSigmoidTest(Double)
        }
    }
}
