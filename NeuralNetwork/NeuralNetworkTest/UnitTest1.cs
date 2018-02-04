using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;

namespace NeuralNetworkTest
{
    [TestClass]
    public class NeuralMathTest
    {
        [TestMethod]
        public void getSigmoidTest()
        {
            Assert.AreEqual(0.597, Math.Round(NeuralMath.getSigmoid(0.4), 3));
        }
    }
}
