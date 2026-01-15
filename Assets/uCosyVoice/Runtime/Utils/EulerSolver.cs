using System;

namespace uCosyVoice.Utils
{
    /// <summary>
    /// Simple Euler ODE solver for Flow Matching.
    /// Solves dx/dt = v(x, t) from t=0 to t=1.
    /// </summary>
    public class EulerSolver
    {
        private readonly int _numSteps;
        private readonly float _dt;

        /// <summary>Number of integration steps</summary>
        public int NumSteps => _numSteps;

        /// <summary>Time delta per step</summary>
        public float Dt => _dt;

        /// <summary>
        /// Create Euler solver with specified number of steps.
        /// </summary>
        /// <param name="numSteps">Number of integration steps (default: 10)</param>
        public EulerSolver(int numSteps = 10)
        {
            if (numSteps < 1)
                throw new ArgumentException("numSteps must be >= 1", nameof(numSteps));

            _numSteps = numSteps;
            _dt = 1.0f / numSteps;
        }

        /// <summary>
        /// Get the time value for a given step index.
        /// </summary>
        /// <param name="stepIndex">Step index (0 to numSteps-1)</param>
        /// <returns>Time value t in [0, 1)</returns>
        public float GetTime(int stepIndex)
        {
            if (stepIndex < 0 || stepIndex >= _numSteps)
                throw new ArgumentOutOfRangeException(nameof(stepIndex));

            return stepIndex * _dt;
        }

        /// <summary>
        /// Perform one Euler step in-place: x = x + dt * velocity
        /// </summary>
        /// <param name="x">State array (modified in-place)</param>
        /// <param name="velocity">Velocity array</param>
        public void StepInPlace(float[] x, float[] velocity)
        {
            if (x.Length != velocity.Length)
                throw new ArgumentException("x and velocity must have same length");

            for (int i = 0; i < x.Length; i++)
            {
                x[i] += _dt * velocity[i];
            }
        }

        /// <summary>
        /// Perform one Euler step, returning new array: x_new = x + dt * velocity
        /// </summary>
        /// <param name="x">State array</param>
        /// <param name="velocity">Velocity array</param>
        /// <returns>New state array</returns>
        public float[] Step(float[] x, float[] velocity)
        {
            if (x.Length != velocity.Length)
                throw new ArgumentException("x and velocity must have same length");

            var result = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                result[i] = x[i] + _dt * velocity[i];
            }
            return result;
        }
    }
}
