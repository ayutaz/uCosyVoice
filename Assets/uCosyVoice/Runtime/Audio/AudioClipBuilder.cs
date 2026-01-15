using UnityEngine;

namespace uCosyVoice.Audio
{
    /// <summary>
    /// Utility class for creating Unity AudioClip from audio sample data.
    /// </summary>
    public static class AudioClipBuilder
    {
        public const int DEFAULT_SAMPLE_RATE = 24000;

        /// <summary>
        /// Create an AudioClip from float array audio data.
        /// </summary>
        /// <param name="samples">Audio samples (mono, normalized to [-1, 1])</param>
        /// <param name="sampleRate">Sample rate in Hz (default: 24000)</param>
        /// <param name="name">Name for the AudioClip</param>
        /// <returns>Unity AudioClip ready for playback</returns>
        public static AudioClip Build(float[] samples, int sampleRate = DEFAULT_SAMPLE_RATE, string name = "GeneratedAudio")
        {
            if (samples == null || samples.Length == 0)
            {
                Debug.LogWarning("AudioClipBuilder: Empty samples provided");
                return null;
            }

            // Create AudioClip
            var clip = AudioClip.Create(
                name: name,
                lengthSamples: samples.Length,
                channels: 1,
                frequency: sampleRate,
                stream: false
            );

            // Set audio data
            clip.SetData(samples, 0);

            return clip;
        }

        /// <summary>
        /// Create an AudioClip from float array audio data with stereo conversion.
        /// </summary>
        /// <param name="samples">Mono audio samples</param>
        /// <param name="sampleRate">Sample rate in Hz</param>
        /// <param name="name">Name for the AudioClip</param>
        /// <returns>Unity AudioClip (stereo)</returns>
        public static AudioClip BuildStereo(float[] samples, int sampleRate = DEFAULT_SAMPLE_RATE, string name = "GeneratedAudio")
        {
            if (samples == null || samples.Length == 0)
            {
                Debug.LogWarning("AudioClipBuilder: Empty samples provided");
                return null;
            }

            // Convert mono to stereo
            var stereoSamples = new float[samples.Length * 2];
            for (int i = 0; i < samples.Length; i++)
            {
                stereoSamples[i * 2] = samples[i];     // Left
                stereoSamples[i * 2 + 1] = samples[i]; // Right
            }

            var clip = AudioClip.Create(
                name: name,
                lengthSamples: samples.Length,
                channels: 2,
                frequency: sampleRate,
                stream: false
            );

            clip.SetData(stereoSamples, 0);

            return clip;
        }

        /// <summary>
        /// Get the duration of audio samples in seconds.
        /// </summary>
        /// <param name="sampleCount">Number of samples</param>
        /// <param name="sampleRate">Sample rate in Hz</param>
        /// <returns>Duration in seconds</returns>
        public static float GetDuration(int sampleCount, int sampleRate = DEFAULT_SAMPLE_RATE)
        {
            return (float)sampleCount / sampleRate;
        }

        /// <summary>
        /// Normalize audio samples to prevent clipping.
        /// </summary>
        /// <param name="samples">Audio samples to normalize</param>
        /// <param name="targetPeak">Target peak amplitude (default: 0.95)</param>
        /// <returns>Normalized samples</returns>
        public static float[] Normalize(float[] samples, float targetPeak = 0.95f)
        {
            if (samples == null || samples.Length == 0)
                return samples;

            // Find peak amplitude
            float maxAbs = 0f;
            for (int i = 0; i < samples.Length; i++)
            {
                float abs = Mathf.Abs(samples[i]);
                if (abs > maxAbs)
                    maxAbs = abs;
            }

            // Normalize if needed
            if (maxAbs > 0.001f && maxAbs != targetPeak)
            {
                float scale = targetPeak / maxAbs;
                var normalized = new float[samples.Length];
                for (int i = 0; i < samples.Length; i++)
                {
                    normalized[i] = samples[i] * scale;
                }
                return normalized;
            }

            return samples;
        }
    }
}
