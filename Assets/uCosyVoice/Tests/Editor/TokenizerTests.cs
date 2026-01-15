using System.IO;
using NUnit.Framework;
using UnityEngine;
using uCosyVoice.Tokenizer;

namespace uCosyVoice.Tests.Editor
{
    /// <summary>
    /// Tests for Phase 5: Tokenizer implementation
    /// </summary>
    public class TokenizerTests
    {
        private string _vocabPath;
        private string _mergesPath;

        [OneTimeSetUp]
        public void Setup()
        {
            _vocabPath = Path.Combine(Application.streamingAssetsPath, "CosyVoice/tokenizer/vocab.json");
            _mergesPath = Path.Combine(Application.streamingAssetsPath, "CosyVoice/tokenizer/merges.txt");
        }

        #region TextNormalizer Tests

        [Test]
        public void TextNormalizer_NumberToWords_Zero()
        {
            Assert.AreEqual("zero", TextNormalizer.NumberToWords(0));
        }

        [Test]
        public void TextNormalizer_NumberToWords_SingleDigits()
        {
            Assert.AreEqual("one", TextNormalizer.NumberToWords(1));
            Assert.AreEqual("five", TextNormalizer.NumberToWords(5));
            Assert.AreEqual("nine", TextNormalizer.NumberToWords(9));
        }

        [Test]
        public void TextNormalizer_NumberToWords_Teens()
        {
            Assert.AreEqual("eleven", TextNormalizer.NumberToWords(11));
            Assert.AreEqual("thirteen", TextNormalizer.NumberToWords(13));
            Assert.AreEqual("nineteen", TextNormalizer.NumberToWords(19));
        }

        [Test]
        public void TextNormalizer_NumberToWords_Tens()
        {
            Assert.AreEqual("twenty", TextNormalizer.NumberToWords(20));
            Assert.AreEqual("twenty one", TextNormalizer.NumberToWords(21));
            Assert.AreEqual("forty two", TextNormalizer.NumberToWords(42));
            Assert.AreEqual("ninety nine", TextNormalizer.NumberToWords(99));
        }

        [Test]
        public void TextNormalizer_NumberToWords_Hundreds()
        {
            Assert.AreEqual("one hundred", TextNormalizer.NumberToWords(100));
            Assert.AreEqual("two hundred fifty six", TextNormalizer.NumberToWords(256));
            Assert.AreEqual("nine hundred ninety nine", TextNormalizer.NumberToWords(999));
        }

        [Test]
        public void TextNormalizer_NumberToWords_Thousands()
        {
            Assert.AreEqual("one thousand", TextNormalizer.NumberToWords(1000));
            Assert.AreEqual("one thousand two hundred thirty four", TextNormalizer.NumberToWords(1234));
            Assert.AreEqual("twelve thousand three hundred forty five", TextNormalizer.NumberToWords(12345));
        }

        [Test]
        public void TextNormalizer_NumberToWords_Large()
        {
            Assert.AreEqual("one million", TextNormalizer.NumberToWords(1000000));
            Assert.AreEqual("one billion", TextNormalizer.NumberToWords(1000000000));
        }

        [Test]
        public void TextNormalizer_NumberToWords_Negative()
        {
            Assert.AreEqual("negative five", TextNormalizer.NumberToWords(-5));
            Assert.AreEqual("negative one hundred", TextNormalizer.NumberToWords(-100));
        }

        [Test]
        public void TextNormalizer_NumberToOrdinal_Basic()
        {
            Assert.AreEqual("first", TextNormalizer.NumberToOrdinal(1));
            Assert.AreEqual("second", TextNormalizer.NumberToOrdinal(2));
            Assert.AreEqual("third", TextNormalizer.NumberToOrdinal(3));
            Assert.AreEqual("fifth", TextNormalizer.NumberToOrdinal(5));
        }

        [Test]
        public void TextNormalizer_Normalize_WithNumbers()
        {
            var result = TextNormalizer.Normalize("I have 3 cats.");
            Assert.AreEqual("I have three cats.", result);
        }

        [Test]
        public void TextNormalizer_Normalize_WithAbbreviations()
        {
            var result = TextNormalizer.Normalize("Dr. Smith lives on Main St.");
            Assert.IsTrue(result.Contains("doctor"));
            Assert.IsTrue(result.Contains("street"));
        }

        [Test]
        public void TextNormalizer_Normalize_ExtraWhitespace()
        {
            var result = TextNormalizer.Normalize("Hello    world   test");
            Assert.AreEqual("Hello world test", result);
        }

        [Test]
        public void TextNormalizer_AddLanguageTag()
        {
            var result = TextNormalizer.AddLanguageTag("Hello world");
            Assert.AreEqual("<|en|>Hello world", result);
        }

        [Test]
        public void TextNormalizer_Normalize_EmptyString()
        {
            Assert.AreEqual("", TextNormalizer.Normalize(""));
            Assert.IsNull(TextNormalizer.Normalize(null));
        }

        #endregion

        #region Qwen2Tokenizer Tests

        [Test]
        public void Qwen2Tokenizer_FilesExist()
        {
            Assert.IsTrue(File.Exists(_vocabPath), $"Vocab file not found: {_vocabPath}");
            Assert.IsTrue(File.Exists(_mergesPath), $"Merges file not found: {_mergesPath}");
        }

        [Test]
        public void Qwen2Tokenizer_CanLoad()
        {
            if (!File.Exists(_vocabPath) || !File.Exists(_mergesPath))
            {
                Assert.Ignore("Tokenizer files not found");
                return;
            }

            var tokenizer = new Qwen2Tokenizer();
            tokenizer.LoadFromPaths(_vocabPath, _mergesPath);

            Assert.IsTrue(tokenizer.IsLoaded);
            Assert.Greater(tokenizer.VocabSize, 100000, "Vocab should be large");
            Debug.Log($"Loaded tokenizer with {tokenizer.VocabSize} tokens");
        }

        [Test]
        public void Qwen2Tokenizer_EncodeSimpleText()
        {
            if (!File.Exists(_vocabPath) || !File.Exists(_mergesPath))
            {
                Assert.Ignore("Tokenizer files not found");
                return;
            }

            var tokenizer = new Qwen2Tokenizer();
            tokenizer.LoadFromPaths(_vocabPath, _mergesPath);

            var tokens = tokenizer.Encode("Hello");
            Assert.IsNotNull(tokens);
            Assert.Greater(tokens.Length, 0, "Should produce tokens");
            Debug.Log($"'Hello' -> {tokens.Length} tokens: [{string.Join(", ", tokens)}]");
        }

        [Test]
        public void Qwen2Tokenizer_EncodeSentence()
        {
            if (!File.Exists(_vocabPath) || !File.Exists(_mergesPath))
            {
                Assert.Ignore("Tokenizer files not found");
                return;
            }

            var tokenizer = new Qwen2Tokenizer();
            tokenizer.LoadFromPaths(_vocabPath, _mergesPath);

            var tokens = tokenizer.Encode("Hello, how are you today?");
            Assert.IsNotNull(tokens);
            Assert.Greater(tokens.Length, 0);
            Debug.Log($"Sentence -> {tokens.Length} tokens");
        }

        [Test]
        public void Qwen2Tokenizer_DecodeRoundtrip()
        {
            if (!File.Exists(_vocabPath) || !File.Exists(_mergesPath))
            {
                Assert.Ignore("Tokenizer files not found");
                return;
            }

            var tokenizer = new Qwen2Tokenizer();
            tokenizer.LoadFromPaths(_vocabPath, _mergesPath);

            var original = "Hello world";
            var tokens = tokenizer.Encode(original);
            var decoded = tokenizer.Decode(tokens);

            Debug.Log($"Original: '{original}' -> Tokens: [{string.Join(", ", tokens)}] -> Decoded: '{decoded}'");

            // The decoded text should match (possibly with minor whitespace differences)
            Assert.IsTrue(decoded.Contains("Hello") || decoded.Contains("hello"));
            Assert.IsTrue(decoded.Contains("world"));
        }

        [Test]
        public void Qwen2Tokenizer_EncodeEmpty()
        {
            if (!File.Exists(_vocabPath) || !File.Exists(_mergesPath))
            {
                Assert.Ignore("Tokenizer files not found");
                return;
            }

            var tokenizer = new Qwen2Tokenizer();
            tokenizer.LoadFromPaths(_vocabPath, _mergesPath);

            var tokens = tokenizer.Encode("");
            Assert.AreEqual(0, tokens.Length);

            tokens = tokenizer.Encode(null);
            Assert.AreEqual(0, tokens.Length);
        }

        [Test]
        public void Qwen2Tokenizer_NotLoaded_ThrowsException()
        {
            var tokenizer = new Qwen2Tokenizer();

            Assert.IsFalse(tokenizer.IsLoaded);
            Assert.Throws<System.InvalidOperationException>(() => tokenizer.Encode("test"));
        }

        [Test]
        public void Qwen2Tokenizer_SpecialToken_EndOfPrompt()
        {
            if (!File.Exists(_vocabPath) || !File.Exists(_mergesPath))
            {
                Assert.Ignore("Tokenizer files not found");
                return;
            }

            var tokenizer = new Qwen2Tokenizer();
            tokenizer.LoadFromPaths(_vocabPath, _mergesPath);

            // Test that <|endofprompt|> is tokenized properly
            var tokens = tokenizer.Encode("You are a helpful assistant.<|endofprompt|>Hello world");
            Assert.IsNotNull(tokens);
            Assert.Greater(tokens.Length, 0);
            Debug.Log($"Text with <|endofprompt|> -> {tokens.Length} tokens: [{string.Join(", ", tokens)}]");

            // Decode and verify the special token is preserved
            var decoded = tokenizer.Decode(tokens);
            Debug.Log($"Decoded: '{decoded}'");

            // The special token should be in the decoded output
            Assert.IsTrue(decoded.Contains("endofprompt") || decoded.Contains("<|"),
                "Special token should be preserved in decode");
        }

        [Test]
        public void Qwen2Tokenizer_CosyVoice3_PromptTextFormat()
        {
            if (!File.Exists(_vocabPath) || !File.Exists(_mergesPath))
            {
                Assert.Ignore("Tokenizer files not found");
                return;
            }

            var tokenizer = new Qwen2Tokenizer();
            tokenizer.LoadFromPaths(_vocabPath, _mergesPath);

            // Test the full CosyVoice3 prompt format
            var promptText = "You are a helpful assistant.<|endofprompt|>Hello, my name is Sarah.";
            var tokens = tokenizer.Encode(promptText);

            Assert.IsNotNull(tokens);
            Assert.Greater(tokens.Length, 0);
            Debug.Log($"CosyVoice3 prompt format -> {tokens.Length} tokens");

            // Test TTS text (plain text without special tokens)
            var ttsText = "Nice to meet you!";
            var ttsTokens = tokenizer.Encode(ttsText);
            Assert.Greater(ttsTokens.Length, 0);
            Debug.Log($"TTS text -> {ttsTokens.Length} tokens");
        }

        #endregion
    }
}
