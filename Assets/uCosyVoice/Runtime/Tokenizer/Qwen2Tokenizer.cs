using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace uCosyVoice.Tokenizer
{
    /// <summary>
    /// Qwen2 BPE Tokenizer for CosyVoice3 text processing.
    /// Implements byte-level BPE similar to GPT-2/Qwen2.
    /// </summary>
    public class Qwen2Tokenizer
    {
        private Dictionary<string, int> _vocab;
        private Dictionary<int, string> _reverseVocab;
        private List<(string, string)> _merges;
        private Dictionary<(string, string), int> _mergeRanks;
        private Dictionary<byte, char> _byteToChar;
        private Dictionary<char, byte> _charToByte;

        private bool _isLoaded;

        /// <summary>
        /// Whether the tokenizer is loaded and ready.
        /// </summary>
        public bool IsLoaded => _isLoaded;

        /// <summary>
        /// Vocabulary size.
        /// </summary>
        public int VocabSize => _vocab?.Count ?? 0;

        /// <summary>
        /// Load tokenizer from StreamingAssets.
        /// </summary>
        /// <param name="vocabPath">Path to vocab.json (relative to StreamingAssets)</param>
        /// <param name="mergesPath">Path to merges.txt (relative to StreamingAssets)</param>
        public void Load(string vocabPath = "CosyVoice/tokenizer/vocab.json",
                        string mergesPath = "CosyVoice/tokenizer/merges.txt")
        {
            string vocabFullPath = Path.Combine(Application.streamingAssetsPath, vocabPath);
            string mergesFullPath = Path.Combine(Application.streamingAssetsPath, mergesPath);

            LoadFromPaths(vocabFullPath, mergesFullPath);
        }

        /// <summary>
        /// Load tokenizer from absolute file paths.
        /// </summary>
        public void LoadFromPaths(string vocabPath, string mergesPath)
        {
            if (!File.Exists(vocabPath))
                throw new FileNotFoundException($"Vocab file not found: {vocabPath}");
            if (!File.Exists(mergesPath))
                throw new FileNotFoundException($"Merges file not found: {mergesPath}");

            // Initialize byte-to-char mapping (GPT-2 style)
            InitByteMapping();

            // Load vocabulary
            LoadVocab(vocabPath);

            // Load BPE merges
            LoadMerges(mergesPath);

            _isLoaded = true;
            Debug.Log($"Qwen2Tokenizer loaded: {_vocab.Count} tokens, {_merges.Count} merges");
        }

        /// <summary>
        /// Encode text to token IDs.
        /// </summary>
        /// <param name="text">Input text</param>
        /// <returns>Array of token IDs</returns>
        public int[] Encode(string text)
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Tokenizer not loaded. Call Load() first.");

            if (string.IsNullOrEmpty(text))
                return Array.Empty<int>();

            var tokens = new List<int>();

            // Process text word by word (split on whitespace)
            var words = SplitIntoWords(text);

            foreach (var word in words)
            {
                var wordTokens = TokenizeWord(word);
                tokens.AddRange(wordTokens);
            }

            return tokens.ToArray();
        }

        /// <summary>
        /// Decode token IDs back to text.
        /// </summary>
        /// <param name="tokenIds">Token IDs</param>
        /// <returns>Decoded text</returns>
        public string Decode(int[] tokenIds)
        {
            if (!_isLoaded)
                throw new InvalidOperationException("Tokenizer not loaded. Call Load() first.");

            if (tokenIds == null || tokenIds.Length == 0)
                return string.Empty;

            var sb = new StringBuilder();
            foreach (var id in tokenIds)
            {
                if (_reverseVocab.TryGetValue(id, out var token))
                {
                    // Convert BPE token back to text
                    var decoded = DecodeBpeToken(token);
                    sb.Append(decoded);
                }
            }

            return sb.ToString();
        }

        private void InitByteMapping()
        {
            // GPT-2 style byte encoding
            // Maps bytes to printable unicode characters
            _byteToChar = new Dictionary<byte, char>();
            _charToByte = new Dictionary<char, byte>();

            int n = 0;
            // Printable ASCII range
            for (int b = (int)'!'; b <= (int)'~'; b++)
            {
                _byteToChar[(byte)b] = (char)b;
                _charToByte[(char)b] = (byte)b;
            }
            // Extended range
            for (int b = (int)'¡'; b <= (int)'¬'; b++)
            {
                _byteToChar[(byte)b] = (char)b;
                _charToByte[(char)b] = (byte)b;
            }
            for (int b = (int)'®'; b <= (int)'ÿ'; b++)
            {
                _byteToChar[(byte)b] = (char)b;
                _charToByte[(char)b] = (byte)b;
            }

            // Map remaining bytes to unicode private use area
            n = 0;
            for (int b = 0; b < 256; b++)
            {
                if (!_byteToChar.ContainsKey((byte)b))
                {
                    char c = (char)(256 + n);
                    _byteToChar[(byte)b] = c;
                    _charToByte[c] = (byte)b;
                    n++;
                }
            }
        }

        private void LoadVocab(string path)
        {
            _vocab = new Dictionary<string, int>();
            _reverseVocab = new Dictionary<int, string>();

            var json = File.ReadAllText(path, Encoding.UTF8);

            // Simple JSON parsing for {"token": id, ...} format
            json = json.Trim();
            if (json.StartsWith("{") && json.EndsWith("}"))
            {
                json = json.Substring(1, json.Length - 2);
            }

            // Parse key-value pairs
            int pos = 0;
            while (pos < json.Length)
            {
                // Skip whitespace
                while (pos < json.Length && char.IsWhiteSpace(json[pos]))
                    pos++;

                if (pos >= json.Length)
                    break;

                // Parse key (string)
                if (json[pos] != '"')
                {
                    pos++;
                    continue;
                }

                var (key, nextPos) = ParseJsonString(json, pos);
                pos = nextPos;

                // Skip colon
                while (pos < json.Length && (char.IsWhiteSpace(json[pos]) || json[pos] == ':'))
                    pos++;

                // Parse value (integer)
                int numStart = pos;
                while (pos < json.Length && (char.IsDigit(json[pos]) || json[pos] == '-'))
                    pos++;

                if (numStart < pos)
                {
                    if (int.TryParse(json.Substring(numStart, pos - numStart), out int value))
                    {
                        _vocab[key] = value;
                        _reverseVocab[value] = key;
                    }
                }

                // Skip comma
                while (pos < json.Length && (char.IsWhiteSpace(json[pos]) || json[pos] == ','))
                    pos++;
            }
        }

        private (string, int) ParseJsonString(string json, int start)
        {
            if (json[start] != '"')
                return ("", start);

            var sb = new StringBuilder();
            int pos = start + 1;

            while (pos < json.Length)
            {
                char c = json[pos];

                if (c == '"')
                {
                    return (sb.ToString(), pos + 1);
                }
                else if (c == '\\' && pos + 1 < json.Length)
                {
                    char next = json[pos + 1];
                    switch (next)
                    {
                        case 'n': sb.Append('\n'); pos += 2; break;
                        case 'r': sb.Append('\r'); pos += 2; break;
                        case 't': sb.Append('\t'); pos += 2; break;
                        case '"': sb.Append('"'); pos += 2; break;
                        case '\\': sb.Append('\\'); pos += 2; break;
                        case 'u':
                            if (pos + 5 < json.Length)
                            {
                                var hex = json.Substring(pos + 2, 4);
                                if (int.TryParse(hex, System.Globalization.NumberStyles.HexNumber, null, out int code))
                                {
                                    sb.Append((char)code);
                                }
                                pos += 6;
                            }
                            else
                            {
                                pos++;
                            }
                            break;
                        default:
                            sb.Append(next);
                            pos += 2;
                            break;
                    }
                }
                else
                {
                    sb.Append(c);
                    pos++;
                }
            }

            return (sb.ToString(), pos);
        }

        private void LoadMerges(string path)
        {
            _merges = new List<(string, string)>();
            _mergeRanks = new Dictionary<(string, string), int>();

            var lines = File.ReadAllLines(path, Encoding.UTF8);
            int rank = 0;

            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                // Skip comment lines
                if (line.StartsWith("#"))
                    continue;

                var parts = line.Split(' ');
                if (parts.Length >= 2)
                {
                    var merge = (parts[0], parts[1]);
                    _merges.Add(merge);
                    _mergeRanks[merge] = rank++;
                }
            }
        }

        private List<string> SplitIntoWords(string text)
        {
            var words = new List<string>();
            var currentWord = new StringBuilder();

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];

                if (char.IsWhiteSpace(c))
                {
                    if (currentWord.Length > 0)
                    {
                        words.Add(currentWord.ToString());
                        currentWord.Clear();
                    }
                    // Add space as prefix to next word (Ġ in BPE)
                    currentWord.Append(' ');
                }
                else
                {
                    currentWord.Append(c);
                }
            }

            if (currentWord.Length > 0)
            {
                words.Add(currentWord.ToString());
            }

            return words;
        }

        private List<int> TokenizeWord(string word)
        {
            // Convert word to byte-level representation
            var bytes = Encoding.UTF8.GetBytes(word);
            var tokens = new List<string>();

            foreach (var b in bytes)
            {
                if (_byteToChar.TryGetValue(b, out char c))
                {
                    tokens.Add(c.ToString());
                }
                else
                {
                    tokens.Add(((char)b).ToString());
                }
            }

            // Apply BPE merges
            tokens = ApplyBpeMerges(tokens);

            // Convert to token IDs
            var ids = new List<int>();
            foreach (var token in tokens)
            {
                if (_vocab.TryGetValue(token, out int id))
                {
                    ids.Add(id);
                }
                else
                {
                    // Unknown token - try to encode as individual bytes
                    foreach (char c in token)
                    {
                        if (_vocab.TryGetValue(c.ToString(), out int charId))
                        {
                            ids.Add(charId);
                        }
                    }
                }
            }

            return ids;
        }

        private List<string> ApplyBpeMerges(List<string> tokens)
        {
            if (tokens.Count < 2)
                return tokens;

            while (true)
            {
                // Find the highest-priority merge that can be applied
                int bestIdx = -1;
                int bestRank = int.MaxValue;

                for (int i = 0; i < tokens.Count - 1; i++)
                {
                    var pair = (tokens[i], tokens[i + 1]);
                    if (_mergeRanks.TryGetValue(pair, out int rank))
                    {
                        if (rank < bestRank)
                        {
                            bestRank = rank;
                            bestIdx = i;
                        }
                    }
                }

                if (bestIdx == -1)
                    break;

                // Apply the merge
                var newToken = tokens[bestIdx] + tokens[bestIdx + 1];
                tokens[bestIdx] = newToken;
                tokens.RemoveAt(bestIdx + 1);
            }

            return tokens;
        }

        private string DecodeBpeToken(string token)
        {
            var bytes = new List<byte>();

            foreach (char c in token)
            {
                if (_charToByte.TryGetValue(c, out byte b))
                {
                    bytes.Add(b);
                }
                else
                {
                    // Fallback: try direct cast
                    bytes.Add((byte)(c & 0xFF));
                }
            }

            try
            {
                return Encoding.UTF8.GetString(bytes.ToArray());
            }
            catch
            {
                return token;
            }
        }
    }
}
