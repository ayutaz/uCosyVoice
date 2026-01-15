using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace uCosyVoice.Tokenizer
{
    /// <summary>
    /// Text normalizer for English TTS input.
    /// Handles number expansion, abbreviations, and basic cleanup.
    /// </summary>
    public static class TextNormalizer
    {
        private static readonly Dictionary<string, string> Abbreviations = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            {"mr.", "mister"},
            {"mrs.", "missus"},
            {"ms.", "miss"},
            {"dr.", "doctor"},
            {"prof.", "professor"},
            {"st.", "street"},
            {"ave.", "avenue"},
            {"blvd.", "boulevard"},
            {"etc.", "etcetera"},
            {"vs.", "versus"},
            {"jr.", "junior"},
            {"sr.", "senior"},
            {"inc.", "incorporated"},
            {"ltd.", "limited"},
            {"co.", "company"},
            {"corp.", "corporation"},
            {"jan.", "january"},
            {"feb.", "february"},
            {"mar.", "march"},
            {"apr.", "april"},
            {"jun.", "june"},
            {"jul.", "july"},
            {"aug.", "august"},
            {"sep.", "september"},
            {"sept.", "september"},
            {"oct.", "october"},
            {"nov.", "november"},
            {"dec.", "december"},
        };

        private static readonly string[] Ones = { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };

        private static readonly string[] Tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

        private static readonly string[] Thousands = { "", "thousand", "million", "billion", "trillion" };

        /// <summary>
        /// Normalize text for TTS input.
        /// </summary>
        /// <param name="text">Raw input text</param>
        /// <returns>Normalized text</returns>
        public static string Normalize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return text;

            var result = text;

            // Normalize whitespace
            result = NormalizeWhitespace(result);

            // Expand abbreviations
            result = ExpandAbbreviations(result);

            // Expand numbers
            result = ExpandNumbers(result);

            // Clean up punctuation
            result = CleanPunctuation(result);

            // Final whitespace cleanup
            result = NormalizeWhitespace(result);

            return result;
        }

        /// <summary>
        /// Add language tag for CosyVoice.
        /// </summary>
        /// <param name="text">Normalized text</param>
        /// <param name="language">Language code (default: "en")</param>
        /// <returns>Text with language tag</returns>
        public static string AddLanguageTag(string text, string language = "en")
        {
            return $"<|{language}|>{text}";
        }

        private static string NormalizeWhitespace(string text)
        {
            // Replace multiple whitespace with single space
            text = Regex.Replace(text, @"\s+", " ");
            return text.Trim();
        }

        private static string ExpandAbbreviations(string text)
        {
            foreach (var kvp in Abbreviations)
            {
                // Use word boundary to avoid partial matches
                var pattern = $@"\b{Regex.Escape(kvp.Key)}";
                text = Regex.Replace(text, pattern, kvp.Value, RegexOptions.IgnoreCase);
            }
            return text;
        }

        private static string ExpandNumbers(string text)
        {
            // Handle decimal numbers (e.g., 3.14)
            text = Regex.Replace(text, @"(\d+)\.(\d+)", m =>
            {
                var intPart = m.Groups[1].Value;
                var decPart = m.Groups[2].Value;
                var intWords = NumberToWords(long.Parse(intPart));
                var decWords = DigitsToWords(decPart);
                return $"{intWords} point {decWords}";
            });

            // Handle percentages
            text = Regex.Replace(text, @"(\d+)%", m =>
            {
                var num = long.Parse(m.Groups[1].Value);
                return NumberToWords(num) + " percent";
            });

            // Handle ordinals (1st, 2nd, 3rd, etc.)
            text = Regex.Replace(text, @"(\d+)(st|nd|rd|th)\b", m =>
            {
                var num = long.Parse(m.Groups[1].Value);
                return NumberToOrdinal(num);
            });

            // Handle currency
            text = Regex.Replace(text, @"\$(\d+)(?:\.(\d{2}))?", m =>
            {
                var dollars = long.Parse(m.Groups[1].Value);
                var cents = m.Groups[2].Success ? int.Parse(m.Groups[2].Value) : 0;

                var result = NumberToWords(dollars) + (dollars == 1 ? " dollar" : " dollars");
                if (cents > 0)
                {
                    result += " and " + NumberToWords(cents) + (cents == 1 ? " cent" : " cents");
                }
                return result;
            });

            // Handle plain integers
            text = Regex.Replace(text, @"\b(\d+)\b", m =>
            {
                if (long.TryParse(m.Groups[1].Value, out long num))
                {
                    return NumberToWords(num);
                }
                return m.Value;
            });

            return text;
        }

        private static string CleanPunctuation(string text)
        {
            // Remove or replace problematic characters using unicode escapes
            text = text.Replace("\u2026", "...");  // ellipsis
            text = text.Replace("\u2014", ", ");   // em dash
            text = text.Replace("\u2013", ", ");   // en dash
            text = text.Replace("\"", "");
            text = text.Replace("\u201C", "");     // left double quote
            text = text.Replace("\u201D", "");     // right double quote
            text = text.Replace("\u2018", "'");    // left single quote
            text = text.Replace("\u2019", "'");    // right single quote

            // Normalize ellipsis
            text = Regex.Replace(text, @"\.{2,}", "...");

            return text;
        }

        /// <summary>
        /// Convert number to English words.
        /// </summary>
        public static string NumberToWords(long num)
        {
            if (num == 0)
                return "zero";

            if (num < 0)
                return "negative " + NumberToWords(-num);

            var words = new StringBuilder();

            int groupIndex = 0;
            while (num > 0)
            {
                int group = (int)(num % 1000);
                if (group > 0)
                {
                    var groupWords = GroupToWords(group);
                    if (groupIndex > 0)
                    {
                        groupWords += " " + Thousands[groupIndex];
                    }

                    if (words.Length > 0)
                    {
                        words.Insert(0, groupWords + " ");
                    }
                    else
                    {
                        words.Append(groupWords);
                    }
                }

                num /= 1000;
                groupIndex++;
            }

            return words.ToString().Trim();
        }

        private static string GroupToWords(int num)
        {
            if (num == 0)
                return "";

            var words = new StringBuilder();

            if (num >= 100)
            {
                words.Append(Ones[num / 100] + " hundred");
                num %= 100;
                if (num > 0)
                    words.Append(" ");
            }

            if (num >= 20)
            {
                words.Append(Tens[num / 10]);
                num %= 10;
                if (num > 0)
                    words.Append(" " + Ones[num]);
            }
            else if (num > 0)
            {
                words.Append(Ones[num]);
            }

            return words.ToString();
        }

        private static string DigitsToWords(string digits)
        {
            var words = new List<string>();
            foreach (char c in digits)
            {
                if (char.IsDigit(c))
                {
                    int d = c - '0';
                    words.Add(d == 0 ? "zero" : Ones[d]);
                }
            }
            return string.Join(" ", words);
        }

        /// <summary>
        /// Convert number to ordinal words.
        /// </summary>
        public static string NumberToOrdinal(long num)
        {
            if (num <= 0)
                return NumberToWords(num);

            // Special cases
            if (num == 1) return "first";
            if (num == 2) return "second";
            if (num == 3) return "third";
            if (num == 5) return "fifth";
            if (num == 8) return "eighth";
            if (num == 9) return "ninth";
            if (num == 12) return "twelfth";

            var words = NumberToWords(num);

            // Apply ordinal suffix rules
            if (words.EndsWith("y"))
            {
                words = words.Substring(0, words.Length - 1) + "ieth";
            }
            else if (words.EndsWith("ve"))
            {
                words = words.Substring(0, words.Length - 2) + "fth";
            }
            else if (words.EndsWith("t") && !words.EndsWith("ght"))
            {
                // eight -> eighth (already handled), but twenty -> twentieth
                words += "h";
            }
            else if (words.EndsWith("e"))
            {
                words = words.Substring(0, words.Length - 1) + "th";
            }
            else
            {
                words += "th";
            }

            return words;
        }
    }
}
