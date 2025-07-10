class Transcriber:
    """Transcriber class to handle transcription tasks with similarity checks
    """
    def __init__(self):
        pass
    
    def _find_longest_common_prefix_with_similarity(self, text1, text2, min_similarity=0.8, return_from="text1", label=""):
        """
        Find the longest common prefix between two transcriptions using similarity scoring
        Returns the common text portion based on similarity threshold
        
        Args:
            text1, text2: Input texts to compare
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            return_from: "text1" or "text2" - which text to return words from
            label: Debug label for logging
        
        Returns:
            String: The longest prefix that maintains the similarity threshold
        """
        if not text1 or not text2:
            return ""
        
        # Split into words for comparison
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        original_words1 = text1.split()
        original_words2 = text2.split()
        
        if not words1 or not words2:
            return ""
        
        common_words = []
        total_comparisons = 0
        similar_matches = 0
        
        # Compare words from the beginning
        max_length = min(len(words1), len(words2))
        
        for i in range(max_length):
            word1 = words1[i]
            word2 = words2[i]
            total_comparisons += 1
            
            # Calculate similarity for this word pair
            word_similarity = self._calculate_word_similarity(word1, word2)
            
            # If words are similar enough, count as match
            if word_similarity >= min_similarity:
                similar_matches += 1
                # Use words from specified text
                if return_from == "text2":
                    original_word = original_words2[i] if i < len(original_words2) else word2
                else:
                    original_word = original_words1[i] if i < len(original_words1) else word1
                common_words.append(original_word)
            else:
                # Check if overall similarity is still above threshold
                current_similarity = similar_matches / total_comparisons if total_comparisons > 0 else 0
                
                if current_similarity < min_similarity:
                    # If adding this dissimilar word drops us below threshold, stop
                    break
                else:
                    # If overall similarity is still good, include this word but mark as different
                    similar_matches += 0.5  # Partial credit for maintaining flow
                    if return_from == "text2":
                        original_word = original_words2[i] if i < len(original_words2) else word2
                    else:
                        original_word = original_words1[i] if i < len(original_words1) else word1
                    common_words.append(original_word)
        
        # Final similarity check - ensure we meet the minimum threshold
        final_similarity = similar_matches / total_comparisons if total_comparisons > 0 else 0
        
        # If we don't meet the threshold, trim back until we do
        if final_similarity < min_similarity and len(common_words) > 1:
            # Try removing words from the end until we meet threshold
            for trim_count in range(1, len(common_words)):
                trimmed_comparisons = total_comparisons - trim_count
                trimmed_matches = similar_matches - (trim_count * 0.5)  # Assume trimmed words were partial matches
                
                if trimmed_comparisons > 0:
                    trimmed_similarity = trimmed_matches / trimmed_comparisons
                    if trimmed_similarity >= min_similarity:
                        common_words = common_words[:-trim_count]
                        final_similarity = trimmed_similarity
                        break
        
        result = " ".join(common_words)
        
        # print(f"DEBUG: Found common prefix with {final_similarity:.1%} similarity: '{result}'")
        print(f"DEBUG: Found common prefix with {final_similarity:.1} similarity: '{result}'")
        print(f"DEBUG: Length {len(result)} characters, returning from {return_from}, caller: {label}")
        
        return result

    def _calculate_word_similarity(self, word1, word2):
        """
        Calculate similarity between two words using edit distance only
        
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if word1 == word2:
            return 1.0
        
        # Handle punctuation
        clean_word1 = self._clean_word_for_comparison(word1)
        clean_word2 = self._clean_word_for_comparison(word2)
        
        if clean_word1 == clean_word2:
            return 0.95  # High similarity for punctuation differences
        
        # Use edit distance for general similarity
        return self._levenshtein_similarity(clean_word1, clean_word2)

    def _clean_word_for_comparison(self, word):
        """
        Clean word by removing punctuation and normalizing
        """
        import string
        # Remove punctuation
        cleaned = word.translate(str.maketrans('', '', string.punctuation))
        return cleaned.lower().strip()

    def _levenshtein_similarity(self, word1, word2):
        """
        Calculate similarity using Levenshtein distance
        
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if not word1 or not word2:
            return 0.0
        
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(word1, word2)
        max_length = max(len(word1), len(word2))
        
        if max_length == 0:
            return 1.0
        
        # Convert distance to similarity (0 distance = 1.0 similarity)
        similarity = 1.0 - (distance / max_length)
        
        # Only consider it similar if it's above a threshold
        return similarity if similarity >= 0.6 else 0.0

    def _levenshtein_distance(self, s1, s2):
        """
        Calculate the Levenshtein distance between two strings
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]



transcriber = Transcriber()
# Example usage:
text1 = "You're doing good. You're doing well. this is my test."
text2 = "You're doing so good. You're doing so well. And this is my test. it is another saying my name"

# Return words from text1 with 80% similarity
result1 = transcriber._find_longest_common_prefix_with_similarity(text1, text2, 0.8, "text1", "test")
print(f"From text1: '{result1}'")

# Return words from text2 with 80% similarity
result2 = transcriber._find_longest_common_prefix_with_similarity(text1, text2, 0.5, "text2", "test")
result5 = transcriber._find_longest_common_prefix_with_similarity(text1, text2, 0.5, "text1", "test")
print(f"From text2: '{result2} \nFrom text1: '{result5}'")

# Higher similarity threshold
result3 = transcriber._find_longest_common_prefix_with_similarity(text1, text2, 0.9, "text1", "test")
print(f"90% threshold from text1: '{result3}'")