class Transcriber:
    """Transcriber class to handle transcription tasks with similarity checks
    """
    def __init__(self):
        pass
    
    def _find_longest_common_prefix_with_similarity(self, text1, text2, min_similarity=0.8, label=""):
        """
        Find the longest common prefix between two transcriptions using similarity scoring
        Returns the common text portion based on similarity threshold
        
        Args:
            text1, text2: Input texts to compare
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            label: Debug label for logging
        
        Returns:
            String: The longest prefix that maintains the similarity threshold
        """
        if not text1 or not text2:
            return ""
        
        # Split into words for comparison
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
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
                # Use the original case from text1 for consistency
                original_word = text1.split()[i] if i < len(text1.split()) else word1
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
                    original_word = text1.split()[i] if i < len(text1.split()) else word1
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
        
        print(f"DEBUG: Found common prefix with {final_similarity:.1%} similarity: '{result}'")
        print(f"DEBUG: Length {len(result)} characters, caller: {label}")
        
        return result

    def _calculate_word_similarity(self, word1, word2):
        """
        Calculate similarity between two words using multiple methods
        
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if word1 == word2:
            return 1.0
        
        # Handle common variations and punctuation
        clean_word1 = self._clean_word_for_comparison(word1)
        clean_word2 = self._clean_word_for_comparison(word2)
        
        if clean_word1 == clean_word2:
            return 0.95  # High similarity for punctuation differences
        
        # Check for common substitutions
        similarity = self._check_common_substitutions(clean_word1, clean_word2)
        if similarity > 0:
            return similarity
        
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

    def _check_common_substitutions(self, word1, word2):
        """
        Check for common word substitutions that should be considered similar
        
        Returns:
            float: Similarity score if substitution found, 0 otherwise
        """
        # Define common substitutions with their similarity scores
        substitutions = {
            # Common word variations
            ('good', 'well'): 0.85,
            ('well', 'good'): 0.85,
            ('so', 'very'): 0.80,
            ('very', 'so'): 0.80,
            ('and', 'but'): 0.70,
            ('but', 'and'): 0.70,
            ('this', 'that'): 0.80,
            ('that', 'this'): 0.80,
            ('my', 'the'): 0.75,
            ('the', 'my'): 0.75,
            ('is', 'was'): 0.85,
            ('was', 'is'): 0.85,
            ('a', 'an'): 0.90,
            ('an', 'a'): 0.90,
            # Add more substitutions as needed
        }
        
        # Check if this pair exists in our substitutions
        pair = (word1, word2)
        if pair in substitutions:
            return substitutions[pair]
        
        # Check for partial matches (one word contains the other)
        if word1 in word2 or word2 in word1:
            shorter = min(len(word1), len(word2))
            longer = max(len(word1), len(word2))
            return shorter / longer * 0.8  # Partial credit for containment
        
        return 0

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



# Example usage with your test texts:
text1 = "You're doing good. You're doing well. this is my test."
text2 = "You're doing so good. You're doing so well. And this is my test. it is another saying my name"


transcriber = Transcriber()
# 80% similarity threshold
result_80 = transcriber._find_longest_common_prefix_with_similarity(text1, text2, 0.8, "test_80")
print(f"80% threshold: '{result_80}'")

# 90% similarity threshold  
result_90 = transcriber._find_longest_common_prefix_with_similarity(text1, text2, 0.6, "test_90")
print(f"90% threshold: '{result_90}'")

# 95% similarity threshold
result_95 = transcriber._find_longest_common_prefix_with_similarity(text1, text2, 0.3, "test_95")

print(f"95% threshold: '{result_95}'")