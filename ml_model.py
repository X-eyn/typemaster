import numpy as np
import os
import json
import random
from collections import Counter
from sklearn.cluster import KMeans
import joblib
import re
from word_dataset import WordDataset  # Import our new word dataset utility

class TypingModel:
    """Machine learning model that learns from user's typing patterns and mistakes."""
    
    def __init__(self):
        self.model_dir = 'models'
        self.data_dir = 'data'
        self.word_bank_file = 'data/word_bank.json'
        self.common_words_file = 'data/common_words.json'
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize word dataset for dynamic word fetching
        self.word_dataset = WordDataset(cache_dir=self.data_dir)
        
        # Load or create word bank (this will be for backward compatibility)
        self.word_bank = self._load_word_bank()
        self.common_words = self._load_common_words()
        
        # Initialize user models dictionary
        self.user_models = {}
    
    def _load_word_bank(self):
        """Load or create word bank for text generation."""
        if os.path.exists(self.word_bank_file):
            with open(self.word_bank_file, 'r') as f:
                return json.load(f)
        else:
            # Initial word bank with different categories
            word_bank = {
                'easy': self._generate_default_word_list('easy'),
                'medium': self._generate_default_word_list('medium'),
                'hard': self._generate_default_word_list('hard')
            }
            
            with open(self.word_bank_file, 'w') as f:
                json.dump(word_bank, f)
            
            return word_bank
    
    def _load_common_words(self):
        """Load or create list of common English words."""
        if os.path.exists(self.common_words_file):
            with open(self.common_words_file, 'r') as f:
                return json.load(f)
        else:
            # List of common English words
            common_words = {
                'easy': [
                    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
                    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she"
                ],
                'medium': [
                    "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "computer", "keyboard", "typing",
                    "practice", "improve", "skills", "speed", "accuracy", "words", "minute", "test", "challenge", "learn",
                    "master", "performance", "training", "efficient", "effective", "productivity", "focus", "concentration", "rhythm"
                ],
                'hard': [
                    "algorithm", "authentication", "bureaucracy", "catastrophic", "disestablishment", "entrepreneurial", "functionality",
                    "heterogeneous", "infrastructure", "juxtaposition", "knowledgeable", "linguistically", "Mediterranean", "neuropsychology",
                    "onomatopoeia", "parallelogram", "quintessential", "reconnaissance", "surreptitious", "thermodynamics"
                ]
            }
            
            with open(self.common_words_file, 'w') as f:
                json.dump(common_words, f)
            
            return common_words
    
    def _generate_default_word_list(self, difficulty):
        """Generate a default word list based on difficulty."""
        if difficulty == 'easy':
            return [
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
            ]
        elif difficulty == 'medium':
            return [
                "computer", "keyboard", "typing", "practice", "improve", "skills",
                "speed", "accuracy", "words", "minute", "test", "challenge", "learn"
            ]
        else:  # hard
            return [
                "algorithm", "authentication", "bureaucracy", "catastrophic", "disestablishment",
                "entrepreneurial", "functionality", "heterogeneous", "infrastructure", "juxtaposition"
            ]
    
    def _load_user_model(self, user_id):
        """Load user model if it exists, otherwise create a new one."""
        model_path = os.path.join(self.model_dir, f"{user_id}_model.pkl")
        
        if user_id in self.user_models:
            return self.user_models[user_id]
        
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                print(f"Error loading user model: {e}")
                # Fall back to creating a new model
        
        # Create new user model with default weights
        user_model = {
            'character_mistakes': {},
            'word_mistakes': {},
            'difficult_sequences': [],
            'mistake_clusters': None,
            'session_count': 0
        }
        
        self.user_models[user_id] = user_model
        return user_model
    
    def _save_user_model(self, user_id, model):
        """Save user model to disk."""
        model_path = os.path.join(self.model_dir, f"{user_id}_model.pkl")
        self.user_models[user_id] = model
        joblib.dump(model, model_path)
    
    def generate_text(self, user_id, difficulty='medium'):
        """Generate a typing test text based on user's history and difficulty."""
        try:
            user_model = self._load_user_model(user_id)
            
            # For new users or after few sessions, use standard texts
            if user_model['session_count'] < 3:
                return self._generate_standard_text(difficulty)
            
            # Generate personalized text based on user's mistakes and patterns
            words = []
            
            # Get words with difficult characters for this user
            difficult_chars = []
            if user_model.get('character_mistakes') and len(user_model['character_mistakes']) > 0:
                difficult_chars = sorted(user_model['character_mistakes'].items(), 
                                      key=lambda x: x[1], reverse=True)
                # Just extract the character from the tuples
                difficult_chars = [char for char, _ in difficult_chars[:7]]
                
                # Get words containing user's difficult characters from our dataset
                if difficult_chars:
                    char_words = self.word_dataset.get_words_by_difficulty(
                        difficulty, 
                        count=10, 
                        containing_chars=difficult_chars
                    )
                    words.extend(char_words)
            
            # Add some words from difficult sequences
            difficult_sequences = []
            if user_model.get('difficult_sequences') and user_model['difficult_sequences']:
                difficult_sequences = user_model['difficult_sequences'][:5]
                
                # Get words containing user's difficult sequences from our dataset
                if difficult_sequences:
                    seq_words = self.word_dataset.get_words_with_sequences(
                        difficult_sequences,
                        difficulty,
                        count=8
                    )
                    words.extend(seq_words)
            
            # Add words user commonly makes mistakes with (if they exist)
            if user_model.get('word_mistakes') and len(user_model['word_mistakes']) > 0:
                difficult_words = sorted(
                    user_model['word_mistakes'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Include the top difficult words (not too many to avoid frustration)
                for word, _ in difficult_words[:5]:
                    if word not in words:
                        words.append(word)
            
            # Add random words to fill out the test and make it less obvious we're targeting difficulties
            # Get at least 20 random words, or more if we don't have many difficult words yet
            random_word_count = max(20, 30 - len(words))
            
            # Get random words (potentially avoiding any we've already included)
            random_words = self.word_dataset.get_words_by_difficulty(difficulty, count=random_word_count)
            words.extend(random_words)
            
            # Ensure we have enough words and no duplicates
            words = list(set(words))
            
            # If we still don't have enough words, get more random ones
            if len(words) < 25:
                additional = self.word_dataset.get_words_by_difficulty(
                    difficulty, 
                    count=30 - len(words)
                )
                words.extend(additional)
                words = list(set(words))
            
            # If we still don't have enough words, fall back to standard text
            if len(words) < 10:
                return self._generate_standard_text(difficulty)
            
            # Shuffle to mix up the difficult and random words (so it's not obvious)
            random.shuffle(words)
            
            # Limit to a reasonable length for a typing test
            words = words[:30]
            
            # For analysis purposes, update session with words that focused on difficulties
            user_model['last_session_focused_on'] = {
                'characters': difficult_chars,
                'sequences': difficult_sequences,
                'words': [word for word, _ in difficult_words[:5]] if 'word_mistakes' in user_model else []
            }
            self._save_user_model(user_id, user_model)
            
            return ' '.join(words)
        except Exception as e:
            print(f"Error generating text: {e}")
            # First try to use the word dataset directly instead of falling back to a hardcoded text
            try:
                print("Attempting to fetch words directly from dataset...")
                words = self.word_dataset.get_words_by_difficulty(difficulty, count=30)
                if words and len(words) >= 10:
                    return ' '.join(words)
                else:
                    raise ValueError("Not enough words returned from dataset")
            except Exception as dataset_error:
                print(f"Dataset fallback also failed: {dataset_error}")
                # Last resort - create a more varied fallback text based on difficulty
                if difficulty == 'easy':
                    return "the and to of in a is for you it not with on as at but by from they we say her she"
                elif difficulty == 'hard':
                    return "algorithm functionality implementation infrastructure sophisticated technology development experience performance significant"
                else:  # medium
                    return "typing practice keyboard skills improve accuracy speed fingers position technique learning focus"
    
    def _generate_standard_text(self, difficulty):
        """Generate a standard typing test text based on difficulty."""
        try:
            # Use our new dataset to get random words based on difficulty
            words = self.word_dataset.get_words_by_difficulty(difficulty, count=30)
            
            # If we couldn't get enough words from the dataset, fall back to common words
            if len(words) < 20:
                word_count = 30  # Default length for a typing test
                
                # Select words based on difficulty
                if difficulty == 'easy':
                    if len(self.common_words['easy']) > 0:
                        words = random.sample(self.common_words['easy'], 
                                             min(word_count, len(self.common_words['easy'])))
                        # Repeat sampling if needed to reach word_count
                        while len(words) < word_count and len(self.common_words['easy']) > 0:
                            more_words = random.sample(self.common_words['easy'], 
                                                      min(word_count - len(words), len(self.common_words['easy'])))
                            words.extend(more_words)
                    else:
                        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
                elif difficulty == 'medium':
                    # Mix of easy and medium words
                    easy_words = self.common_words['easy'] if len(self.common_words['easy']) > 0 else ["the", "is", "and", "to"]
                    medium_words = self.common_words['medium'] if len(self.common_words['medium']) > 0 else ["typing", "practice", "keyboard", "skills"]
                    
                    easy = random.sample(easy_words, min(word_count // 3, len(easy_words)))
                    medium = random.sample(medium_words, min(word_count - len(easy), len(medium_words)))
                    words = easy + medium
                    
                    # Ensure we have enough words
                    while len(words) < word_count and len(medium_words) > 0:
                        more_words = random.sample(medium_words, 
                                                  min(word_count - len(words), len(medium_words)))
                        words.extend(more_words)
                else:  # hard
                    # Mix of medium and hard words
                    medium_words = self.common_words['medium'] if len(self.common_words['medium']) > 0 else ["typing", "practice", "keyboard", "skills"]
                    hard_words = self.common_words['hard'] if len(self.common_words['hard']) > 0 else ["algorithm", "sophisticated", "productivity", "efficiency"]
                    
                    medium = random.sample(medium_words, min(word_count // 3, len(medium_words)))
                    hard = random.sample(hard_words, min(word_count - len(medium), len(hard_words)))
                    words = medium + hard
                    
                    # Ensure we have enough words
                    while len(words) < word_count and len(hard_words) > 0:
                        more_words = random.sample(hard_words, 
                                                  min(word_count - len(words), len(hard_words)))
                        words.extend(more_words)
            
            # Shuffle words
            random.shuffle(words)
            
            return ' '.join(words)
        except Exception as e:
            print(f"Error generating standard text: {e}")
            # Try directly using the word dataset with no fancy logic
            try:
                print("Direct fallback to word dataset...")
                # Force a direct call to the word dataset's get_words_by_difficulty
                # This should always work due to the improved fallback mechanisms in WordDataset
                simple_words = self.word_dataset.get_words_by_difficulty(difficulty, count=25)
                if simple_words and len(simple_words) > 0:
                    return ' '.join(simple_words)
                else:
                    raise ValueError("WordDataset returned empty list")
            except Exception as dataset_error:
                print(f"Direct dataset fallback failed: {dataset_error}")
                # Absolutely last resort - different texts by difficulty
                if difficulty == 'easy':
                    return "short easy words for basic typing practice to improve your skills"
                elif difficulty == 'hard':
                    return "sophisticated vocabulary enhances comprehensive writing capabilities demonstrating professional communication abilities"
                else:  # medium
                    return "practice makes perfect when learning to type efficiently and accurately on a keyboard"
    
    def update(self, user_id, original_text, typed_text, mistakes):
        """Update the model with new typing data."""
        user_model = self._load_user_model(user_id)
        
        # Update session count
        user_model['session_count'] += 1
        
        # Process character mistakes
        self._update_character_mistakes(user_model, original_text, typed_text, mistakes)
        
        # Process word mistakes
        self._update_word_mistakes(user_model, original_text, typed_text)
        
        # Identify difficult typing sequences
        self._identify_difficult_sequences(user_model, mistakes, original_text)
        
        # Cluster mistakes if we have enough data
        if user_model['session_count'] >= 5:
            self._cluster_mistakes(user_model, mistakes)
        
        # Save updated model
        self._save_user_model(user_id, user_model)
        
        # Update word bank with new words
        self._update_word_bank(original_text)
    
    def _update_character_mistakes(self, user_model, original_text, typed_text, mistakes):
        """Update character mistake statistics."""
        # Initialize character mistakes dict if needed
        if 'character_mistakes' not in user_model:
            user_model['character_mistakes'] = {}
        
        # Analyze character-level mistakes
        for char in set(original_text):
            if char == ' ':  # Skip spaces
                continue
                
            # Count occurrences of this character in mistakes
            mistake_count = sum(1 for m in mistakes if original_text[m] == char)
            
            # Update mistake count for this character
            if char in user_model['character_mistakes']:
                user_model['character_mistakes'][char] += mistake_count
            else:
                user_model['character_mistakes'][char] = mistake_count
    
    def _update_word_mistakes(self, user_model, original_text, typed_text):
        """Update word mistake statistics."""
        # Initialize word mistakes dict if needed
        if 'word_mistakes' not in user_model:
            user_model['word_mistakes'] = {}
        
        # Split texts into words - use regex to get proper word boundaries
        original_words = re.findall(r'\b\w+\b', original_text.lower())
        typed_words = re.findall(r'\b\w+\b', typed_text.lower())
        
        # Compare words
        min_len = min(len(original_words), len(typed_words))
        for i in range(min_len):
            if original_words[i] != typed_words[i]:
                word = original_words[i]
                # Count this as a mistake on this word
                if word in user_model['word_mistakes']:
                    user_model['word_mistakes'][word] += 1
                else:
                    user_model['word_mistakes'][word] = 1
                
                # Also add this word to our word bank for future tests
                # But only if it's an actual word and not gibberish
                if len(word) > 2:
                    self._update_word_bank(word)
    
    def _identify_difficult_sequences(self, user_model, mistakes, original_text):
        """Identify difficult typing sequences (e.g., 'th', 'ing')."""
        if 'difficult_sequences' not in user_model:
            user_model['difficult_sequences'] = []
        
        # Look for patterns in mistakes
        sequences = []
        for mistake_pos in mistakes:
            if mistake_pos > 0 and mistake_pos < len(original_text) - 1:
                # Get the character and its context (preceding and following char)
                seq = original_text[mistake_pos-1:mistake_pos+2]
                if not re.search(r'\s', seq):  # Ignore sequences with spaces
                    sequences.append(seq)
        
        # Count occurrences of each sequence
        if sequences:
            counter = Counter(sequences)
            common_sequences = [seq for seq, count in counter.most_common(5) if count > 1]
            
            # Update difficult sequences
            user_model['difficult_sequences'] = list(
                set(user_model['difficult_sequences'] + common_sequences)
            )[:10]  # Keep top 10 difficult sequences
    
    def _cluster_mistakes(self, user_model, mistakes):
        """Use KMeans to cluster mistakes and identify patterns."""
        if not mistakes or len(mistakes) < 5:
            return
        
        # Convert mistakes to feature vectors (just positions for now)
        X = np.array(mistakes).reshape(-1, 1)
        
        # Determine number of clusters (at most 3)
        n_clusters = min(3, len(mistakes) // 2)
        if n_clusters < 2:
            return
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        
        # Store clusters in user model
        user_model['mistake_clusters'] = {
            'centers': kmeans.cluster_centers_.tolist(),
            'labels': kmeans.labels_.tolist()
        }
    
    def _update_word_bank(self, text):
        """Update word bank with new words from typing tests."""
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text))  # Only include words with 3+ letters
        
        # Categorize words by difficulty
        for word in words:
            # Simple difficulty classification based on word length
            if len(word) <= 4:
                if word not in self.word_bank['easy']:
                    self.word_bank['easy'].append(word)
            elif len(word) <= 7:
                if word not in self.word_bank['medium']:
                    self.word_bank['medium'].append(word)
            else:
                if word not in self.word_bank['hard']:
                    self.word_bank['hard'].append(word)
        
        # Save updated word bank
        with open(self.word_bank_file, 'w') as f:
            json.dump(self.word_bank, f)
