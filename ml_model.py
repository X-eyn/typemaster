import numpy as np
import os
import json
import random
from collections import Counter
from sklearn.cluster import KMeans
import joblib
import re

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
        
        # Load or create word bank
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
            return joblib.load(model_path)
        
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
        user_model = self._load_user_model(user_id)
        
        # For new users or after few sessions, use standard texts
        if user_model['session_count'] < 3:
            return self._generate_standard_text(difficulty)
        
        # Generate personalized text based on user's mistakes and patterns
        words = []
        
        # Get words with difficult characters for this user
        difficult_chars = sorted(user_model['character_mistakes'].items(), 
                                key=lambda x: x[1], reverse=True)
        
        # Include some words with user's difficult characters
        for char, _ in difficult_chars[:5]:
            candidates = [w for w in self.word_bank[difficulty] if char in w]
            if candidates:
                words.extend(random.sample(candidates, min(2, len(candidates))))
        
        # Add some words from difficult sequences
        if user_model['difficult_sequences']:
            for seq in user_model['difficult_sequences'][:3]:
                # Find words containing this sequence
                candidates = [w for w in self.word_bank[difficulty] if seq in w]
                if candidates:
                    words.extend(random.sample(candidates, min(2, len(candidates))))
        
        # Add some random words from the appropriate difficulty
        random_words = random.sample(self.word_bank[difficulty], 
                                     min(20, len(self.word_bank[difficulty])))
        words.extend(random_words)
        
        # Ensure we have enough words and no duplicates
        words = list(set(words))
        if len(words) < 25:
            additional = random.sample(self.word_bank[difficulty], 
                                      min(25 - len(words), len(self.word_bank[difficulty])))
            words.extend(additional)
            words = list(set(words))
        
        # Shuffle and limit to a reasonable length
        random.shuffle(words)
        words = words[:30]  # Limit to 30 words for a reasonable typing test
        
        return ' '.join(words)
    
    def _generate_standard_text(self, difficulty):
        """Generate a standard typing test text based on difficulty."""
        word_count = 30  # Default length for a typing test
        
        # Select words based on difficulty
        if difficulty == 'easy':
            words = random.sample(self.common_words['easy'], 
                                 min(word_count, len(self.common_words['easy'])))
            # Repeat sampling if needed to reach word_count
            while len(words) < word_count:
                more_words = random.sample(self.common_words['easy'], 
                                          min(word_count - len(words), len(self.common_words['easy'])))
                words.extend(more_words)
        elif difficulty == 'medium':
            # Mix of easy and medium words
            easy = random.sample(self.common_words['easy'], word_count // 3)
            medium = random.sample(self.common_words['medium'], 
                                  min(word_count - len(easy), len(self.common_words['medium'])))
            words = easy + medium
            # Ensure we have enough words
            while len(words) < word_count:
                more_words = random.sample(self.common_words['medium'], 
                                          min(word_count - len(words), len(self.common_words['medium'])))
                words.extend(more_words)
        else:  # hard
            # Mix of medium and hard words
            medium = random.sample(self.common_words['medium'], word_count // 3)
            hard = random.sample(self.common_words['hard'], 
                                min(word_count - len(medium), len(self.common_words['hard'])))
            words = medium + hard
            # Ensure we have enough words
            while len(words) < word_count:
                more_words = random.sample(self.common_words['hard'], 
                                          min(word_count - len(words), len(self.common_words['hard'])))
                words.extend(more_words)
        
        # Shuffle words
        random.shuffle(words)
        
        return ' '.join(words)
    
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
        
        # Split texts into words
        original_words = original_text.split()
        typed_words = typed_text.split()
        
        # Compare words
        for i in range(min(len(original_words), len(typed_words))):
            if original_words[i] != typed_words[i]:
                word = original_words[i]
                if word in user_model['word_mistakes']:
                    user_model['word_mistakes'][word] += 1
                else:
                    user_model['word_mistakes'][word] = 1
    
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
