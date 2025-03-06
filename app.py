from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
from ml_model import TypingModel
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'typemaster_secret_key'

# Initialize ML model
model = TypingModel()

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_text', methods=['GET'])
def get_text():
    """Return a new text for typing test based on user's history"""
    user_id = request.args.get('user_id', 'anonymous')
    difficulty = request.args.get('difficulty', 'medium')
    
    # Get personalized text based on user's history if available
    try:
        text = model.generate_text(user_id, difficulty)
        
        # Ensure text is a proper string
        if not text or not isinstance(text, str):
            text = "The quick brown fox jumps over the lazy dog. This is a simple typing test."
        
        return jsonify({
            'text': text,
            'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        })
    except Exception as e:
        app.logger.error(f"Error generating text: {e}")
        return jsonify({
            'text': "The quick brown fox jumps over the lazy dog. This is a simple typing test.",
            'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            'error': str(e)
        })

@app.route('/api/submit_result', methods=['POST'])
def submit_result():
    """Submit typing test results and update the model"""
    data = request.json
    
    # Extract data
    user_id = data.get('user_id', 'anonymous')
    text = data.get('text', '')
    typed_text = data.get('typed_text', '')
    time_taken = data.get('time_taken', 0)
    mistakes = data.get('mistakes', [])
    
    # Calculate metrics
    wpm = calculate_wpm(text, time_taken)
    accuracy = calculate_accuracy(text, typed_text)
    
    # Update model with new data
    model.update(user_id, text, typed_text, mistakes)
    
    # Save typing session data
    save_session_data(user_id, {
        'text': text,
        'typed_text': typed_text,
        'time_taken': time_taken,
        'wpm': wpm,
        'accuracy': accuracy,
        'mistakes': mistakes,
        'timestamp': str(datetime.now().isoformat())
    })
    
    return jsonify({
        'success': True,
        'wpm': wpm,
        'accuracy': accuracy
    })

@app.route('/api/get_stats', methods=['GET'])
def get_stats():
    """Get user's typing statistics"""
    user_id = request.args.get('user_id', 'anonymous')
    stats = get_user_stats(user_id)
    
    return jsonify(stats)

def calculate_wpm(text, time_taken_ms):
    """Calculate words per minute"""
    word_count = len(text.split())
    minutes = time_taken_ms / (1000 * 60)
    return round(word_count / minutes, 2) if minutes > 0 else 0

def calculate_accuracy(original, typed):
    """Calculate typing accuracy"""
    if not original or not typed:
        return 0
    
    # Simple character-by-character accuracy
    min_len = min(len(original), len(typed))
    correct = sum(1 for i in range(min_len) if original[i] == typed[i])
    return round((correct / len(original)) * 100, 2)

def save_session_data(user_id, data):
    """Save typing session data to a file"""
    user_file = f'data/{user_id}_sessions.json'
    
    # Load existing data if any
    if os.path.exists(user_file):
        with open(user_file, 'r') as f:
            sessions = json.load(f)
    else:
        sessions = []
    
    # Add new session
    sessions.append(data)
    
    # Save updated data
    with open(user_file, 'w') as f:
        json.dump(sessions, f)

def get_user_stats(user_id):
    """Get user statistics from saved sessions"""
    user_file = f'data/{user_id}_sessions.json'
    
    if not os.path.exists(user_file):
        return {
            'sessions': 0,
            'avg_wpm': 0,
            'avg_accuracy': 0,
            'common_mistakes': [],
            'improvement': 0
        }
    
    with open(user_file, 'r') as f:
        sessions = json.load(f)
    
    if not sessions:
        return {
            'sessions': 0,
            'avg_wpm': 0,
            'avg_accuracy': 0,
            'common_mistakes': [],
            'improvement': 0
        }
    
    # Calculate average metrics
    avg_wpm = sum(s.get('wpm', 0) for s in sessions) / len(sessions)
    avg_accuracy = sum(s.get('accuracy', 0) for s in sessions) / len(sessions)
    
    # Get common mistakes
    all_mistakes = []
    for s in sessions:
        all_mistakes.extend(s.get('mistakes', []))
    
    mistake_counts = {}
    for m in all_mistakes:
        if m in mistake_counts:
            mistake_counts[m] += 1
        else:
            mistake_counts[m] = 1
    
    common_mistakes = sorted(mistake_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Calculate improvement (comparing first and last 3 sessions)
    if len(sessions) >= 6:
        first_three = sessions[:3]
        last_three = sessions[-3:]
        
        first_wpm = sum(s.get('wpm', 0) for s in first_three) / 3
        last_wpm = sum(s.get('wpm', 0) for s in last_three) / 3
        
        improvement = round(((last_wpm - first_wpm) / first_wpm) * 100, 2) if first_wpm > 0 else 0
    else:
        improvement = 0
    
    return {
        'sessions': len(sessions),
        'avg_wpm': round(avg_wpm, 2),
        'avg_accuracy': round(avg_accuracy, 2),
        'common_mistakes': [{'char': m[0], 'count': m[1]} for m in common_mistakes],
        'improvement': improvement
    }

if __name__ == '__main__':
    app.run(debug=True)
