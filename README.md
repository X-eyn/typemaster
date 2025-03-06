# TypeMaster

<div align="center">

![TypeMaster Logo](static/images/logo.png)

**An intelligent typing test application that adapts to your typing patterns and helps you improve where you need it most.**

</div>

## Overview

TypeMaster is a modern, adaptive typing test application designed to help users improve their typing speed and accuracy. Unlike traditional typing tests, TypeMaster incorporates a dynamic learning system that analyzes your typing patterns, identifies your weaknesses, and creates personalized tests to target those areas for improvement.

## Key Features

- **Adaptive Learning System**: Our custom ML model analyzes your typing patterns and creates personalized tests that focus on your specific weaknesses
- **Character-Level Analysis**: Identifies which specific characters you struggle with most frequently
- **Pattern Recognition**: Detects challenging letter combinations and sequences that slow you down
- **Difficulty Levels**: Choose from easy, medium, or hard tests to match your skill level
- **Detailed Performance Metrics**: Track your WPM (words per minute), accuracy percentage, and time taken
- **Responsive Design**: Beautiful, minimal UI with smooth animations that works on various screen sizes
- **Mistake Analysis**: Visual breakdown of your most common errors to focus your practice
- **Progression Tracking**: Monitor your improvement over time with persistent user profiles

## Installation

### Prerequisites
- Python 3.7+ installed on your system
- Git (optional, for cloning the repository)

### Steps

1. Clone this repository or download the source code:
   ```bash
   git clone https://github.com/yourusername/type-master.git
   cd type-master
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## How It Works

TypeMaster uses a hybrid statistical learning system that:

1. **Collects Data**: Every time you complete a typing test, TypeMaster records your performance data
2. **Analyzes Patterns**: The system identifies patterns in your mistakes across three categories:
   - Character-level mistakes (specific letters you mistype)
   - Word-level mistakes (complete words you struggle with)
   - Difficult sequences (combinations of 2-3 letters that give you trouble)
3. **Customizes Tests**: After 3+ sessions, the application begins generating personalized tests targeting your weak areas
4. **Adapts Over Time**: As your skills improve, the system continuously updates to focus on your current challenges

### The Learning Model

While not a traditional deep learning model, TypeMaster employs:

- **Statistical Pattern Analysis**: Tracks frequency and context of errors
- **K-means Clustering**: Groups similar mistakes to identify broader patterns (after 5+ sessions)
- **Dynamic Difficulty Adjustment**: Selects words containing your problem characters and sequences

## Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Python with Flask framework
- **Data Analysis**: NumPy, scikit-learn for K-means clustering
- **Persistent Storage**: JSON-based local data storage for user models and word datasets

## Usage

1. **Start a Test**: Click the "Start" button to begin a typing test
2. **Type the Text**: Type the displayed text as quickly and accurately as possible
3. **View Results**: After completing the test, review your performance metrics
4. **Track Progress**: Complete multiple tests to see improvement over time
5. **Adjust Difficulty**: Use the difficulty selector to change the challenge level

## Contributing

Contributions to TypeMaster are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

**TypeMaster** — Developed with ❤️ by the TypeMaster Team

</div>
