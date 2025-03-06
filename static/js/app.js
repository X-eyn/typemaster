document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const typingTextElement = document.getElementById('typing-text');
    const inputField = document.getElementById('input-field');
    const cursor = document.querySelector('.cursor');
    const timerElement = document.getElementById('timer');
    const wpmElement = document.getElementById('wpm');
    const accuracyElement = document.getElementById('accuracy');
    const errorsElement = document.getElementById('errors');
    const startBtn = document.getElementById('start-btn');
    const restartBtn = document.getElementById('restart-test-btn');
    const difficultyBtns = document.querySelectorAll('.difficulty-btn');
    const resultsScreen = document.getElementById('results-screen');
    const statsScreen = document.getElementById('stats-screen');
    const resultWpmElement = document.getElementById('result-wpm');
    const resultAccuracyElement = document.getElementById('result-accuracy');
    const resultTimeElement = document.getElementById('result-time');
    const mistakeCharsElement = document.getElementById('mistake-chars');
    const restartResultBtn = document.getElementById('restart-btn');
    const viewStatsBtn = document.getElementById('view-stats-btn');
    const backBtn = document.getElementById('back-btn');
    const commonMistakesList = document.getElementById('common-mistakes-list');
    const improvementValue = document.getElementById('improvement-value');

    // Variables
    let time;
    let timer;
    let charIndex = 0;
    let mistakes = 0;
    let isTyping = false;
    let testStarted = false;
    let testCompleted = false;
    let currentText = '';
    let typedText = '';
    let mistakePositions = [];
    let mistakeChars = {};
    let timeStart;
    let difficulty = 'medium';
    let userId = localStorage.getItem('user_id') || generateUserId();

    // Save user ID to localStorage
    function generateUserId() {
        const id = 'user_' + Math.random().toString(36).substring(2, 15);
        localStorage.setItem('user_id', id);
        return id;
    }

    // Initialization
    function init() {
        // Set focus to input field
        inputField.focus();

        // Add event listeners
        inputField.addEventListener('input', handleTyping);
        startBtn.addEventListener('click', startTest);
        restartBtn.addEventListener('click', resetTest);
        restartResultBtn.addEventListener('click', resetTest);
        viewStatsBtn.addEventListener('click', showStats);
        backBtn.addEventListener('click', hideStats);
        
        // Difficulty buttons
        difficultyBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active class from all buttons
                difficultyBtns.forEach(b => b.classList.remove('active'));
                // Add active class to clicked button
                btn.classList.add('active');
                // Set difficulty
                difficulty = btn.dataset.difficulty;
                
                if (!testStarted || testCompleted) {
                    // If test hasn't started or is completed, load new text
                    loadNewText();
                }
            });
        });

        // Initial text load
        loadNewText();

        // Handle input field focus
        document.addEventListener('click', () => {
            if (testStarted && !testCompleted) {
                inputField.focus();
            }
        });

        // Prevent tab key from moving focus
        inputField.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
            }
        });
    }

    // Load new typing text from server
    function loadNewText() {
        // Show loading state
        typingTextElement.innerHTML = 'Loading...';
        
        // Fetch new text based on difficulty and user ID
        fetch(`/api/get_text?user_id=${userId}&difficulty=${difficulty}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (!data || !data.text) {
                    throw new Error('Invalid response data');
                }
                currentText = data.text;
                renderText();
                resetTestState();
            })
            .catch(error => {
                console.error('Error fetching text:', error);
                // Fallback text for testing
                currentText = "The quick brown fox jumps over the lazy dog. This is a simple typing test to measure your speed and accuracy.";
                renderText();
                resetTestState();
            });
    }

    // Render text with character spans
    function renderText() {
        typingTextElement.innerHTML = '';
        // Split text into characters and create spans
        currentText.split('').forEach(char => {
            const span = document.createElement('span');
            span.textContent = char;
            typingTextElement.appendChild(span);
        });
        
        // Set initial active character
        if (typingTextElement.children.length > 0) {
            typingTextElement.children[0].classList.add('active');
            // Position cursor at first character
            updateCursorPosition(0);
        }
    }

    // Handle typing input
    function handleTyping(e) {
        if (!testStarted && !testCompleted) {
            // Start the test on first keystroke
            startTest();
        }
        
        if (testCompleted) return;
        
        const characters = typingTextElement.querySelectorAll('span');
        const typedChar = e.data;
        
        // Only process if there's input and we haven't reached the end
        if (typedChar !== null && charIndex < characters.length) {
            // Add typed character to the typedText for submission
            typedText += typedChar;
            
            // Check if character is correct
            if (typedChar === currentText[charIndex]) {
                characters[charIndex].classList.add('correct');
                characters[charIndex].classList.remove('incorrect', 'active');
            } else {
                characters[charIndex].classList.add('incorrect');
                characters[charIndex].classList.remove('correct', 'active');
                mistakes++;
                mistakePositions.push(charIndex);
                
                // Track mistake characters for analysis
                if (mistakeChars[currentText[charIndex]]) {
                    mistakeChars[currentText[charIndex]]++;
                } else {
                    mistakeChars[currentText[charIndex]] = 1;
                }
            }
            
            // Move to next character
            charIndex++;
            
            // Mark next character as active if not at the end
            if (charIndex < characters.length) {
                characters[charIndex].classList.add('active');
                // Update cursor position
                updateCursorPosition(charIndex);
            } else {
                // Test complete
                completeTest();
            }
            
            // Update stats in real time
            updateStats();
        } else if (e.inputType === 'deleteContentBackward' && charIndex > 0) {
            // Handle backspace
            charIndex--;
            typedText = typedText.slice(0, -1);
            
            // Reset class for current character
            characters[charIndex].classList.remove('correct', 'incorrect');
            characters[charIndex].classList.add('active');
            
            // Reset class for previous character
            if (charIndex < characters.length - 1) {
                characters[charIndex + 1].classList.remove('active');
            }
            
            // If the deleted char was a mistake, remove it
            if (characters[charIndex].classList.contains('incorrect')) {
                mistakes--;
                const index = mistakePositions.indexOf(charIndex);
                if (index > -1) {
                    mistakePositions.splice(index, 1);
                }
                
                // Update mistake character count
                if (mistakeChars[currentText[charIndex]] > 1) {
                    mistakeChars[currentText[charIndex]]--;
                } else {
                    delete mistakeChars[currentText[charIndex]];
                }
            }
            
            // Update cursor position
            updateCursorPosition(charIndex);
            
            // Update stats in real time
            updateStats();
        }
    }

    // Start the test
    function startTest() {
        testStarted = true;
        isTyping = true;
        timeStart = new Date().getTime();
        
        // Reset stats
        mistakes = 0;
        charIndex = 0;
        mistakePositions = [];
        mistakeChars = {};
        typedText = '';
        
        // Focus input field
        inputField.focus();
        
        // Start timer
        startTimer();
        
        // Update UI
        startBtn.style.display = 'none';
        restartBtn.style.display = 'block';
    }

    // Reset test
    function resetTest() {
        // Stop timer
        clearInterval(timer);
        
        // Reset variables
        charIndex = 0;
        mistakes = 0;
        isTyping = false;
        testStarted = false;
        testCompleted = false;
        mistakePositions = [];
        mistakeChars = {};
        typedText = '';
        time = 0;
        
        // Update UI
        timerElement.innerText = '00:00';
        wpmElement.innerText = '0';
        accuracyElement.innerText = '100%';
        errorsElement.innerText = '0';
        startBtn.style.display = 'block';
        restartBtn.style.display = 'none';
        resultsScreen.classList.remove('active');
        
        // Clear input field
        inputField.value = '';
        
        // Load new text
        loadNewText();
        
        // Focus input field
        inputField.focus();
    }

    // Complete the test
    function completeTest() {
        testCompleted = true;
        isTyping = false;
        clearInterval(timer);
        
        // Calculate final stats
        const timeInSeconds = (new Date().getTime() - timeStart) / 1000;
        const wpm = calculateWPM(currentText, timeInSeconds);
        const accuracy = calculateAccuracy();
        
        // Submit results to server
        submitResults(wpm, accuracy, timeInSeconds);
        
        // Show results screen
        showResults(wpm, accuracy, timeInSeconds);
    }

    // Show results screen
    function showResults(wpm, accuracy, timeInSeconds) {
        // Update result elements
        resultWpmElement.innerText = wpm;
        resultAccuracyElement.innerText = `${accuracy}%`;
        resultTimeElement.innerText = `${timeInSeconds.toFixed(1)}s`;
        
        // Display mistake analysis
        displayMistakeAnalysis();
        
        // Show results screen
        resultsScreen.classList.add('active');
    }

    // Display mistake analysis
    function displayMistakeAnalysis() {
        mistakeCharsElement.innerHTML = '';
        
        // Sort mistakes by count
        const sortedMistakes = Object.entries(mistakeChars)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5); // Show top 5 mistakes
        
        if (sortedMistakes.length === 0) {
            const perfectElement = document.createElement('div');
            perfectElement.className = 'perfect-score';
            perfectElement.innerText = 'Perfect! No mistakes.';
            mistakeCharsElement.appendChild(perfectElement);
            return;
        }
        
        // Create elements for each mistake
        sortedMistakes.forEach(([char, count]) => {
            const mistakeElement = document.createElement('div');
            mistakeElement.className = 'mistake-char';
            
            const charElement = document.createElement('div');
            charElement.className = 'char';
            charElement.innerText = char === ' ' ? 'SPACE' : char;
            
            const countElement = document.createElement('div');
            countElement.className = 'count';
            countElement.innerText = `${count} times`;
            
            mistakeElement.appendChild(charElement);
            mistakeElement.appendChild(countElement);
            mistakeCharsElement.appendChild(mistakeElement);
        });
    }

    // Show stats screen
    function showStats() {
        resultsScreen.classList.remove('active');
        statsScreen.classList.add('active');
        
        // Fetch user stats
        fetchUserStats();
    }

    // Hide stats screen
    function hideStats() {
        statsScreen.classList.remove('active');
        resultsScreen.classList.add('active');
    }

    // Fetch user stats from server
    function fetchUserStats() {
        fetch(`/api/get_stats?user_id=${userId}`)
            .then(response => response.json())
            .then(data => {
                // Display common mistakes
                displayCommonMistakes(data.common_mistakes);
                
                // Display improvement
                displayImprovement(data.improvement);
                
                // Display progress chart if available
                if (data.sessions > 0) {
                    displayProgressChart(data);
                }
            })
            .catch(error => {
                console.error('Error fetching stats:', error);
                commonMistakesList.innerHTML = '<div class="no-data">No statistics available yet.</div>';
            });
    }

    // Display common mistakes
    function displayCommonMistakes(mistakes) {
        commonMistakesList.innerHTML = '';
        
        if (!mistakes || mistakes.length === 0) {
            commonMistakesList.innerHTML = '<div class="no-data">No common mistakes found.</div>';
            return;
        }
        
        // Create elements for each common mistake
        mistakes.forEach(mistake => {
            const mistakeElement = document.createElement('div');
            mistakeElement.className = 'mistake-char';
            
            const charElement = document.createElement('div');
            charElement.className = 'char';
            charElement.innerText = mistake.char === ' ' ? 'SPACE' : mistake.char;
            
            const countElement = document.createElement('div');
            countElement.className = 'count';
            countElement.innerText = `${mistake.count} times`;
            
            mistakeElement.appendChild(charElement);
            mistakeElement.appendChild(countElement);
            commonMistakesList.appendChild(mistakeElement);
        });
    }

    // Display improvement
    function displayImprovement(improvement) {
        if (improvement > 0) {
            improvementValue.innerText = `+${improvement}%`;
            improvementValue.style.color = 'var(--success)';
        } else if (improvement < 0) {
            improvementValue.innerText = `${improvement}%`;
            improvementValue.style.color = 'var(--error)';
        } else {
            improvementValue.innerText = '0%';
            improvementValue.style.color = 'var(--text-secondary)';
        }
    }

    // Display progress chart
    function displayProgressChart(data) {
        // If Chart.js is loaded
        if (typeof Chart !== 'undefined' && data.sessions > 0) {
            // Create dummy data for demonstration
            // In a real app, this would come from the backend
            const sessions = Array.from({ length: data.sessions }, (_, i) => i + 1);
            const wpmProgress = Array.from({ length: data.sessions }, (_, i) => {
                // Generate a trend that improves over time with some variation
                const baseWpm = 30;
                const improvement = i * 0.5;
                const variation = Math.sin(i) * 5;
                return baseWpm + improvement + variation;
            });
            
            const accuracyProgress = Array.from({ length: data.sessions }, (_, i) => {
                // Generate a trend that improves over time with some variation
                const baseAccuracy = 85;
                const improvement = i * 0.2;
                const variation = Math.cos(i) * 3;
                return Math.min(100, baseAccuracy + improvement + variation);
            });
            
            // Get canvas element
            const ctx = document.getElementById('progress-chart').getContext('2d');
            
            // Create chart
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: sessions,
                    datasets: [
                        {
                            label: 'WPM',
                            data: wpmProgress,
                            borderColor: '#3a86ff',
                            backgroundColor: 'rgba(58, 134, 255, 0.1)',
                            tension: 0.3,
                            fill: true
                        },
                        {
                            label: 'Accuracy (%)',
                            data: accuracyProgress,
                            borderColor: '#8338ec',
                            backgroundColor: 'rgba(131, 56, 236, 0.1)',
                            tension: 0.3,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Sessions',
                                color: 'rgba(255, 255, 255, 0.7)'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
        }
    }

    // Submit results to server
    function submitResults(wpm, accuracy, timeInSeconds) {
        const data = {
            user_id: userId,
            text: currentText,
            typed_text: typedText,
            time_taken: timeInSeconds * 1000, // Convert to milliseconds
            mistakes: mistakePositions,
            wpm: wpm,
            accuracy: accuracy
        };
        
        fetch('/api/submit_result', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Results submitted successfully:', data);
        })
        .catch(error => {
            console.error('Error submitting results:', error);
        });
    }

    // Start the timer
    function startTimer() {
        time = 0;
        timer = setInterval(() => {
            time++;
            let minutes = Math.floor(time / 60);
            let seconds = time % 60;
            
            // Format time as MM:SS
            minutes = minutes < 10 ? `0${minutes}` : minutes;
            seconds = seconds < 10 ? `0${seconds}` : seconds;
            
            timerElement.innerText = `${minutes}:${seconds}`;
        }, 1000);
    }

    // Update stats in real time
    function updateStats() {
        // Calculate WPM
        const timeInMinutes = (new Date().getTime() - timeStart) / (1000 * 60);
        const wordsTyped = charIndex / 5; // Standard: 5 characters = 1 word
        const currentWpm = Math.round(wordsTyped / timeInMinutes) || 0;
        
        // Calculate accuracy
        const currentAccuracy = calculateAccuracy();
        
        // Update DOM
        wpmElement.innerText = currentWpm;
        accuracyElement.innerText = `${currentAccuracy}%`;
        errorsElement.innerText = mistakes;
    }

    // Calculate WPM (Words Per Minute)
    function calculateWPM(text, timeInSeconds) {
        const words = text.length / 5; // Standard: 5 characters = 1 word
        const minutes = timeInSeconds / 60;
        return Math.round(words / minutes) || 0;
    }

    // Calculate accuracy
    function calculateAccuracy() {
        if (charIndex === 0) return 100;
        const correctChars = charIndex - mistakes;
        return Math.round((correctChars / charIndex) * 100);
    }

    // Update cursor position
    function updateCursorPosition(index) {
        if (index >= currentText.length) return;
        
        const charElement = typingTextElement.querySelectorAll('span')[index];
        if (!charElement) return;
        
        const charRect = charElement.getBoundingClientRect();
        const containerRect = typingTextElement.getBoundingClientRect();
        
        // Calculate new cursor position relative to typing container
        const left = charRect.left - containerRect.left;
        const top = charRect.top - containerRect.top;
        
        // Set cursor position with smooth transition
        cursor.style.transform = `translate(${left}px, ${top}px)`;
        
        // Ensure text is scrolled to keep cursor in view
        if (charRect.top > containerRect.bottom - 50) {
            typingTextElement.scrollTop += charRect.height * 2;
        }
    }

    // Initialize the app
    init();
});