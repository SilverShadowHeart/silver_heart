const chatLog = document.getElementById('chat-log');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const moodBarsContainer = document.getElementById('mood-bars-container');

// --- MOOD VISUALIZATION MAPPING ---
const moodConfig = {
    // Defines colors and order for the dashboard
    "joy": { color: "#ffd700" },
    "focus": { color: "#ffffff" },
    "anger": { color: "#ff4136" },
    "sad": { color: "#0074d9" },
    "cold": { color: "#7fdbff" },
    "support": { color: "#2ecc40" },
    "sarcasm": { color: "#b10dc9" },
    "anxiety": { color: "#ff851b" },
    "fatigue": { color: "#aaaaaa" },
    "romantic": { color: "#f012be" },
    "suspicion": { color: "#ffdc00" },
    "inspiration": { color: "#39cccc" }
};

let currentSessionId = localStorage.getItem('silverSessionId');
console.log(currentSessionId ? `Loaded existing session ID: ${currentSessionId}` : "No existing session ID.");

function initializeMoodDashboard() {
    for (const [mood, config] of Object.entries(moodConfig)) {
        const moodBar = document.createElement('div');
        moodBar.classList.add('mood-bar');
        moodBar.id = `mood-${mood}`;

        moodBar.innerHTML = `
            <div class="mood-label">${mood}</div>
            <div class="mood-indicator">
                <div class="mood-circle"></div>
                <div class="mood-progress-bg">
                    <div class="mood-progress-fg"></div>
                </div>
            </div>
        `;
        moodBarsContainer.appendChild(moodBar);
    }
}

function updateMoodVisuals(moodVector) {
    if (!moodVector) return;

    for (const [mood, score] of Object.entries(moodVector)) {
        const moodBar = document.getElementById(`mood-${mood}`);
        if (!moodBar) continue;

        const circle = moodBar.querySelector('.mood-circle');
        const progressBar = moodBar.querySelector('.mood-progress-fg');
        const config = moodConfig[mood];
        
        const isActive = score > 0.05;
        const color = isActive ? config.color : '#333';
        const progressColor = isActive ? config.color : '#555';

        circle.style.backgroundColor = color;
        circle.style.boxShadow = isActive ? `0 0 8px ${color}` : 'none';
        progressBar.style.width = `${score * 100}%`;
        progressBar.style.backgroundColor = progressColor;
    }
}

function sendMessage() {
    const messageText = userInput.value.trim();
    if (messageText === '') return;

    const initialMessage = document.querySelector('.initial-message');
    if (initialMessage) initialMessage.remove();

    appendMessage('user', messageText);
    userInput.value = '';
    userInput.focus();

    fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText, session_id: currentSessionId }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.session_id && data.session_id !== currentSessionId) {
            currentSessionId = data.session_id;
            localStorage.setItem('silverSessionId', currentSessionId);
            console.log(`Updated session ID to: ${currentSessionId}`);
        }
        appendMessage('ai', data.reply);
        updateMoodVisuals(data.mood_vector);
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage('ai', 'System error. The brain is... disconnected.');
        // Create an error state for the dashboard
        const errorVector = Object.keys(moodConfig).reduce((acc, key) => ({ ...acc, [key]: 0 }), {});
        errorVector.anger = 0.9;
        updateMoodVisuals(errorVector);
    });
}

function appendMessage(sender, text) {
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender === 'user' ? 'user-message' : 'ai-message');
    messageElement.innerText = text;
    chatLog.appendChild(messageElement);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// Initialize the UI on page load
initializeMoodDashboard();
// Set a default neutral state for the dashboard
updateMoodVisuals(Object.keys(moodConfig).reduce((acc, key) => ({ ...acc, [key]: 0 }), {}));

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => e.key === 'Enter' && sendMessage());