// Get references to the form and messages container
const form = document.getElementById('frm');
const messages = document.getElementById('messages');

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent default form behavior

    const q = document.getElementById('q').value; // Get user input
    if (!q) return; // Ignore empty input

    appendBubble(q, 'user'); // Show user's message
    document.getElementById('q').value = ''; // Clear input field

    // Prepare form data for POST request
    const fd = new FormData();
    fd.append('q', q);

    // Send user input to server and get response
    const res = await fetch('/chat', { method: 'POST', body: fd });
    const j = await res.json();

    // Show bot's response and sources
    appendBubble(
        j.answer + '\n\nSources: ' + (j.sources || []).join(', '),
        'bot'
    );
});

/**
 * Appends a chat bubble to the messages container.
 * @param {string} text - The message text.
 * @param {string} cls - The bubble class ('user' or 'bot').
 */
function appendBubble(text, cls) {
    const d = document.createElement('div');
    d.className = 'bubble ' + cls;
    d.innerText = text;
    messages.appendChild(d);

    // Scroll to the bottom of the page
    window.scrollTo(0, document.body.scrollHeight);
}