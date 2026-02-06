// MDIMG QA — Chat interface JS
function sendChat() {
    const input = document.getElementById("chatInput");
    const messages = document.getElementById("chatMessages");
    const placeholder = document.getElementById("chatPlaceholder");
    const sendBtn = document.getElementById("chatSend");
    const message = input.value.trim();

    if (!message || !RUN_ID) return;

    // Remove placeholder
    if (placeholder) placeholder.remove();

    // Append user message
    appendMessage("user", message);
    input.value = "";
    sendBtn.disabled = true;

    // Show typing indicator
    const typing = document.createElement("div");
    typing.id = "typingIndicator";
    typing.className = "mb-2";
    typing.innerHTML = '<span class="badge bg-secondary">assistant</span>' +
        '<p class="mb-0 mt-1 text-muted"><em>Thinking…</em></p>';
    messages.appendChild(typing);
    messages.scrollTop = messages.scrollHeight;

    fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: RUN_ID, message: message }),
    })
        .then((res) => {
            if (!res.ok) throw new Error("Chat request failed");
            return res.json();
        })
        .then((data) => {
            const t = document.getElementById("typingIndicator");
            if (t) t.remove();
            appendMessage("assistant", data.reply || "No response.");
        })
        .catch((err) => {
            const t = document.getElementById("typingIndicator");
            if (t) t.remove();
            appendMessage("assistant", "⚠️ Error: " + err.message);
        })
        .finally(() => {
            sendBtn.disabled = false;
            input.focus();
        });
}

function appendMessage(role, content) {
    const messages = document.getElementById("chatMessages");
    const div = document.createElement("div");
    div.className = "mb-2 " + (role === "user" ? "text-end" : "");
    div.innerHTML =
        '<span class="badge ' +
        (role === "user" ? "bg-primary" : "bg-secondary") +
        '">' + role + "</span>" +
        '<p class="mb-0 mt-1">' + escapeHtml(content) + "</p>";
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Allow Enter key to send
document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("chatInput");
    if (input) {
        input.addEventListener("keydown", function (e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendChat();
            }
        });
    }
});
