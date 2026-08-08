/* ==============================================================================
   DA-IICT Faculty AI Buddy - Chat Logic (chat.js)
   ============================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // Require authentication to view chat page
    const authData = JSON.parse(localStorage.getItem("dau_buddy_auth") || "null");
    if (!authData || !authData.credential) {
        window.location.href = '/?view=login';
        return;
    }

    // DOM Elements
    const form = document.getElementById("input-form");
    const userInput = document.getElementById("user-input");
    const viewport = document.getElementById("chat-viewport");
    const welcomeScreen = document.getElementById("welcome-screen");
    const messagesContainer = document.getElementById("messages-container");
    const themeToggle = document.getElementById("theme-toggle");
    const clearChatBtn = document.getElementById("clear-chat");
    const suggestedLinks = document.querySelectorAll(".suggested-link");
    const promptCards = document.querySelectorAll(".prompt-card");

    const sendBtn = document.getElementById("send-btn");
    const micBtn = document.getElementById("mic-btn");

    // Client-side request budget. Must cover worst case: Gemini timeout (30s)
    // + OpenAI fallback (45s) + overhead. Server answers before this fires.
    const CHAT_TIMEOUT_MS = 100000;
    // Turns posted back to the server; it re-caps this itself.
    const HISTORY_TURNS_SENT = 12;

    class HttpError extends Error {
        constructor(status) {
            super(`HTTP ${status}`);
            this.status = status;
        }
    }

    /**
     * Turn a fetch failure into something a user can act on. The previous
     * message blamed the database for every failure, including cases where the
     * request never left the browser.
     */
    function describeChatError(error) {
        if (error && error.name === "AbortError") {
            return "That request took too long and I stopped waiting. Please try again, or ask a narrower question.";
        }
        if (error instanceof HttpError) {
            if (error.status === 401) {
                return "Your session has expired or is invalid. Please log in again.";
            }
            if (error.status === 403) {
                return "This Google account isn't eligible.\nDAU Buddy is limited to @dau.ac.in\nand @daiict.ac.in accounts.";
            }
            if (error.status === 429) {
                return "You're sending messages faster than I can handle. Please wait a moment and try again.";
            }
            if (error.status === 413) {
                return "That message is too long for me to process. Try shortening it.";
            }
            if (error.status >= 500) {
                return "The server hit an error answering that. Please try again in a moment.";
            }
            return `The server rejected that request (${error.status}). Please try again.`;
        }
        // TypeError from fetch() — the request never completed: offline, or the
        // server dropped the connection / is restarting.
        return "I couldn't reach the server. Check your connection and try again — if this keeps happening, the service may be restarting.";
    }

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme");
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", newTheme);
            themeToggle.innerHTML = newTheme === "dark" ? '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
        });
    }



    let isResponding = false;

    function setInputState(enabled) {
        isResponding = !enabled;
        sendBtn.disabled = !enabled;
        if (enabled) {
            sendBtn.style.opacity = "1";
            sendBtn.style.cursor = "pointer";
            userInput.focus();
        } else {
            sendBtn.style.opacity = "0.5";
            sendBtn.style.cursor = "not-allowed";
        }
    }

    // New DOM Elements for Chat History
    const chatHistoryList = document.getElementById("chat-history-list");
    const newChatBtn = document.getElementById("new-chat-btn");
    const clearHistoryBtn = document.getElementById("clear-history-btn");

    // Initialize state
    let chatSessions = [];
    let activeChatId = null;

    // Helper: Premium Markdown to HTML Renderer using Marked.js
    function renderMarkdown(text) {
        if (!text) return "";
        try {
            return marked.parse(text);
        } catch (e) {
            console.error("Marked parsing failed, falling back to raw text", e);
            return text;
        }
    }

    // Toggle Theme (Dark / Light)
    if (themeToggle) themeToggle.addEventListener("click", () => {
        const currentTheme = document.body.getAttribute("data-theme");
        const newTheme = currentTheme === "light" ? "dark" : "light";
        document.body.setAttribute("data-theme", newTheme);

        // Update Theme Icon
        const icon = themeToggle.querySelector("i");
        if (newTheme === "light") {
            icon.className = "fa-solid fa-sun";
        } else {
            icon.className = "fa-solid fa-moon";
        }
    });

    // Helper to get user-specific storage key
    function getChatStorageKey() {
        let userKey = "guest";
        try {
            const authData = JSON.parse(localStorage.getItem("dau_buddy_auth"));
            if (authData && authData.email) {
                userKey = authData.email.split('@')[0];
            }
        } catch (e) { }

        const storageKey = "dau_buddy_chats_" + userKey;

        if (userKey !== "guest") {
            const legacyChats = localStorage.getItem("dau_buddy_chats");
            if (legacyChats && !localStorage.getItem(storageKey)) {
                localStorage.setItem(storageKey, legacyChats);
                localStorage.removeItem("dau_buddy_chats");
            }
        }

        return storageKey;
    }

    // Load chat history from localStorage
    function loadChatHistory() {
        const stored = localStorage.getItem(getChatStorageKey());
        if (stored) {
            try {
                chatSessions = JSON.parse(stored);
            } catch (e) {
                console.error("Error parsing stored chat sessions", e);
                chatSessions = [];
            }
        }

        // Select active chat or create a fresh one if empty
        if (chatSessions.length > 0) {
            activeChatId = chatSessions[0].id;
        } else {
            createNewChat();
        }

        renderChatHistoryList();
        loadActiveChat();
    }

    // Save chat history to localStorage
    function saveChatHistory() {
        localStorage.setItem(getChatStorageKey(), JSON.stringify(chatSessions));
    }

    // Create a new chat session
    function createNewChat() {
        // If there's already an active empty chat, just reuse it
        const currentActive = chatSessions.find(s => s.id === activeChatId);
        if (currentActive && currentActive.messages.length === 0) {
            return;
        }

        const newId = Date.now().toString();
        const newSession = {
            id: newId,
            title: "New Chat",
            messages: [],
            timestamp: Date.now()
        };

        chatSessions.unshift(newSession);
        activeChatId = newId;
        saveChatHistory();
        renderChatHistoryList();
        loadActiveChat();
        closeMobileSidebar();
    }

    // Render left sidebar chat history items
    function renderChatHistoryList() {
        chatHistoryList.innerHTML = "";

        if (chatSessions.length === 0) {
            const emptyEl = document.createElement("div");
            emptyEl.style.padding = "16px";
            emptyEl.style.textAlign = "center";
            emptyEl.style.color = "var(--text-muted)";
            emptyEl.style.fontSize = "12px";
            emptyEl.textContent = "No past chats";
            chatHistoryList.appendChild(emptyEl);
            return;
        }

        chatSessions.forEach(session => {
            const item = document.createElement("div");
            item.className = `chat-history-item${session.id === activeChatId ? " active" : ""}`;
            item.setAttribute("data-id", session.id);

            const mainDiv = document.createElement("div");
            mainDiv.className = "chat-item-main";

            const icon = document.createElement("i");
            icon.className = "fa-regular fa-message";

            const titleSpan = document.createElement("span");
            titleSpan.className = "chat-item-title";
            titleSpan.textContent = session.title || "New Chat";

            mainDiv.appendChild(icon);
            mainDiv.appendChild(titleSpan);

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "delete-chat-btn";
            deleteBtn.title = "Delete Chat";
            deleteBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';

            // Delete chat click handler
            deleteBtn.addEventListener("click", (e) => {
                e.stopPropagation(); // Avoid selecting the chat when deleting
                deleteChat(session.id);
            });

            // Select chat click handler
            item.addEventListener("click", () => {
                selectChat(session.id);
            });

            item.appendChild(mainDiv);
            item.appendChild(deleteBtn);
            chatHistoryList.appendChild(item);
        });
    }

    // Select a specific chat session
    function selectChat(id) {
        if (activeChatId === id) return;
        activeChatId = id;
        renderChatHistoryList();
        loadActiveChat();
        closeMobileSidebar();
    }

    // Load active chat messages into the viewport
    function loadActiveChat() {
        messagesContainer.innerHTML = "";

        const activeSession = chatSessions.find(s => s.id === activeChatId);
        if (!activeSession || activeSession.messages.length === 0) {
            welcomeScreen.style.display = "flex";
            welcomeScreen.style.flexDirection = "column";
            welcomeScreen.style.alignItems = "center";
            welcomeScreen.style.justifyContent = "center";
            return;
        }

        welcomeScreen.style.display = "none";
        activeSession.messages.forEach((msg, idx) => {
            appendMessageHTML(msg.sender, msg.text, idx);
        });
        scrollToBottom();
    }

    // Delete a specific chat session
    function deleteChat(id) {
        const index = chatSessions.findIndex(s => s.id === id);
        if (index === -1) return;

        chatSessions.splice(index, 1);
        saveChatHistory();

        if (activeChatId === id) {
            if (chatSessions.length > 0) {
                activeChatId = chatSessions[0].id;
            } else {
                activeChatId = null;
                createNewChat();
                return;
            }
        }

        renderChatHistoryList();
        loadActiveChat();
    }

    // Clear all history
    function clearAllHistory() {
        if (confirm("Are you sure you want to delete all chat history? This cannot be undone.")) {
            chatSessions = [];
            activeChatId = null;
            localStorage.removeItem(getChatStorageKey());
            createNewChat();
        }
    }

    // Clear Active Chat contents (from top right header action)
    if (clearChatBtn) clearChatBtn.addEventListener("click", () => {
        const activeSession = chatSessions.find(s => s.id === activeChatId);
        if (activeSession && activeSession.messages.length > 0) {
            if (confirm("Clear messages in this chat session?")) {
                activeSession.messages = [];
                activeSession.title = "New Chat";
                saveChatHistory();
                renderChatHistoryList();
                loadActiveChat();
            }
        }
    });

    // Submit user question
    async function handleSend(text) {
        // Stop mic immediately on any send attempt (before the empty-text guard)
        stopDictation();

        if (!text.trim() || isResponding) return;

        // Auto-stop dictation if active
        if (window.stopDictation) window.stopDictation();

        // Require authentication to chat
        const authData = JSON.parse(localStorage.getItem("dau_buddy_auth") || "null");
        if (!authData || !authData.credential) {
            if (window.openLoginModal) {
                window.openLoginModal();
            } else {
                window.location.href = '/?view=login';
            }
            return;
        }

        setInputState(false);

        // Hide welcome screen
        welcomeScreen.style.display = "none";

        // Get or create active session
        let activeSession = chatSessions.find(s => s.id === activeChatId);
        if (!activeSession) {
            createNewChat();
            activeSession = chatSessions.find(s => s.id === activeChatId);
        }

        // Generate title if it's the first message
        if (activeSession.messages.length === 0) {
            const shortTitle = text.length > 28 ? text.substring(0, 25) + "..." : text;
            activeSession.title = shortTitle;
            renderChatHistoryList();
        }

        // Append to state history and save
        activeSession.messages.push({ sender: "user", text: text });
        saveChatHistory();

        // 1. Render User Message
        appendMessageHTML("user", text, activeSession.messages.length - 1);
        userInput.value = "";
        scrollToBottom();

        // 2. Render AI Typing Indicator
        const typingIndicator = appendTypingIndicator();
        scrollToBottom();

        // 3. Perform Server API Call
        try {
            // We know authData and authData.credential exist from earlier check
            const currentAuthData = JSON.parse(localStorage.getItem("dau_buddy_auth"));

            // Abort client-side before the server's own deadline, so a stalled
            // request fails cleanly instead of hanging until the connection is reset.
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);

            let response;
            try {
                response = await fetch("/api/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${currentAuthData.credential}`
                    },
                    body: JSON.stringify({
                        message: text,
                        // Only the last few turns are sent; the server caps this again.
                        history: activeSession.messages.slice(-HISTORY_TURNS_SENT)
                    }),
                    signal: controller.signal
                });
            } finally {
                clearTimeout(timer);
            }

            if (!response.ok) {
                throw new HttpError(response.status);
            }

            const data = await response.json();

            // Remove typing indicator
            typingIndicator.remove();

            // Append to state history and save
            activeSession.messages.push({ sender: "ai", text: data.response });
            saveChatHistory();

            // Render AI response
            appendMessageHTML("ai", data.response, activeSession.messages.length - 1, true);
        } catch (error) {
            typingIndicator.remove();
            setInputState(true);

            if (error instanceof HttpError && error.status === 401) {
                if (window.openLoginModal) {
                    window.openLoginModal();
                } else {
                    window.location.href = '/?view=login';
                }
            }

            const errorMsg = `⚠️ ${describeChatError(error)}`;
            activeSession.messages.push({ sender: "ai", text: errorMsg });
            saveChatHistory();

            appendMessageHTML("ai", errorMsg, activeSession.messages.length - 1, true);
        }

        scrollToBottom();
    }

    // Append Message Row to Container (UI rendering only)
    function appendMessageHTML(sender, text, index, animate = false) {
        const row = document.createElement("div");
        row.className = `msg-row ${sender}`;
        if (sender === "user" && typeof index === "number") {
            row.setAttribute("data-idx", index);
        }

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerHTML = sender === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fas fa-graduation-cap"></i>';

        const wrapper = document.createElement("div");
        wrapper.className = "bubble-wrapper";

        const bubble = document.createElement("div");
        bubble.className = "bubble";

        if (sender === "user") {
            bubble.textContent = text;

            // Create user bubble actions (Copy and Edit buttons) below the bubble
            const actions = document.createElement("div");
            actions.className = "bubble-actions";

            const copyBtn = document.createElement("button");
            copyBtn.className = "bubble-action-btn copy-btn";
            copyBtn.title = "Copy prompt";
            copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
            copyBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                const copyText = () => {
                    copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
                    copyBtn.style.color = "#10b981";
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
                        copyBtn.style.color = "";
                    }, 2000);
                };
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(copyText);
                } else {
                    const textarea = document.createElement("textarea");
                    textarea.value = text;
                    textarea.style.position = "fixed";
                    document.body.appendChild(textarea);
                    textarea.select();
                    try {
                        document.execCommand("copy");
                        copyText();
                    } catch (err) {
                        console.error("Fallback copy failed", err);
                    }
                    document.body.removeChild(textarea);
                }
            });

            const editBtn = document.createElement("button");
            editBtn.className = "bubble-action-btn edit-btn";
            editBtn.title = "Edit prompt";
            editBtn.innerHTML = '<i class="fa-solid fa-pen-to-square"></i>';
            editBtn.addEventListener("click", (e) => {
                e.stopPropagation();

                // Switch bubble to Edit Mode (in-place text editing)
                bubble.innerHTML = "";

                const textarea = document.createElement("textarea");
                textarea.className = "edit-textarea";
                textarea.value = text;

                const btnContainer = document.createElement("div");
                btnContainer.className = "edit-bubble-actions";

                const cancelBtn = document.createElement("button");
                cancelBtn.className = "edit-action-btn cancel";
                cancelBtn.textContent = "Cancel";
                cancelBtn.addEventListener("click", (e2) => {
                    e2.stopPropagation();
                    loadActiveChat(); // Simply reload the active chat to restore state
                });

                const submitBtn = document.createElement("button");
                submitBtn.className = "edit-action-btn submit";
                submitBtn.textContent = "Save & Submit";
                submitBtn.addEventListener("click", async (e2) => {
                    e2.stopPropagation();
                    const newText = textarea.value.trim();
                    if (!newText) return;

                    const activeSession = chatSessions.find(s => s.id === activeChatId);
                    if (!activeSession) return;

                    // Slice session history to exclude this message and all subsequent messages
                    if (typeof index === "number") {
                        activeSession.messages = activeSession.messages.slice(0, index);
                    }

                    // Clear and re-render the chat window to remove subsequent messages
                    loadActiveChat();

                    // Triggers the standard sending pipeline with the newly edited prompt
                    handleSend(newText);
                });

                // Keydown listener to submit on Enter key (without Shift)
                textarea.addEventListener("keydown", (eKey) => {
                    if (eKey.key === "Enter" && !eKey.shiftKey) {
                        eKey.preventDefault();
                        submitBtn.click();
                    }
                });

                btnContainer.appendChild(cancelBtn);
                btnContainer.appendChild(submitBtn);

                bubble.appendChild(textarea);
                bubble.appendChild(btnContainer);

                // Focus textarea and position cursor at the end
                textarea.focus();
                textarea.setSelectionRange(textarea.value.length, textarea.value.length);
            });

            actions.appendChild(copyBtn);
            actions.appendChild(editBtn);

            wrapper.appendChild(bubble);
            wrapper.appendChild(actions);
        } else {
            // AI Message rendering with optional streaming animation
            if (animate) {
                // progressive streaming/typing effect
                let currentLength = 0;

                // Hide actions while typing
                const actions = document.createElement("div");
                actions.className = "bubble-actions";
                actions.style.opacity = "0";
                actions.style.pointerEvents = "none";
                actions.style.transition = "opacity 0.3s ease";

                const copyBtn = document.createElement("button");
                copyBtn.className = "bubble-action-btn copy-btn";
                copyBtn.title = "Copy response";
                copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
                copyBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const copyText = () => {
                        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
                        copyBtn.style.color = "#10b981";
                        setTimeout(() => {
                            copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
                            copyBtn.style.color = "";
                        }, 2000);
                    };
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        navigator.clipboard.writeText(text).then(copyText);
                    } else {
                        const textarea = document.createElement("textarea");
                        textarea.value = text;
                        textarea.style.position = "fixed";
                        document.body.appendChild(textarea);
                        textarea.select();
                        try {
                            document.execCommand("copy");
                            copyText();
                        } catch (err) {
                            console.error("Fallback copy failed", err);
                        }
                        document.body.removeChild(textarea);
                    }
                });

                actions.appendChild(copyBtn);
                wrapper.appendChild(bubble);
                wrapper.appendChild(actions);

                function streamText() {
                    if (currentLength < text.length) {
                        // Advance by a larger chunk of characters for a faster streaming effect
                        let increment = Math.floor(Math.random() * 3) + 2; // 6 to 10 chars
                        currentLength = Math.min(text.length, currentLength + increment);

                        const chunk = text.substring(0, currentLength);
                        bubble.innerHTML = renderMarkdown(chunk) + '<span class="typing-cursor"></span>';

                        scrollToBottom();

                        let delay = 8;
                        if (text[currentLength - 1] === '\n') {
                            delay = 35; // Shorter pause for newlines
                        }

                        setTimeout(streamText, delay);
                    } else {
                        // Finished typing
                        bubble.innerHTML = renderMarkdown(text);
                        actions.style.opacity = "";
                        actions.style.pointerEvents = "";
                        scrollToBottom();
                        setInputState(true);
                    }
                }

                setTimeout(streamText, 50);

            } else {
                bubble.innerHTML = renderMarkdown(text);

                // Create AI bubble actions (Copy response) below the bubble
                const actions = document.createElement("div");
                actions.className = "bubble-actions";

                const copyBtn = document.createElement("button");
                copyBtn.className = "bubble-action-btn copy-btn";
                copyBtn.title = "Copy response";
                copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
                copyBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const copyText = () => {
                        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
                        copyBtn.style.color = "#10b981";
                        setTimeout(() => {
                            copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
                            copyBtn.style.color = "";
                        }, 2000);
                    };
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        navigator.clipboard.writeText(text).then(copyText);
                    } else {
                        const textarea = document.createElement("textarea");
                        textarea.value = text;
                        textarea.style.position = "fixed";
                        document.body.appendChild(textarea);
                        textarea.select();
                        try {
                            document.execCommand("copy");
                            copyText();
                        } catch (err) {
                            console.error("Fallback copy failed", err);
                        }
                        document.body.removeChild(textarea);
                    }
                });

                actions.appendChild(copyBtn);
                wrapper.appendChild(bubble);
                wrapper.appendChild(actions);
            }
        }

        row.appendChild(avatar);
        row.appendChild(wrapper);
        messagesContainer.appendChild(row);

        if (!animate && sender === "ai") {
            setInputState(true);
        }
    }

    // Append Typing Indicator Row
    function appendTypingIndicator() {
        const row = document.createElement("div");
        row.className = "msg-row ai";

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerHTML = '<i class="fas fa-graduation-cap"></i>';

        const wrapper = document.createElement("div");
        wrapper.className = "bubble-wrapper";

        const bubble = document.createElement("div");
        bubble.className = "bubble";

        const dots = document.createElement("div");
        dots.className = "typing-dots";
        dots.innerHTML = "<span></span><span></span><span></span>";

        bubble.appendChild(dots);
        wrapper.appendChild(bubble);
        row.appendChild(avatar);
        row.appendChild(wrapper);
        messagesContainer.appendChild(row);

        return row;
    }

    // Smooth scroll chat to bottom
    function scrollToBottom() {
        viewport.scrollTo({
            top: viewport.scrollHeight,
            behavior: "smooth"
        });
    }

    // Form Event Listener
    // Voice Dictation (Web Speech API)
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition = null;
    let isRequestingPermission = false;
    let isListening = false;
    let permissionTimeout = null;
    let placeholderOverride = false;

    function stopDictation() {
        if (isListening && recognition) {
            recognition.abort(); // Force abort to prevent late audio processing from appending text after send
            stopListening(); // Eagerly reset UI
        }
    }

    if (SpeechRecognition && micBtn) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;

        // Set the primary listening engine. (Web Speech API only accepts one active lang code per session)
        // Note: Chrome's engine is smart enough to parse English even when set to Hindi ('hi-IN')
        recognition.lang = {
            english: "en-IN",
            hindi: "hi-IN"
        };

        recognition.onstart = () => {
            if (permissionTimeout) {
                clearTimeout(permissionTimeout);
                permissionTimeout = null;
            }
            isRequestingPermission = false;
            isListening = true;
            micBtn.classList.add("listening");
            const icon = micBtn.querySelector('i');
            if (icon) {
                icon.classList.remove('fa-microphone');
                icon.classList.add('fa-stop');
            }
            userInput.placeholder = "Listening...";
        };

        recognition.onresult = (event) => {
            let newText = "";

            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    newText += event.results[i][0].transcript;
                }
            }

            const transcript = newText.trim();
            if (transcript) {
                const currentVal = userInput.value;
                if (currentVal && !currentVal.endsWith(' ')) {
                    userInput.value += ' ' + transcript;
                } else {
                    userInput.value += transcript;
                }

                // Auto resize textarea
                userInput.style.height = "auto";
                userInput.style.height = (userInput.scrollHeight) + "px";

                userInput.focus();
            }
        };

        recognition.onerror = (event) => {
            if (permissionTimeout) {
                clearTimeout(permissionTimeout);
                permissionTimeout = null;
            }
            isRequestingPermission = false;
            if (event.error === 'not-allowed') {
                userInput.placeholder = "Microphone access denied.";
                placeholderOverride = true;
            } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
                userInput.placeholder = `Mic error: ${event.error}`;
                placeholderOverride = true;
            }
        };

        recognition.onend = () => {
            stopListening();
            if (!placeholderOverride) {
                userInput.placeholder = "Ask about a faculty member, specialization, or staff role…";
            }
            placeholderOverride = false;
            userInput.focus();
        };

        function stopListening() {
            isListening = false;
            micBtn.classList.remove("listening");
            const icon = micBtn.querySelector('i');
            if (icon) {
                icon.classList.remove('fa-stop');
                icon.classList.add('fa-microphone');
            }
        }

        window.stopDictation = () => {
            if (isListening && recognition) {
                recognition.stop();
            }
        };

        micBtn.addEventListener("click", () => {
            if (isRequestingPermission) return;

            if (isListening) {
                recognition.stop();
                stopListening(); // Eagerly reset UI
            } else {
                isRequestingPermission = true;
                try {
                    recognition.start();
                    permissionTimeout = setTimeout(() => {
                        isRequestingPermission = false;
                        permissionTimeout = null;
                    }, 5000);
                } catch (e) {
                    isRequestingPermission = false;
                    console.error("Speech recognition start error:", e);
                }
            }
        });
    } else if (micBtn) {
        micBtn.style.display = "none";
    }

    // Auto-resize textarea and handle Enter vs Shift+Enter
    userInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault(); // Prevent default newline
            const text = userInput.value;
            if (text.trim()) {
                handleSend(text);
                userInput.style.height = "auto"; // Reset height after send
            }
        }
    });

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = userInput.value;
        if (text.trim()) {
            handleSend(text);
            userInput.style.height = "auto"; // Reset height after send
        }
    });

    // Suggested Actions / Prompts Click Event (Welcome screen prompt cards)
    promptCards.forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.getAttribute("data-prompt");
            handleSend(prompt);
        });
    });

    // New Chat button event
    newChatBtn.addEventListener("click", createNewChat);

    // Clear all history button event
    clearHistoryBtn.addEventListener("click", clearAllHistory);

    // Initialize/Load chat history
    loadChatHistory();

    // Mobile Responsive Sidebar Navigation Toggle
    const sidebarToggle = document.getElementById("mobile-menu-btn");
    const sidebar = document.getElementById("sidebar") || document.querySelector(".sidebar");

    // Inject sidebar overlay dynamically into DOM
    const overlay = document.createElement("div");
    overlay.className = "sidebar-overlay";
    overlay.id = "sidebar-overlay";
    document.body.appendChild(overlay);

    if (sidebarToggle) {
        const handleToggle = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const isMobile = window.innerWidth <= 768;
            if (isMobile) {
                sidebar.classList.toggle("open");
                overlay.classList.toggle("active");
                sidebar.classList.remove("collapsed");
            } else {
                sidebar.classList.toggle("collapsed");
                sidebar.classList.remove("open");
                overlay.classList.remove("active");
            }
        };
        if (sidebarToggle) sidebarToggle.addEventListener("click", handleToggle);
        if (sidebarToggle) sidebarToggle.addEventListener("touchstart", handleToggle, { passive: false });
    }

    if (overlay) {
        overlay.addEventListener("click", closeMobileSidebar);
        overlay.addEventListener("touchstart", (e) => {
            e.preventDefault();
            closeMobileSidebar();
        }, { passive: false });
    }

    function closeMobileSidebar() {
        if (sidebar && sidebar.classList.contains("open")) {
            sidebar.classList.remove("open");
        }
        if (overlay && overlay.classList.contains("active")) {
            overlay.classList.remove("active");
        }
    }
});
