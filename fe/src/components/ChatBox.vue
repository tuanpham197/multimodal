<script setup>
import { ref, nextTick, watch } from 'vue'
import { useChatStore } from '../stores/chat'

const chatStore = useChatStore()
const inputMessage = ref('')
const messagesContainer = ref(null)
const inputRef = ref(null)

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

watch(() => chatStore.messages.length, scrollToBottom)
watch(() => chatStore.isTyping, scrollToBottom)

const handleSend = async () => {
  if (!inputMessage.value.trim() || chatStore.isTyping) return
  const message = inputMessage.value
  inputMessage.value = ''
  await chatStore.sendMessage(message)
  nextTick(() => {
    if (inputRef.value) {
      inputRef.value.focus()
    }
  })
}

const handleKeydown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

const formatTime = (date) => {
  return new Date(date).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

const selectedImage = ref(null)

const openImage = (img) => {
  selectedImage.value = img
}

const closeModal = () => {
  selectedImage.value = null
}
</script>

<template>
  <div class="chat-container">
    <header class="chat-header">
      <div class="header-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <div class="header-info">
        <h1>Multimodal RAG</h1>
        <span class="status">
          <span class="status-dot"></span>
          Online
        </span>
      </div>
      <button class="clear-btn" @click="chatStore.clearMessages" title="Clear chat">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
        </svg>
      </button>
    </header>

    <div class="messages" ref="messagesContainer">
      <div v-if="chatStore.messages.length === 0" class="empty-state">
        <div class="empty-icon">💬</div>
        <p>Start a conversation</p>
        <span>Ask anything about your documents</span>
      </div>

      <div
        v-for="msg in chatStore.messages"
        :key="msg.id"
        :class="['message', msg.type]"
      >
        <div class="message-content">
          <p>{{ msg.content }}</p>
          
          <div v-if="msg.images && msg.images.length > 0" class="message-images">
            <div class="images-label">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
              </svg>
              Context Images ({{ msg.images.length }})
            </div>
            <div class="images-grid">
              <div 
                v-for="(img, idx) in msg.images" 
                :key="idx" 
                class="image-wrapper"
                @click="openImage(img)"
              >
                <img :src="img" :alt="'Context image ' + (idx + 1)" />
                <div class="image-overlay">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="m21 21-4.35-4.35"/>
                    <path d="M11 8v6M8 11h6"/>
                  </svg>
                </div>
              </div>
            </div>
          </div>
          
          <span class="timestamp">{{ formatTime(msg.timestamp) }}</span>
        </div>
      </div>

      <div v-if="chatStore.isTyping" class="message assistant typing">
        <div class="message-content">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    </div>

    <div v-if="selectedImage" class="image-modal" @click="closeModal">
      <div class="modal-content" @click.stop>
        <button class="modal-close" @click="closeModal">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6 6 18M6 6l12 12"/>
          </svg>
        </button>
        <img :src="selectedImage" alt="Full size image" />
      </div>
    </div>

    <footer class="chat-input">
      <div class="input-wrapper">
        <textarea
          ref="inputRef"
          v-model="inputMessage"
          @keydown="handleKeydown"
          placeholder="Type your message..."
          rows="1"
        ></textarea>
        <button
          v-if="chatStore.isTyping"
          class="cancel-btn"
          @click="chatStore.cancelRequest"
          title="Cancel request"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6 6 18M6 6l12 12"/>
          </svg>
        </button>
        <button
          v-else
          class="send-btn"
          @click="handleSend"
          :disabled="!inputMessage.trim()"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
          </svg>
        </button>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.chat-container {
  width: 100%;
  max-width: 720px;
  height: 85vh;
  max-height: 800px;
  background: var(--bg-secondary);
  border-radius: 24px;
  border: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 
    0 0 0 1px rgba(255, 255, 255, 0.03),
    0 25px 50px -12px rgba(0, 0, 0, 0.5),
    0 0 100px var(--accent-glow);
}

.chat-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 20px 24px;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border);
}

.header-icon {
  width: 44px;
  height: 44px;
  background: var(--user-bg);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.header-info {
  flex: 1;
}

.header-info h1 {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.status {
  font-size: 0.8rem;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  width: 8px;
  height: 8px;
  background: #22c55e;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.clear-btn {
  padding: 10px;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 10px;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;
}

.clear-btn:hover {
  background: var(--bg-primary);
  color: var(--text-primary);
  border-color: var(--text-muted);
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.messages::-webkit-scrollbar {
  width: 6px;
}

.messages::-webkit-scrollbar-track {
  background: transparent;
}

.messages::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 3px;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: var(--text-muted);
}

.empty-icon {
  font-size: 3rem;
  margin-bottom: 16px;
  opacity: 0.5;
}

.empty-state p {
  font-size: 1.1rem;
  color: var(--text-secondary);
  margin-bottom: 4px;
}

.empty-state span {
  font-size: 0.9rem;
}

.message {
  display: flex;
  max-width: 85%;
}

.message.user {
  align-self: flex-end;
}

.message.assistant,
.message.error {
  align-self: flex-start;
}

.message-content {
  padding: 14px 18px;
  border-radius: 18px;
  position: relative;
}

.message.user .message-content {
  background: var(--user-bg);
  color: white;
  border-bottom-right-radius: 6px;
}

.message.assistant .message-content {
  background: var(--assistant-bg);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-bottom-left-radius: 6px;
}

.message.error .message-content {
  background: var(--error-bg);
  color: var(--error-text);
  border: 1px solid rgba(248, 113, 113, 0.2);
  border-bottom-left-radius: 6px;
}

.message-content p {
  font-size: 0.95rem;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.timestamp {
  display: block;
  font-size: 0.7rem;
  margin-top: 8px;
  opacity: 0.6;
}

.typing-indicator {
  display: flex;
  gap: 5px;
  padding: 4px 0;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  background: var(--accent);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) { animation-delay: 0s; }
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 80%, 100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.chat-input {
  padding: 20px 24px;
  background: var(--bg-tertiary);
  border-top: 1px solid var(--border);
}

.input-wrapper {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px 16px;
  transition: border-color 0.2s;
}

.input-wrapper:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}

.input-wrapper textarea {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: var(--text-primary);
  font-family: inherit;
  font-size: 0.95rem;
  resize: none;
  max-height: 120px;
  line-height: 1.5;
}

.input-wrapper textarea::placeholder {
  color: var(--text-muted);
}

.send-btn {
  width: 40px;
  height: 40px;
  background: var(--user-bg);
  border: none;
  border-radius: 12px;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.send-btn:hover:not(:disabled) {
  transform: scale(1.05);
  box-shadow: 0 4px 15px var(--accent-glow);
}

.send-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.cancel-btn {
  width: 40px;
  height: 40px;
  background: #ef4444;
  border: none;
  border-radius: 12px;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.cancel-btn:hover {
  background: #dc2626;
  transform: scale(1.05);
}

.message-images {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}

.images-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-bottom: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 8px;
}

.image-wrapper {
  position: relative;
  aspect-ratio: 1;
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
  border: 1px solid var(--border);
  transition: all 0.2s;
}

.image-wrapper:hover {
  border-color: var(--accent);
  transform: scale(1.02);
}

.image-wrapper img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.2s;
  color: white;
}

.image-wrapper:hover .image-overlay {
  opacity: 1;
}

.image-modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 40px;
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.modal-content {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
}

.modal-content img {
  max-width: 100%;
  max-height: 85vh;
  border-radius: 12px;
  box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
}

.modal-close {
  position: absolute;
  top: -50px;
  right: 0;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.2s;
}

.modal-close:hover {
  background: var(--accent);
  border-color: var(--accent);
}
</style>

