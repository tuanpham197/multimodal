import { defineStore } from 'pinia'
import axios from 'axios'

const API_URL = 'http://localhost:8000'
const REQUEST_TIMEOUT = 60000

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [],
    isTyping: false,
    error: null,
    abortController: null
  }),

  actions: {
    async sendMessage(text) {
      if (!text.trim()) return

      this.messages.push({
        id: Date.now(),
        type: 'user',
        content: text,
        timestamp: new Date()
      })

      this.isTyping = true
      this.error = null
      this.abortController = new AbortController()

      try {
        const response = await axios.post(`${API_URL}/chat`, {
          message: text
        }, {
          timeout: REQUEST_TIMEOUT,
          signal: this.abortController.signal
        })

        const contextImages = response.data.context_images || []
        const images = contextImages.map(img => img.data_url || img)
        
        this.messages.push({
          id: Date.now() + 1,
          type: 'assistant',
          content: response.data.response,
          contexts: response.data.context_texts || [],
          images: images,
          timestamp: new Date()
        })
      } catch (err) {
        if (axios.isCancel(err) || err.name === 'CanceledError') {
          this.error = 'Request cancelled'
        } else if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
          this.error = 'Request timeout - server took too long'
        } else if (err.code === 'ERR_NETWORK') {
          this.error = 'Network error - check if server is running'
        } else {
          this.error = err.response?.data?.detail || 'Failed to get response'
        }
        this.messages.push({
          id: Date.now() + 1,
          type: 'error',
          content: this.error,
          timestamp: new Date()
        })
      } finally {
        this.isTyping = false
        this.abortController = null
      }
    },

    cancelRequest() {
      if (this.abortController) {
        this.abortController.abort()
        this.isTyping = false
      }
    },

    clearMessages() {
      this.messages = []
      this.error = null
    }
  }
})

