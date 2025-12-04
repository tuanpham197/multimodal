<script setup>
import ChatBox from './components/ChatBox.vue'
</script>

<template>
  <div class="app">
    <ChatBox />
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --bg-primary: #0a0a0f;
  --bg-secondary: #12121a;
  --bg-tertiary: #1a1a25;
  --accent: #6366f1;
  --accent-glow: rgba(99, 102, 241, 0.3);
  --text-primary: #e4e4e7;
  --text-secondary: #a1a1aa;
  --text-muted: #71717a;
  --border: #27272a;
  --user-bg: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  --assistant-bg: #1e1e2a;
  --error-bg: #2d1b1b;
  --error-text: #f87171;
}

body {
  font-family: 'Outfit', sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
}

.app {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: 
    radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
    radial-gradient(ellipse at bottom right, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
    var(--bg-primary);
}
</style>

