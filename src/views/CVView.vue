<script setup>
import { ref } from 'vue'

// 定义 CV 的 URL
const cvUrl = `${import.meta.env.BASE_URL}assets/CV-JixuanHe.pdf`
</script>


<template>
  <div class="container">
    <header class="cv-header">
      <h1>Curriculum Vitáe</h1>
      <router-link to="/" class="back-link">← Back to Home</router-link>
    </header>
    
    <p>You may preview my CV online or download it <a :href="cvUrl" download>here</a>.</p>
    <iframe :src="cvUrl" class="cv-preview" frameborder="0"></iframe>
  </div>
</template>

<style scoped>
a {
  color: #42b883;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
}

.cv-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.back-link {
  color: #42b883;
  text-decoration: none;
  font-weight: 500;
}

.back-link:hover {
  text-decoration: underline;
}

.cv-preview {
  width: 100%;
  height: 80vh;
  border: none;
}
</style>