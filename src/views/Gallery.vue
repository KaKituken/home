<script setup>
const images = [
  {
    id: 1,
    title: 'Research Lab',
    description: 'Our state-of-the-art research facility',
    url: 'https://picsum.photos/800/600?random=1'
  },
  {
    id: 2,
    title: 'Conference Presentation',
    description: 'Presenting at International AI Conference 2023',
    url: 'https://picsum.photos/800/600?random=2'
  },
  {
    id: 3,
    title: 'Team Meeting',
    description: 'Weekly research group discussion',
    url: 'https://picsum.photos/800/600?random=3'
  },
  {
    id: 4,
    title: 'Workshop',
    description: 'Machine Learning Workshop with Graduate Students',
    url: 'https://picsum.photos/800/600?random=4'
  }
]
</script>

<template>
  <div class="container">
    <header class="gallery-header">
      <h1>Gallery</h1>
      <router-link to="/" class="back-link">← Back to Home</router-link>
    </header>
    
    <div class="gallery-grid">
      <div v-for="image in images" :key="image.id" class="gallery-item">
        <img :src="image.url" :alt="image.title">
        <div class="image-info">
          <h3>{{ image.title }}</h3>
          <p>{{ image.description }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
}

.gallery-header {
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

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
}

.gallery-item {
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
}

.gallery-item:hover {
  transform: translateY(-5px);
}

.gallery-item img {
  width: 100%;
  height: 200px;
  object-fit: cover;
}

.image-info {
  padding: 1rem;
}

.image-info h3 {
  margin: 0 0 0.5rem;
  color: #2c3e50;
}

.image-info p {
  margin: 0;
  color: #666;
  font-size: 0.9rem;
}

@media (max-width: 768px) {
  .gallery-grid {
    grid-template-columns: 1fr;
  }
}
</style>