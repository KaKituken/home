<script setup>
import { ref, onMounted } from 'vue'
import { BlogLoader } from '@/utils/blogLoader'

const posts = ref([])
const loading = ref(true)
const error = ref(null)

onMounted(async () => {
  try {
    posts.value = await BlogLoader.getAllPosts()
  } catch (err) {
    error.value = 'Failed to load blog posts'
    console.error('Error loading blog posts:', err)
  } finally {
    loading.value = false
  }
})

function formatDate(dateString) {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}
</script>

<template>
  <div class="container">
    <header class="blog-header">
      <h1>Blog Posts</h1>
      <router-link to="/" class="back-link">← Back to Home</router-link>
    </header>

    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <p>Loading blog posts...</p>
    </div>

    <div v-else-if="error" class="error-state">
      <h2>{{ error }}</h2>
      <p>Please try again later.</p>
    </div>

    <div v-else-if="posts.length === 0" class="empty-state">
      <h2>No blog posts yet</h2>
      <p>Check back later for new content!</p>
    </div>

    <div v-else class="posts-grid">
      <article v-for="post in posts" :key="post.id" class="post-card">
        <div class="post-content">
          <h2>
            <router-link :to="`/blog/${post.id}`">{{ post.title }}</router-link>
          </h2>
          <p class="post-meta">
            Published on {{ formatDate(post.date) }}
          </p>
          <p class="post-excerpt">{{ post.excerpt }}</p>
          <div class="post-tags" v-if="post.tags && post.tags.length">
            <span v-for="tag in post.tags" :key="tag" class="tag">{{ tag }}</span>
          </div>
        </div>
      </article>
    </div>
  </div>
</template>

<style scoped>
.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
}

.blog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  flex-wrap: wrap;
  gap: 1rem;
}

.back-link {
  color: #228b22;
  text-decoration: none;
  font-weight: 500;
}

.back-link:hover {
  text-decoration: underline;
}

.loading-state, .error-state, .empty-state {
  text-align: center;
  padding: 4rem 2rem;
  color: #666;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #228b22;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-state h2, .empty-state h2 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.posts-grid {
  display: grid;
  gap: 2rem;
}

.post-card {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 2rem;
  transition: transform 0.2s, box-shadow 0.2s;
  border: 1px solid #e9ecef;
}

.post-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.post-content h2 {
  margin: 0 0 1rem;
}

.post-content h2 a {
  color: #2c3e50;
  text-decoration: none;
}

.post-content h2 a:hover {
  color: #228b22;
}

.post-meta {
  color: #666;
  font-size: 0.9rem;
  margin: 0 0 1rem;
}

.post-excerpt {
  color: #555;
  line-height: 1.6;
  margin: 0 0 1rem;
}

.post-tags {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.tag {
  background: #e9ecef;
  color: #495057;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 500;
}

@media (max-width: 768px) {
  .blog-header {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>