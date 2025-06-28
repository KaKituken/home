<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { BlogLoader } from '@/utils/blogLoader'
import { renderMarkdown } from '@/utils/markdownRenderer'

const route = useRoute()
const router = useRouter()

const post = ref(null)
const loading = ref(true)
const error = ref(null)

const renderedContent = computed(() => {
  if (!post.value?.content) return ''
  return renderMarkdown(post.value.content)
})

onMounted(() => {
  // Load KaTeX CSS
  const katexCSS = document.createElement('link')
  katexCSS.rel = 'stylesheet'
  katexCSS.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'
  document.head.appendChild(katexCSS)

  // Load Highlight.js CSS
  const hlCSS = document.createElement('link')
  hlCSS.rel = 'stylesheet'
  hlCSS.href = 'https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github.min.css'
  document.head.appendChild(hlCSS)

  loadPost()
})

async function loadPost() {
  try {
    const postId = route.params.id
    const foundPost = await BlogLoader.getPost(postId)
    
    if (foundPost) {
      post.value = foundPost
    } else {
      error.value = 'Blog post not found'
    }
  } catch (err) {
    error.value = 'Failed to load blog post'
    console.error('Error loading blog post:', err)
  } finally {
    loading.value = false
  }
}

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
    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <p>Loading blog post...</p>
    </div>

    <div v-else-if="error" class="error-state">
      <h1>{{ error }}</h1>
      <p>The blog post you're looking for doesn't exist or couldn't be loaded.</p>
      <router-link to="/blog" class="back-link">← Back to Blog</router-link>
    </div>

    <article v-else-if="post" class="blog-post">
      <header class="post-header">
        <router-link to="/blog" class="back-link">
          <i class="bi bi-arrow-left"></i> Back to Blog
        </router-link>

        <h1 class="post-title">{{ post.title }}</h1>
        
        <div class="post-meta">
          <time :datetime="post.date">
            Published on {{ formatDate(post.date) }}
          </time>
        </div>

        <div v-if="post.tags && post.tags.length" class="post-tags">
          <span v-for="tag in post.tags" :key="tag" class="tag">{{ tag }}</span>
        </div>
      </header>

      <div class="post-content" v-html="renderedContent"></div>
    </article>
  </div>
</template>

<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

.loading-state, .error-state {
  text-align: center;
  padding: 4rem 2rem;
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

.error-state h1 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.error-state p {
  color: #666;
  margin-bottom: 2rem;
}

.back-link {
  color: #228b22;
  text-decoration: none;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 2rem;
}

.back-link:hover {
  text-decoration: underline;
}

.post-header {
  margin-bottom: 3rem;
}

.post-title {
  font-size: 2.5rem;
  color: #2c3e50;
  margin: 1rem 0;
  line-height: 1.2;
}

.post-meta {
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 1rem;
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

.post-content {
  line-height: 1.8;
  color: #2c3e50;
}

/* Content styling */
.post-content :deep(h1),
.post-content :deep(h2),
.post-content :deep(h3),
.post-content :deep(h4),
.post-content :deep(h5),
.post-content :deep(h6) {
  color: #2c3e50;
  margin: 2rem 0 1rem;
  line-height: 1.3;
}

.post-content :deep(h1) {
  font-size: 2rem;
  border-bottom: 2px solid #e9ecef;
  padding-bottom: 0.5rem;
}

.post-content :deep(h2) {
  font-size: 1.6rem;
}

.post-content :deep(h3) {
  font-size: 1.3rem;
}

.post-content :deep(p) {
  margin: 1.5rem 0;
}

.post-content :deep(ul),
.post-content :deep(ol) {
  margin: 1.5rem 0;
  padding-left: 2rem;
}

.post-content :deep(li) {
  margin: 0.5rem 0;
}

.post-content :deep(blockquote) {
  border-left: 4px solid #228b22;
  padding-left: 1rem;
  margin: 1.5rem 0;
  color: #666;
  font-style: italic;
}

.post-content :deep(code) {
  background: #f8f9fa;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-size: 0.9rem;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.post-content :deep(pre) {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  overflow-x: auto;
  margin: 2rem 0;
  border: 1px solid #e9ecef;
}

.post-content :deep(pre code) {
  background: none;
  padding: 0;
}

.post-content :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  margin: 2rem 0;
}

.post-content :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 2rem 0;
}

.post-content :deep(th),
.post-content :deep(td) {
  border: 1px solid #e9ecef;
  padding: 0.75rem;
  text-align: left;
}

.post-content :deep(th) {
  background: #f8f9fa;
  font-weight: 600;
}

.post-content :deep(.katex) {
  font-size: 1.1em;
}

.post-content :deep(.katex-display) {
  margin: 2rem 0;
  text-align: center;
}

.post-content :deep(.math-error) {
  color: #dc3545;
  background: #f8d7da;
  padding: 0.5rem;
  border-radius: 4px;
  border: 1px solid #f5c6cb;
  margin: 1rem 0;
}

@media (max-width: 768px) {
  .post-title {
    font-size: 2rem;
  }

  .post-content :deep(pre) {
    padding: 1rem;
    font-size: 0.8rem;
  }
}
</style>