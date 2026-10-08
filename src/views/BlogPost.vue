<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { BlogLoader } from '@/utils/blogLoader'
import { renderMarkdown } from '@/utils/markdownRenderer'

const route = useRoute()

const post = ref(null)
const loading = ref(true)
const error = ref(null)
const activeId = ref('')

// Headings listed in the sidebar contents.
const HEADING_SELECTOR = 'h1, h2, h3, h4'

function slugify(text) {
  const slug = text
    .toLowerCase()
    .trim()
    .replace(/[^\w一-龥\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
  return slug || 'section'
}

// Render once, then walk the headings to assign stable ids and build the TOC.
const rendered = computed(() => {
  if (!post.value?.content) return { html: '', toc: [] }

  const doc = new DOMParser().parseFromString(renderMarkdown(post.value.content), 'text/html')
  const toc = []
  const seen = new Map()

  const normalize = text => text.toLowerCase().replace(/\s+/g, ' ').trim()
  const titleText = normalize(post.value.title || '')

  // Every heading gets an id (in-post anchor links rely on them), but a leading
  // heading that just repeats the post title is left out of the contents.
  doc.body.querySelectorAll(HEADING_SELECTOR).forEach((heading, index) => {
    // KaTeX duplicates its content in MathML, so drop that before reading the text.
    const clone = heading.cloneNode(true)
    clone.querySelectorAll('.katex-mathml').forEach(node => node.remove())
    const text = clone.textContent.replace(/\s+/g, ' ').trim()
    if (!text) return

    let id = heading.id || slugify(text)
    const count = (seen.get(id) || 0) + 1
    seen.set(id, count)
    if (count > 1) id = `${id}-${count}`

    heading.id = id
    if (index === 0 && normalize(text) === titleText) return
    toc.push({ id, text, level: Number(heading.tagName[1]) })
  })

  // Map the heading levels a post actually uses onto consecutive indents, so a
  // post written with h2/h3 nests the same way as one written with h1/h2/h3.
  const levels = [...new Set(toc.map(item => item.level))].sort((a, b) => a - b)
  return {
    html: doc.body.innerHTML,
    toc: toc.map(item => ({ ...item, depth: levels.indexOf(item.level) }))
  }
})

let headingElements = []
let ticking = false

function collectHeadings() {
  headingElements = rendered.value.toc
    .map(item => document.getElementById(item.id))
    .filter(Boolean)
  updateActive()
}

function updateActive() {
  if (!headingElements.length) return
  let current = headingElements[0]
  for (const el of headingElements) {
    if (el.getBoundingClientRect().top <= 120) current = el
    else break
  }
  // Near the bottom of the page the last section may never reach the threshold.
  if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) {
    current = headingElements[headingElements.length - 1]
  }
  activeId.value = current.id
}

function onScroll() {
  if (ticking) return
  ticking = true
  requestAnimationFrame(() => {
    updateActive()
    ticking = false
  })
}

function scrollToHeading(id) {
  const el = document.getElementById(id)
  if (el) el.scrollIntoView({ behavior: 'smooth' })
}

// In-post anchor links (e.g. a hand-written table of contents) would otherwise
// replace the router's hash and drop us off the page.
function onContentClick(event) {
  const link = event.target.closest('a[href^="#"]')
  if (!link) return
  const id = decodeURIComponent(link.getAttribute('href').slice(1))
  if (document.getElementById(id)) {
    event.preventDefault()
    scrollToHeading(id)
  }
}

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

  window.addEventListener('scroll', onScroll, { passive: true })
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', onScroll)
})

// Switching to another post without unmounting the view (e.g. editing the URL).
watch(() => route.params.id, () => {
  post.value = null
  error.value = null
  loading.value = true
  activeId.value = ''
  headingElements = []
  loadPost()
})

watch(post, async () => {
  if (!post.value) return
  await nextTick()
  collectHeadings()

  // 尝试滚动到锚点
  const [, , anchor] = window.location.hash.split('#')
  if (anchor) scrollToHeading(decodeURIComponent(anchor))
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
  <div v-if="loading" class="container">
    <div class="loading-state">
      <div class="loading-spinner"></div>
      <p>Loading blog post...</p>
    </div>
  </div>

  <div v-else-if="error" class="container">
    <div class="error-state">
      <h1>{{ error }}</h1>
      <p>The blog post you're looking for doesn't exist or couldn't be loaded.</p>
      <router-link to="/blog" class="back-link">← Back to Blog</router-link>
    </div>
  </div>

  <div v-else-if="post" class="post-layout">
    <aside class="toc" :class="{ 'is-empty': !rendered.toc.length }">
      <nav class="toc-inner" aria-label="Table of contents">
        <p class="toc-title">Contents</p>
        <ul>
          <li v-for="item in rendered.toc" :key="item.id">
            <a
              :href="`#${item.id}`"
              :class="['toc-link', `depth-${item.depth}`, { active: activeId === item.id }]"
              @click.prevent="scrollToHeading(item.id)"
            >{{ item.text }}</a>
          </li>
        </ul>
      </nav>
    </aside>

    <article class="blog-post">
      <header class="post-header">
        <router-link to="/blog" class="back-link">
          <i class="bi bi-arrow-left"></i> Back to Blog
        </router-link>

        <h1 class="post-title">{{ post.title }}</h1>

        <div class="post-meta">
          <time :datetime="post.date">
            Published on {{ formatDate(post.date) }}
          </time>
          <template v-if="post.tags && post.tags.length">
            <span class="meta-dot">·</span>
            <span v-for="tag in post.tags" :key="tag" class="tag">{{ tag }}</span>
          </template>
        </div>
      </header>

      <div class="post-content" v-html="rendered.html" @click="onContentClick"></div>
    </article>
  </div>
</template>

<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

.post-layout {
  display: grid;
  grid-template-columns: minmax(0, 210px) minmax(0, 720px);
  justify-content: center;
  gap: 3.5rem;
  max-width: 1140px;
  margin: 0 auto;
  padding: 2.5rem 2rem 5rem;
}

/* ---- Table of contents ---- */
.toc {
  position: sticky;
  top: 2rem;
  align-self: start;
  max-height: calc(100vh - 4rem);
  overflow-y: auto;
  padding-top: 3.25rem;
}

.toc.is-empty {
  visibility: hidden;
}

.toc-title {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #9aa4b2;
  margin: 0 0 0.6rem 0.9rem;
}

.toc ul {
  list-style: none;
  margin: 0;
  padding: 0;
}

.toc-link {
  display: block;
  font-size: 0.8125rem;
  line-height: 1.45;
  padding: 0.3rem 0.5rem 0.3rem 0.85rem;
  color: #5a6475;
  text-decoration: none;
  border-left: 2px solid #ebedf0;
  transition: color 0.15s, border-color 0.15s;
}

.toc-link.depth-1 {
  padding-left: 1.65rem;
  font-size: 0.78rem;
  color: #7a8597;
}

.toc-link.depth-2 {
  padding-left: 2.45rem;
  font-size: 0.75rem;
  color: #8d97a6;
}

.toc-link.depth-3 {
  padding-left: 3.25rem;
  font-size: 0.75rem;
  color: #9aa4b2;
}

.toc-link:hover {
  color: #228b22;
}

.toc-link.active {
  color: #228b22;
  font-weight: 600;
  border-left-color: #228b22;
}

/* ---- States ---- */
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

/* ---- Header ---- */
.back-link {
  color: #228b22;
  text-decoration: none;
  font-weight: 500;
  font-size: 0.875rem;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  margin-bottom: 1.25rem;
}

.back-link:hover {
  text-decoration: underline;
}

.post-header {
  margin-bottom: 2.25rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid #ebedf0;
}

.post-title {
  font-size: 2.15rem;
  color: #1c2b3a;
  margin: 0 0 0.6rem;
  line-height: 1.25;
  letter-spacing: -0.01em;
}

.post-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem;
  color: #78838f;
  font-size: 0.8125rem;
}

.meta-dot {
  color: #c4cad1;
}

.tag {
  background: #f1f3f5;
  color: #5a6475;
  padding: 0.1rem 0.55rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 500;
}

/* ---- Content ---- */
.post-content {
  font-size: 1rem;
  line-height: 1.75;
  color: #1f2a37;
}

.post-content :deep(h1),
.post-content :deep(h2),
.post-content :deep(h3),
.post-content :deep(h4),
.post-content :deep(h5),
.post-content :deep(h6) {
  color: #1c2b3a;
  line-height: 1.3;
  scroll-margin-top: 1.5rem;
}

.post-content :deep(h1) {
  font-size: 1.75rem;
  margin: 2.75rem 0 0.9rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid #e9ecef;
}

.post-content :deep(h2) {
  font-size: 1.4rem;
  margin: 2.5rem 0 0.85rem;
}

.post-content :deep(h3) {
  font-size: 1.125rem;
  margin: 1.9rem 0 0.6rem;
}

.post-content :deep(h4) {
  font-size: 1rem;
  margin: 1.5rem 0 0.5rem;
}

.post-content :deep(> *:first-child) {
  margin-top: 0;
}

.post-content :deep(p) {
  margin: 0 0 1.15rem;
}

.post-content :deep(ul),
.post-content :deep(ol) {
  margin: 0 0 1.15rem;
  padding-left: 1.5rem;
}

.post-content :deep(li) {
  margin: 0.3rem 0;
}

.post-content :deep(li > ul),
.post-content :deep(li > ol) {
  margin: 0.3rem 0;
}

.post-content :deep(blockquote) {
  border-left: 3px solid #cfe3cf;
  background: #fafbfa;
  padding: 0.75rem 1rem;
  margin: 1.5rem 0;
  color: #4a5568;
  border-radius: 0 4px 4px 0;
}

.post-content :deep(blockquote > *:last-child) {
  margin-bottom: 0;
}

.post-content :deep(code) {
  background: #f4f6f8;
  padding: 0.15rem 0.35rem;
  border-radius: 3px;
  font-size: 0.875em;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.post-content :deep(pre) {
  background: #f8f9fa;
  padding: 1rem 1.15rem;
  border-radius: 6px;
  overflow-x: auto;
  margin: 1.5rem 0;
  border: 1px solid #e9ecef;
  font-size: 0.875rem;
  line-height: 1.6;
}

.post-content :deep(pre code) {
  background: none;
  padding: 0;
  font-size: inherit;
}

/* Figures: full-width image with an optional centered caption below. */
.post-content :deep(figure) {
  margin: 1.75rem 0;
}

.post-content :deep(figure img) {
  display: block;
  width: 100%;
  height: auto;
  margin: 0 auto;
  border-radius: 4px;
  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14),
              0 3px 1px -2px rgba(0, 0, 0, 0.2),
              0 1px 5px 0 rgba(0, 0, 0, 0.12);
}

.post-content :deep(figcaption) {
  margin-top: 0.7rem;
  font-size: 0.875rem;
  line-height: 1.6;
  color: #78838f;
  text-align: center;
}

.post-content :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}

.post-content :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  font-size: 0.925rem;
}

.post-content :deep(th),
.post-content :deep(td) {
  border: 1px solid #e9ecef;
  padding: 0.5rem 0.7rem;
  text-align: left;
}

.post-content :deep(th) {
  background: #f8f9fa;
  font-weight: 600;
}

.post-content :deep(hr) {
  border: none;
  border-top: 1px solid #e9ecef;
  margin: 2.5rem 0;
}

.post-content :deep(.katex) {
  font-size: 1.05em;
}

.post-content :deep(.katex-display) {
  margin: 1.4rem 0;
  text-align: center;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.2rem 0;
}

.post-content :deep(.math-error) {
  color: #dc3545;
  background: #f8d7da;
  padding: 0.5rem;
  border-radius: 4px;
  border: 1px solid #f5c6cb;
  margin: 1rem 0;
}

@media (max-width: 1080px) {
  .post-layout {
    grid-template-columns: minmax(0, 720px);
    justify-content: center;
    padding: 2rem 1.5rem 4rem;
  }

  .toc {
    display: none;
  }
}

@media (max-width: 768px) {
  .post-title {
    font-size: 1.75rem;
  }

  .post-content :deep(pre) {
    padding: 0.85rem;
    font-size: 0.8rem;
  }
}
</style>
