import { marked } from 'marked'
import katex from 'katex'
import hljs from 'highlight.js'

// Configure marked with custom renderer
const renderer = new marked.Renderer()

// Custom renderer for math expressions
renderer.code = function(code, language) {
  if (language === 'math') {
    try {
      return katex.renderToString(code, { displayMode: true })
    } catch (e) {
      return `<div class="math-error">Math Error: ${e.message}</div>`
    }
  }
  
  if (language && hljs.getLanguage(language)) {
    try {
      return `<pre><code class="hljs ${language}">${hljs.highlight(code, { language }).value}</code></pre>`
    } catch (e) {
      // Fall back to plain text
    }
  }
  
  return `<pre><code class="hljs">${hljs.highlightAuto(code).value}</code></pre>`
}

// Configure marked options
marked.setOptions({
  renderer: renderer,
  gfm: true,
  breaks: true,
  sanitize: false
})

export function renderMarkdown(markdown) {
  // Handle inline math expressions
  let processed = markdown.replace(/\$\$([^$]+)\$\$/g, (match, math) => {
    try {
      return katex.renderToString(math, { displayMode: true })
    } catch (e) {
      return `<span class="math-error">Math Error: ${e.message}</span>`
    }
  })

  // Handle inline math expressions
  processed = processed.replace(/\$([^$\n]+)\$/g, (match, math) => {
    try {
      return katex.renderToString(math, { displayMode: false })
    } catch (e) {
      return `<span class="math-error">Math Error: ${e.message}</span>`
    }
  })

  return marked(processed)
}