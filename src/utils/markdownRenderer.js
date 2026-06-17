import { marked } from 'marked'
import katex from 'katex'
import hljs from 'highlight.js'

// Configure marked with custom renderer
const renderer = new marked.Renderer()

// Custom renderer for fenced code blocks (also supports ```math for display math)
renderer.code = function(code, language) {
  if (language === 'math') {
    try {
      return katex.renderToString(code, { displayMode: true, throwOnError: false })
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

function renderKatex(math, displayMode) {
  try {
    return katex.renderToString(math, { displayMode, throwOnError: false })
  } catch (e) {
    const tag = displayMode ? 'div' : 'span'
    return `<${tag} class="math-error">Math Error: ${e.message}</${tag}>`
  }
}

export function renderMarkdown(markdown) {
  // Extract math expressions BEFORE running marked, replacing them with inert
  // placeholders. Otherwise marked re-parses KaTeX's generated HTML as Markdown
  // and mangles subscripts/superscripts (e.g. \hat{\mathbf{x}}_{t-1} breaks).
  const mathBlocks = []
  const placeholder = (i) => `@@KATEXMATH${i}@@`

  // Protect fenced/inline code so a `$` inside code isn't treated as math.
  const codeBlocks = []
  let processed = markdown.replace(/```[\s\S]*?```|`[^`\n]+`/g, (match) => {
    codeBlocks.push(match)
    return `@@KATEXCODE${codeBlocks.length - 1}@@`
  })

  // Display math: $$ ... $$ (may span multiple lines)
  processed = processed.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => {
    mathBlocks.push({ math: math.trim(), displayMode: true })
    return placeholder(mathBlocks.length - 1)
  })

  // Inline math: $ ... $ (single line)
  processed = processed.replace(/\$([^$\n]+?)\$/g, (match, math) => {
    mathBlocks.push({ math: math.trim(), displayMode: false })
    return placeholder(mathBlocks.length - 1)
  })

  // Restore code placeholders before markdown so marked renders them normally.
  processed = processed.replace(/@@KATEXCODE(\d+)@@/g, (m, i) => codeBlocks[Number(i)])

  let html = marked(processed)

  // Render math into the placeholders. Strip a wrapping <p> around display math
  // so we don't nest a block-level KaTeX element inside a paragraph.
  html = html.replace(/<p>\s*@@KATEXMATH(\d+)@@\s*<\/p>/g, (m, i) => {
    const { math, displayMode } = mathBlocks[Number(i)]
    return renderKatex(math, displayMode)
  })
  html = html.replace(/@@KATEXMATH(\d+)@@/g, (m, i) => {
    const { math, displayMode } = mathBlocks[Number(i)]
    return renderKatex(math, displayMode)
  })

  return html
}
