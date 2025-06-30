// Blog loader utility for reading markdown files from the repository
export class BlogLoader {
  static async getAllPosts() {
    try {
      // In a real implementation, you would have a manifest file or API endpoint
      // that lists all available blog posts. For now, we'll use a static list.
      const postFiles = [
        // 'sample-post-1.md',
        // 'sample-post-2.md',
        'rl-basic.md',
        'Reinforcement_Learning_for_LLMs.md'
        // Add more post filenames here as you create them
      ]

      const posts = await Promise.all(
        postFiles.map(async (filename) => {
          try {
            const response = await fetch(`${import.meta.env.BASE_URL}blog-posts/${filename}`)
            if (!response.ok) return null
            
            const content = await response.text()
            const post = this.parseMarkdownPost(content, filename)
            return post
          } catch (error) {
            console.warn(`Failed to load blog post: ${filename}`, error)
            return null
          }
        })
      )

      return posts
        .filter(post => post !== null)
        .sort((a, b) => new Date(b.date) - new Date(a.date))
    } catch (error) {
      console.error('Failed to load blog posts:', error)
      return []
    }
  }

  static async getPost(id) {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}blog-posts/${id}.md`)
      if (!response.ok) return null
      
      const content = await response.text()
      return this.parseMarkdownPost(content, `${id}.md`)
    } catch (error) {
      console.error(`Failed to load blog post: ${id}`, error)
      return null
    }
  }

  static parseMarkdownPost(content, filename) {
    // Extract frontmatter (YAML between --- delimiters)
    const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/
    const match = content.match(frontmatterRegex)
    
    if (!match) {
      // No frontmatter found, treat entire content as markdown
      return {
        id: filename.replace('.md', ''),
        title: filename.replace('.md', '').replace(/-/g, ' '),
        date: new Date().toISOString(),
        tags: [],
        excerpt: this.generateExcerpt(content),
        content: content
      }
    }

    const [, frontmatter, markdownContent] = match
    const metadata = this.parseFrontmatter(frontmatter)
    
    return {
      id: filename.replace('.md', ''),
      title: metadata.title || filename.replace('.md', '').replace(/-/g, ' '),
      date: metadata.date || new Date().toISOString(),
      tags: metadata.tags || [],
      excerpt: metadata.excerpt || this.generateExcerpt(markdownContent),
      content: markdownContent.trim()
    }
  }

  static parseFrontmatter(frontmatter) {
    const metadata = {}
    const lines = frontmatter.trim().split('\n')
    
    for (const line of lines) {
      const colonIndex = line.indexOf(':')
      if (colonIndex === -1) continue
      
      const key = line.substring(0, colonIndex).trim()
      let value = line.substring(colonIndex + 1).trim()
      
      // Remove quotes if present
      if ((value.startsWith('"') && value.endsWith('"')) || 
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1)
      }
      
      // Parse arrays (tags)
      if (value.startsWith('[') && value.endsWith(']')) {
        value = value.slice(1, -1)
          .split(',')
          .map(item => item.trim().replace(/['"]/g, ''))
          .filter(item => item.length > 0)
      }
      
      metadata[key] = value
    }
    
    return metadata
  }

  static generateExcerpt(content, maxLength = 150) {
    // Remove markdown formatting and get first N characters
    const plainText = content
      .replace(/[#*`_~]/g, '')
      .replace(/\$\$[^$]+\$\$/g, '[Math]')
      .replace(/\$[^$\n]+\$/g, '[Math]')
      .replace(/```[\s\S]*?```/g, '[Code]')
      .trim()
    
    return plainText.length > maxLength 
      ? plainText.substring(0, maxLength) + '...'
      : plainText
  }
}