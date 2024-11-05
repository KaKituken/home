<script setup>
import { computed } from 'vue'

function openImage(url) {
  if (url) {
    window.open(url, '_blank');
  }
}

const publications = [
  {
    title: "Putting Any Object into Any Scene: Affordance-Aware Object Insertion via Mask-Aware Dual Diffusion",
    authors: "Jixuan He, Wanhua Li*, Ye Liu, Junsik Kim, Donglai Wei, Hanspeter Pfister",
    journal: "Under Review",
    year: null,
    preview: "/assets/he2024affordance.jpg",
    abstract: null,
    arxiv: null,
    code: null,
    poster: null,
    demo: null
  },
  {
    title: "R2-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding",
    authors: "Ye Liu, Jixuan He, Wanhua Li*, Junsik Kim, Donglai Wei, Hanspeter Pfister, Chang Wen Chen*",
    journal: "The European Conference on Computer Vision (ECCV)",
    year: 2024,
    preview: "/assets/liu2024tuning.jpg",
    abstract: null,
    arxiv: null,
    code: null,
    poster: null,
    demo: null
  }
]

const highlightedPublications = computed(() => {
  return publications.map(pub => ({
    ...pub,
    highlightedAuthors: pub.authors.replace(/\bJixuan He\b/g, '<span class="highlighted-author">Jixuan He</span>')
  }))
})
</script>

<template>
  <section id="publications" class="section">
    <h2>Publications</h2>
    <div class="publications-list">
      <article v-for="pub in highlightedPublications" :key="pub.doi" class="publication">
        <div class="publication-content">
          <div class="publication-text">
            <h3>{{ pub.title }}</h3>
            <p class="authors" v-html="pub.highlightedAuthors"></p>
            <p class="journal">{{ pub.journal }} {{ pub.year }}</p>
            <p class="abstract">{{ pub.abstract }}</p>
          </div>
          <div class="publication-preview">
            <img 
              :src="pub.preview" 
              :alt="'Preview of ' + pub.title"
              @click="openImage(pub.preview)"
            >
          </div>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.section {
  padding: 2rem 0 0 0;
  border-bottom: 1px solid #eee;
}

.section h2{
    margin-bottom: 2rem;
}

.publication {
  margin-bottom: 2rem;
  height: 10rem;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 8px;
  transition: transform 0.2s;
}

.publication:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.publication-content {
  display: flex;
  gap: 2rem;
  height: 100%;
}

.publication-text {
  flex: 6;
}

.publication h3 {
  margin: 0 0 0.5rem;
  color: #2c3e50;
}

.authors {
  color: #34495e;
  margin: 0.5rem 0;
  font-weight: 500;
}

.authors span {
  font-weight: bold;
  font-size: x-large;
  color: #e74c3c; /* Choose a color that stands out */
}

.journal {
  color: #7f8c8d;
  font-style: italic;
  margin: 0.5rem 0;
}

.abstract {
  color: #2c3e50;
  margin: 1rem 0;
  line-height: 1.6;
}

.doi {
  color: #95a5a6;
  font-size: 0.9rem;
  margin: 0.5rem 0 0;
}

.publication-preview {
  flex: 1 0 200px;
  height: 100%;
  display: flex;
  align-items: flex-start;
  background-color: white;
}

.publication-preview img {
  width: 100%;
  height: 100%;
  object-fit:contain;
  border-radius: 4px;
  cursor: pointer;
  transition: transform 0.2s;
}

.publication-preview img:hover {
  transform: scale(1.05);
}

@media (max-width: 768px) {
  .publication-content {
    flex-direction: column;
  }

  .publication-preview {
    flex: 0 0 auto;
    width: 100%;
    margin-top: 1rem;
  }

  .publication-preview img {
    height: 200px;
  }
}
</style>