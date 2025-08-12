<script setup>
import { computed } from 'vue'

function openImage(url) {
  if (url) {
    window.open(url, '_blank');
  }
}

const publications = [
  {
    title: "Restage4D: Reanimating Deformable 3D Reconstruction from a Single Video",
    authors: "<strong>Jixuan He</strong>, Chieh Hubert Lin, Lu Qi, Ming-Hsuan Yang",
    journal: "Under Review",
    year: null,
    preview: `${import.meta.env.BASE_URL}assets/he2025restage.png`,
    arxiv: "https://arxiv.org/abs/2508.06715",
    code: null,
    project: null,
    demo: null,
    poster: null
  },
  {
    title: "Affordance-Aware Object Insertion via Mask-Aware Dual Diffusion",
    authors: "<strong>Jixuan He</strong>, Wanhua Li*, Ye Liu, Junsik Kim, Donglai Wei, Hanspeter Pfister",
    journal: "Under Review",
    year: null,
    preview: `${import.meta.env.BASE_URL}assets/he2024affordance.png`,
    arxiv: "https://arxiv.org/abs/2412.14462",
    code: "https://github.com/KaKituken/affordance-aware-any",
    project: "https://kakituken.github.io/affordance-any.github.io/",
    demo: null,
    poster: null
  },
  {
    title: "R2-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding",
    authors: "Ye Liu, <strong>Jixuan He</strong>, Wanhua Li*, Junsik Kim, Donglai Wei, Hanspeter Pfister, Chang Wen Chen*",
    journal: "The European Conference on Computer Vision (ECCV)",
    year: 2024,
    preview: `${import.meta.env.BASE_URL}assets/liu2024tuning.jpg`,
    arxiv: "https://arxiv.org/abs/2404.00801",
    code: "https://github.com/yeliudev/R2-Tuning?tab=readme-ov-file",
    project: null,
    demo: "https://huggingface.co/spaces/yeliudev/R2-Tuning",
    poster: "https://yeliu.dev/lib/files/r2tuning_poster.pdf"
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
            <div class="links">
              <a class="links-item arxiv" v-if="pub.arxiv" :href="pub.arxiv">
                arxiv
              </a>
              <a class="links-item code" v-if="pub.code" :href="pub.code">
                code
              </a>
              <a class="links-item project" v-if="pub.project" :href="pub.project">
                project
              </a>
              <a class="links-item poster" v-if="pub.poster" :href="pub.poster">
                poster
              </a>
              <a class="links-item demo" v-if="pub.demo" :href="pub.demo">
                demo
              </a>
            </div>
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
  height: 12rem;
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
  position: relative;
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

.publication-content .links {
  display: flex;
  position: absolute;
  width: 100%;
  height: 2rem;
  /* background-color: white; */
  bottom: 0;
  gap: 0.7rem;
  line-height: inherit;
}

.links .links-item {
  border-width: 1px;
  border-color: #b9b7bd;
  border-style: solid;
  line-height: inherit;
  padding-left: 5px;
  padding-right: 5px;
  font-style: italic;
  border-radius: 5%;
  font-weight: 200;
  text-decoration: none;
  transition: transform 0.2s, background-color 0.2s;
}

.links .links-item:hover {
  transform: scale(1.05);
  font-weight: 400;
  color: #f2f1e8;
}

.links .arxiv {
  color: #db1f48;
  border-color: #db1f48;
}

.links .arxiv:hover {
  background-color: #db1f48;
}

.links .code {
  color: #FFA500;
  border-color: #FFA500;
}

.links .code:hover {
  background-color: #FFA500;
}

.links .project {
  color: #01949a;
  border-color: #01949a;
}

.links .project:hover{
  background-color: #01949a;
}

.links .poster {
  color: #104210;
  border-color: #104210;
}

.links .poster:hover{
  background-color: #104210;
}

.links .demo {
  color: #004369;
  border-color: #004369;
}

.links .demo:hover{
  background-color: #004369;
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