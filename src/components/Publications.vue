<script setup>
const publications = [
  {
    title: "Machine Learning Applications in Modern Computing",
    authors: "Doe, J., Smith, A., Johnson, B.",
    journal: "Journal of Computer Science",
    year: 2023,
    doi: "10.1000/js.2023.1234",
    preview: "https://picsum.photos/400/500?random=1",
    abstract: "This paper explores the latest applications of machine learning in modern computing systems, focusing on practical implementations and performance optimization."
  },
  {
    title: "Neural Networks: A New Perspective",
    authors: "Doe, J., Anderson, M.",
    journal: "Artificial Intelligence Review",
    year: 2022,
    doi: "10.1000/air.2022.5678",
    preview: "https://picsum.photos/400/500?random=2",
    abstract: "A comprehensive review of neural network architectures and their applications in solving complex computational problems."
  }
]
</script>

<template>
  <section id="publications" class="section">
    <h2>Publications</h2>
    <div class="publications-list">
      <article v-for="pub in publications" :key="pub.doi" class="publication">
        <div class="publication-content">
          <div class="publication-text">
            <h3>{{ pub.title }}</h3>
            <p class="authors">{{ pub.authors }}</p>
            <p class="journal">{{ pub.journal }} ({{ pub.year }})</p>
            <p class="abstract">{{ pub.abstract }}</p>
            <p class="doi">DOI: {{ pub.doi }}</p>
          </div>
          <div class="publication-preview">
            <img 
              :src="pub.preview" 
              :alt="'Preview of ' + pub.title"
              @click="() => window.open(pub.preview, '_blank')"
            >
          </div>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.section {
  padding: 3rem 0;
  border-bottom: 1px solid #eee;
}

.publication {
  margin-bottom: 2rem;
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
}

.publication-text {
  flex: 1;
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
  flex: 0 0 200px;
  display: flex;
  align-items: flex-start;
}

.publication-preview img {
  width: 100%;
  height: 250px;
  object-fit: cover;
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