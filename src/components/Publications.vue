<script setup>
const publications = [
  {
    title: "Machine Learning Applications in Modern Computing",
    authors: "Doe, J., Smith, A., Johnson, B.",
    journal: "Journal of Computer Science",
    year: 2023,
    doi: "10.1000/js.2023.1234"
  },
  {
    title: "Neural Networks: A New Perspective",
    authors: "Doe, J., Anderson, M.",
    journal: "Artificial Intelligence Review",
    year: 2022,
    doi: "10.1000/air.2022.5678"
  }
]
</script>

<template>
  <section id="publications" class="section">
    <h2>Publications</h2>
    <div class="publications-list">
      <article v-for="pub in publications" :key="pub.doi" class="publication">
        <h3>{{ pub.title }}</h3>
        <p class="authors">{{ pub.authors }}</p>
        <p class="journal">{{ pub.journal }} ({{ pub.year }})</p>
        <p class="doi">DOI: {{ pub.doi }}</p>
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
}

.publication h3 {
  margin: 0 0 0.5rem;
  color: #2c3e50;
}

.authors {
  color: #34495e;
  margin: 0.5rem 0;
}

.journal {
  color: #7f8c8d;
  font-style: italic;
  margin: 0.5rem 0;
}

.doi {
  color: #95a5a6;
  font-size: 0.9rem;
  margin: 0.5rem 0 0;
}
</style>