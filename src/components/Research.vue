<script setup>
const interests = [
  {
    title: "Machine Learning",
    description: "Deep learning, reinforcement learning, and neural networks"
  },
  {
    title: "Artificial Intelligence",
    description: "Natural language processing and computer vision"
  },
  {
    title: "Data Science",
    description: "Big data analytics and statistical modeling"
  }
]
</script>

<template>
  <section id="research" class="section">
    <h2>Research Interests</h2>
    <div class="interests-grid">
      <div v-for="interest in interests" :key="interest.title" class="interest-card">
        <h3>{{ interest.title }}</h3>
        <p>{{ interest.description }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.section {
  padding: 3rem 0;
  border-bottom: 1px solid #eee;
}

.interests-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
}

.interest-card {
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 8px;
  transition: transform 0.2s;
}

.interest-card:hover {
  transform: translateY(-5px);
}

.interest-card h3 {
  color: #2c3e50;
  margin: 0 0 1rem;
}

.interest-card p {
  color: #34495e;
  margin: 0;
  line-height: 1.6;
}
</style>