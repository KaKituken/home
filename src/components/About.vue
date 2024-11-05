<script setup>
const bio = `I am a researcher in the field of Computer Science, specializing in artificial intelligence and machine learning. My work focuses on developing novel algorithms for real-world applications.`
</script>

<template>
  <section id="about" class="section">
    <h2>About Me</h2>
    <p>{{ bio }}</p>
    <div class="education">
      <h3>Education</h3>
      <ul>
        <li>Ph.D. in Computer Science, Stanford University</li>
        <li>M.S. in Computer Science, MIT</li>
        <li>B.S. in Computer Science, Berkeley</li>
      </ul>
    </div>
  </section>
</template>

<style scoped>
.section {
  padding: 3rem 0;
  border-bottom: 1px solid #eee;
}

h2 {
  color: #2c3e50;
  margin-bottom: 1.5rem;
}

h3 {
  color: #34495e;
  margin: 1.5rem 0 1rem;
}

p {
  line-height: 1.6;
  color: #2c3e50;
}

ul {
  list-style: none;
  padding: 0;
}

li {
  margin: 0.5rem 0;
  color: #2c3e50;
}
</style>