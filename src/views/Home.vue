<script setup>
import Header from '../components/Header.vue'
import About from '../components/About.vue'
import Publications from '../components/Publications.vue'
import Research from '../components/Research.vue'
import Visit from '../components/Visit.vue'

function scrollToSection(sectionId) {
  const element = document.getElementById(sectionId);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth' });
  }
}
</script>

<template>
  <div class="container">
    <Header :scrollToSection="scrollToSection"/>
    <main>
      <About />
      <Publications />
      <Research />
      <Visit />
    </main>
  </div>
</template>

<style scoped>
.container {
  /* max-width: 50vw; */
  max-width: 1000px;
  margin: 0 auto;
  padding: 0 2rem;
}

main {
  margin-top: 2rem;
}
</style>