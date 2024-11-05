<script setup>
import Header from '../components/Header.vue'
import About from '../components/About.vue'
import Publications from '../components/Publications.vue'
import Research from '../components/Research.vue'
import Contact from '../components/Contact.vue'
</script>

<template>
  <div class="container">
    <Header />
    <main>
      <About />
      <Publications />
      <Research />
      <Contact />
    </main>
  </div>
</template>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

main {
  margin-top: 2rem;
}
</style>