<script setup>
const name = "Jason Jixuan He (何吉轩)"
const title = "M.Eng. CS"
const institution = "Cornell University (Cornell Tech)"

const props = defineProps({
  scrollToSection: Function
});
</script>

<template>
  <header class="header">
    <div class="profile">
      <div class="profile-image">
        <img src="../assets/profile_edit.jpg">
      </div>
      <div class="profile-info">
        <h1>{{ name }}</h1>
        <h2>{{ title }}</h2>
        <h3>{{ institution }}</h3>
        <div class="social-media">
            <a href="mailto:jh2926@cornell.edu" class="icon-link"><i class="fa-solid fa-envelope"></i></a>
            <a href="https://github.com/KaKituken" target="_blank" rel="noopener noreferrer"><i class="fa-brands fa-github"></i></a>
            <a href="https://www.linkedin.com/in/jixuan-he/" target="_blank" rel="noopener noreferrer"><i class="fa-brands fa-linkedin"></i></a>
            <a href="https://scholar.google.com/citations?user=xCIYDzsAAAAJ&hl=en" target="_blank" rel="noopener noreferrer"><i class="fa-brands fa-google-scholar"></i></a>
            <a href="https://x.com/kirino_ken" target="_blank" rel="noopener noreferrer"><i class="fa-brands fa-x-twitter"></i></a>
        </div>
      </div>
    </div>
    <nav>
      <a href="#about" @click.prevent="scrollToSection('about')">About</a>
      <a href="#publications" @click.prevent="scrollToSection('publications')">Publications</a>
      <a href="#research" @click.prevent="scrollToSection('research')">Research</a>
      <router-link to="/cv" class="gallery-link">CV</router-link>
      <router-link to="/blog" class="gallery-link">Blog</router-link>
      <!-- <router-link to="/gallery" class="gallery-link">Gallery</router-link> -->
    </nav>
  </header>
</template>

<style scoped>

.header {
  padding: 2rem 0;
  border-bottom: 1px solid #eee;
}

.profile {
  display: flex;
  align-items: center;
  gap: 2rem;
  margin-bottom: 2rem;
}

.profile-image {
  width: 150px;
  height: 150px;
  border-radius: 50%;
  background-color: #f0f0f0;
  overflow: hidden;
}

.profile-image img {
  width: 100%;
  height: 100%;
  object-fit: cover; 
}

.profile-info h1 {
  font-size: 2.5rem;
  margin: 0;
  color: #2c3e50;
}

.profile-info h2 {
  font-size: 1.5rem;
  margin: 0.5rem 0;
  color: #34495e;
  font-weight: normal;
}

.profile-info h3 {
  font-size: 1.2rem;
  margin: 0;
  color: #7f8c8d;
  font-weight: normal;
}

.social-media{
    /* height: 10rem;
    width: 10rem; */
    display: flex;
    /* background-color: #eee; */
    gap: 0.5rem;
}

.social-media a {
  text-align: center;
}

.social-media i {
    font-size: 1.2rem;
    text-align: center;
    color: black;
    line-height: inherit;
}

.social-media i:hover {
  color: #228b22;
}

nav {
  display: flex;
  gap: 2rem;
}

nav a {
  color: #2c3e50;
  text-decoration: none;
  font-weight: 500;
  padding: 0.5rem 0;
}

nav a:hover, nav a.router-link-active {
  color: #228b22;
}

.gallery-link {
  color: #2c3e50;
}
</style>