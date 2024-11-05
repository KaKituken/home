<script setup>
const name = "Jixuan He (何吉轩)"
const title = "M.Eng. CS"
const institution = "Cornell University (Cornell Tech)"
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
            <a href="https://github.com/KaKituken" target="_blank" rel="noopener noreferrer"><i class="bi bi-github"></i></a>
            <a href="https://www.linkedin.com/in/jixuan-he/" target="_blank" rel="noopener noreferrer"><i class="bi bi-linkedin"></i></a>
            <a href="https://scholar.google.com/citations?user=xCIYDzsAAAAJ&hl=en" target="_blank" rel="noopener noreferrer"><i class="bi bi-google"></i></a>
        </div>
      </div>
    </div>
    <nav>
      <a href="#about">About</a>
      <a href="#publications">Publications</a>
      <a href="#research">Research</a>
      <a href="#contact">Contact</a>
      <router-link to="/gallery" class="gallery-link">Gallery</router-link>
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

.social-media i {
    font-size: 1.2rem;
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
  color: #42b883;
}

.gallery-link {
  color: #42b883;
}
</style>