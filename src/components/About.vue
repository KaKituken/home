<script setup>

</script>

<template>
  <section id="about" class="section">
    <div class="about-container">
      <h2>About Me</h2>
      <p>I am a first-year Master student at Cornell Tech. I recieved my B.E degree in Software Engineering from
        Tsinghua University on 2024.
        Previously, I worked with <a href="https://vcg.seas.harvard.edu/people/hanspeter-pfister">Prof. Hanspeter
          Pfister</a> at Harvard University Vision Computing Group on diffusion-based image editing and video
        understanding.
        I was also a student intern at System Intelligence Group, <a
          href="https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/">Microsoft Research Asia</a>
        working on LLM reasoning supervised by <a
          href="https://scholar.google.com/citations?user=UUyVnqgAAAAJ&hl=en">Dr. Xueting Han</a>.
        Currently, I am collaborating with <a href="https://faculty.ucmerced.edu/mhyang/">Prof. Ming-Hsuan Yang</a> on
        3D/4D editing tasks.<br><br>

        As a (not that professional) video creator, I think images and videos are fun! My research interest is enhancing
        multi-modal models' ability on visual reasoning tasks, such as image/video editing and vision fundation models.
        Besides, I'd love for you to subscribe to my <a
          href="https://space.bilibili.com/15085238?spm_id_from=333.1007.0.0">Bilibili</a> channel!
      </p>
      <div class="education">
        <h3>Education</h3>
        <ul>
          <li><b> M.Eng. in Computer Science</b>, Cornell Tech</li>
          <li><b>B.S. in Software Engineering</b>, Tsinghua University</li>
        </ul>
      </div>
    </div>
    <div class="news-container">
      <h2>News</h2>
      <ul>
        <li>🎯 <b>[2025.09]</b> One paper about 4D Content Creation accepeted by NeurIPS25!</li>
        <li>🎯 <b>[2025.08]</b> One paper about Image Composition accepeted by ICCV25 P13N Workshop!</li>
        <li>🎓 <b>[2025.06]</b> I graduated from Cornell Tech!</li>
        <li>🗽 <b>[2024.08]</b> I began my Master's journey @ Cornell Tech NYC!</li>
        <li>🎯 <b>[2024.07]</b> One paper about Video Temporal Grounding accepeted by ECCV24!</li>
        <li>🎓 <b>[2024.06]</b> I graduated from Tsinghua University!</li>
        <li>💻 <b>[2024.03]</b> I began my intership at Microsoft Research Asia!</li>
        <li>🏫 <b>[2023.06]</b> I joined VCG@Harvard Univeristy for summer internship!</li>
        <li>🦁 <b>[2023.01]</b> I joined National University of Singapore(NUS) for undergraduate exchange!</li>
      </ul>
    </div>
  </section>
</template>

<style scoped>
.section {
  display: flex;
  padding: 0rem 0;
  border-bottom: 1px solid #eee;
  gap: 3rem;
}

.about-container {
  flex: 3;
}

a {
  color: #228b22;
}

a:hover {
  color: #3dba56;
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
  text-align: justify;
}

ul {
  list-style: none;
  padding: 0;
}

li {
  margin: 0.5rem 0;
  color: #2c3e50;
}

.news-container {
  flex: 3;
  width: 10rem;
  height: 10rem;
  /* background-color: #228b22; */
}

.news-container ul {
  font-family: comic sans ms;
  font-size: small
}
</style>