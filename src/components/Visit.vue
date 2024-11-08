<script setup>
const contact = {
  email: "jh2926 [AT] cornell [DOT] edu",
  office: "Computer Science Building, Room 405",
  address: "123 University Avenue, City, State 12345",
  phone: "(123) 456-7890"
}
</script>

<template>
  <section id="visit" class="section">
    <h2>Also Visit</h2>
    <div class="contact-info">
      <div class="contact-item">
        <strong>Yaqing Yang:</strong>
        <a href="https://yyqbeatrice.github.io/yaqing">https://yyqbeatrice.github.io/yaqing/</a>
      </div>
      <div class="contact-item">
        <strong>Siwei Li:</strong>
        <a href="https://www.siweili.tech/">https://www.siweili.tech/</a>
      </div>
    </div>
  </section>
</template>

<style scoped>
.section {
  padding: 3rem 0;
}

.contact-info {
  max-width: 600px;
  margin: 2rem 0;
}

.contact-item {
  margin: 1rem 0;
  display: flex;
  gap: 1rem;
}

.contact-item strong {
  /* min-width: 80px; */
  color: #2c3e50;
}

.contact-item a {
  color: #42b883;
  text-decoration: none;
}

.contact-item a:hover {
  text-decoration: underline;
}
</style>