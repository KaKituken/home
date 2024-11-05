<script setup>
const contact = {
  email: "jh2926 [AT] cornell [DOT] edu",
  office: "Computer Science Building, Room 405",
  address: "123 University Avenue, City, State 12345",
  phone: "(123) 456-7890"
}
</script>

<template>
  <section id="contact" class="section">
    <h2>Contact</h2>
    <div class="contact-info">
      <div class="contact-item">
        <strong>Email:</strong>
        <a :href="'mailto:' + contact.email">{{ contact.email }}</a>
      </div>
      <!-- <div class="contact-item">
        <strong>Office:</strong>
        <span>{{ contact.office }}</span>
      </div>
      <div class="contact-item">
        <strong>Address:</strong>
        <span>{{ contact.address }}</span>
      </div> -->
      <!-- <div class="contact-item">
        <strong>Phone:</strong>
        <span>{{ contact.phone }}</span>
      </div> -->
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
  min-width: 80px;
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