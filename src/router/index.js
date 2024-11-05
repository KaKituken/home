import { createRouter, createWebHashHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Gallery from '../views/Gallery.vue'

const router = createRouter({
  history: createWebHashHistory('/home/'), // 使用 hash 模式，并加上 /home/ 前缀
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home
    },
    {
      path: '/gallery',
      name: 'gallery',
      component: Gallery
    }
  ]
})

export default router