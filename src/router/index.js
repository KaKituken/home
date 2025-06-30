import { createRouter, createWebHashHistory } from 'vue-router'
// import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Gallery from '../views/Gallery.vue'
import CVView from '@/views/CVView.vue'
import BlogList from '@/views/BlogList.vue'
import BlogPost from '@/views/BlogPost.vue'

const router = createRouter({
  history: createWebHashHistory('/home/'), // 使用 hash 模式，并加上 /home/ 前缀
  // history: createWebHistory('/'),
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
    },
    {
      path: '/cv', // 新增 CV 路由
      name: 'cv',
      component: CVView
    },
    {
      path: '/blog',
      name: 'blog',
      component: BlogList
    },
    {
      path: '/blog/:id',
      name: 'blog-post',
      component: BlogPost,
      props: true
    }
  ],
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    } else if (to.hash) {
      return {
        el: to.hash,
        behavior: 'smooth', // 让滚动更平滑
      }
    } else {
      return { top: 0 }
    }
  }
})

export default router