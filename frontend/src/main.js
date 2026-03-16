import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

const app = createApp(App)

app.use(router)

// Global error handler — prevents silent crashes in component lifecycle
app.config.errorHandler = (err, instance, info) => {
  console.error(`[MiroFish] Unhandled error in ${info}:`, err)
}

app.mount('#app')
