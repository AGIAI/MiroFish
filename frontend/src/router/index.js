import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Process from '../views/MainView.vue'
import SimulationView from '../views/SimulationView.vue'
import SimulationRunView from '../views/SimulationRunView.vue'
import ReportView from '../views/ReportView.vue'
import InteractionView from '../views/InteractionView.vue'

// Validates that a route param looks like a plausible ID (UUID, "new", or alphanumeric slug)
const validId = /^[a-zA-Z0-9_-]{1,64}$/

function requireValidId(paramName) {
  return (to) => {
    const id = to.params[paramName]
    if (!id || !validId.test(id)) {
      return { name: 'Home' }
    }
    return true
  }
}

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/process/:projectId',
    name: 'Process',
    component: Process,
    props: true,
    beforeEnter: requireValidId('projectId')
  },
  {
    path: '/simulation/:simulationId',
    name: 'Simulation',
    component: SimulationView,
    props: true,
    beforeEnter: requireValidId('simulationId')
  },
  {
    path: '/simulation/:simulationId/start',
    name: 'SimulationRun',
    component: SimulationRunView,
    props: true,
    beforeEnter: requireValidId('simulationId')
  },
  {
    path: '/report/:reportId',
    name: 'Report',
    component: ReportView,
    props: true,
    beforeEnter: requireValidId('reportId')
  },
  {
    path: '/interaction/:reportId',
    name: 'Interaction',
    component: InteractionView,
    props: true,
    beforeEnter: requireValidId('reportId')
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    redirect: { name: 'Home' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
