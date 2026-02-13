import { NavLink, Navigate, Route, Routes } from 'react-router-dom'

import './App.css'
import HomePage from './pages/HomePage'
import RenderPage from './pages/RenderPage'

function App() {
  return (
    <div className="route-shell">
      <header className="route-header">
        <nav className="route-nav" aria-label="Primary">
          <NavLink
            to="/"
            end
            className={({ isActive }) => (isActive ? 'route-link active' : 'route-link')}
          >
            Home
          </NavLink>
          <NavLink
            to="/render"
            className={({ isActive }) => (isActive ? 'route-link active' : 'route-link')}
          >
            Render
          </NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/render" element={<RenderPage />} />
        <Route path="*" element={<Navigate to="/render" replace />} />
      </Routes>
    </div>
  )
}

export default App
