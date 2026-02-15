import { Navigate, Route, Routes } from 'react-router-dom'

import './App.css'
import HomePage from './pages/HomePage'
import RenderPage from './pages/RenderPage'

function App() {
  return (
    <div className="route-shell">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/render" element={<RenderPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  )
}

export default App
