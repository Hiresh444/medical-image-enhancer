import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import UploadPage from './pages/UploadPage'
import RunsListPage from './pages/RunsListPage'
import RunDetailPage from './pages/RunDetailPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<UploadPage />} />
          <Route path="/runs" element={<RunsListPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
