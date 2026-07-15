import { Routes, Route, Navigate } from 'react-router-dom'
import AppLayout from '@/components/layout/AppLayout.tsx'
import DashboardPage from '@/components/dashboard/DashboardPage.tsx'
import UploadPage from '@/components/upload/UploadPage.tsx'
import JobPage from '@/components/job/JobPage.tsx'
import FaceAppearancesPage from '@/faces/FaceAppearancesPage.tsx'

function NotFound() {
  return (
    <div style={{ padding: 'var(--space-8)' }}>
      <h1>Sayfa bulunamadı</h1>
      <p>Aradığınız sayfa mevcut değil.</p>
    </div>
  )
}

export default function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/videos/new" element={<UploadPage />} />
        <Route path="/videos/jobs/:jobId" element={<JobPage />} />
        <Route path="/faces/:faceId" element={<FaceAppearancesPage />} />
        <Route path="/404" element={<NotFound />} />
        <Route path="*" element={<Navigate to="/404" replace />} />
      </Route>
    </Routes>
  )
}
