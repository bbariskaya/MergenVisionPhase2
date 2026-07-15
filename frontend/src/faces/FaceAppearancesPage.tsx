import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Clock, Video } from 'lucide-react'
import { facesApi } from '@/api/faces.ts'
import { Card, CardBody, CardHeader } from '@/components/ui/Card.tsx'
import StatusBadge from '@/components/ui/StatusBadge.tsx'
import ErrorState from '@/components/ui/ErrorState.tsx'
import Skeleton from '@/components/ui/Skeleton.tsx'
import { formatDuration } from '@/lib/durationFormat.ts'
import styles from './faceAppearances.module.css'

export default function FaceAppearancesPage() {
  const { faceId } = useParams<{ faceId: string }>()
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['face', faceId, 'appearances'],
    queryFn: () => facesApi.appearances(faceId ?? ''),
    enabled: !!faceId,
  })

  if (isLoading) {
    return (
      <div>
        <Skeleton height={28} width={260} />
        <Skeleton height={160} style={{ marginTop: 'var(--space-5)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <ErrorState
        title="Yüz geçmişi yüklenemedi"
        message="Bu yüze ait geçmiş bilgisi alınamadı."
        onRetry={refetch}
      />
    )
  }

  return (
    <div>
      <Link to="/" className={styles.backLink}>
        <ArrowLeft size={16} aria-hidden="true" />
        Genel Bakışa Dön
      </Link>

      <header className={styles.header}>
        <h1 className={styles.title}>{data.name ?? data.faceId}</h1>
        <div className={styles.subtitle}>
          <StatusBadge status={data.status} />
          <span>{data.totalVideos} videoda görünüyor</span>
        </div>
      </header>

      <Card>
        <CardHeader title="Görünümler" />
        <CardBody>
          {data.appearances.length === 0 ? (
            <p className={styles.empty}>Kayıtlı görünüm bulunmuyor.</p>
          ) : (
            <ul className={styles.list}>
              {data.appearances.map((app, idx) => (
                <li key={idx} className={styles.item}>
                  <div className={styles.itemIcon} aria-hidden="true">
                    <Video size={18} />
                  </div>
                  <div className={styles.itemContent}>
                    <div className={styles.itemTitle}>{app.videoName ?? app.jobId}</div>
                    <div className={styles.itemMeta}>
                      <Clock size={12} aria-hidden="true" />
                      {formatDuration(app.start)} - {formatDuration(app.end)} (kare{' '}
                      {app.startFrame}-{app.endFrame})
                      <span style={{ margin: '0 var(--space-1)' }}>·</span>
                      <StatusBadge status={app.status} />
                    </div>
                  </div>
                  <Link to={`/videos/jobs/${app.jobId}`} className={styles.itemLink}>
                    Videoya Git
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </CardBody>
      </Card>
    </div>
  )
}
