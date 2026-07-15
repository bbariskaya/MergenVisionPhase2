import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Plus } from 'lucide-react'
import { jobsApi } from '@/api/jobs.ts'
import Button from '@/components/ui/Button.tsx'
import { Card, CardBody, CardHeader } from '@/components/ui/Card.tsx'
import EmptyState from '@/components/ui/EmptyState.tsx'
import ErrorState from '@/components/ui/ErrorState.tsx'
import Skeleton from '@/components/ui/Skeleton.tsx'
import StatusBadge from '@/components/ui/StatusBadge.tsx'
import { formatDuration } from '@/lib/durationFormat.ts'
import styles from './dashboard.module.css'

function useJobs() {
  return useQuery({
    queryKey: ['jobs'],
    queryFn: jobsApi.list,
    refetchInterval: 2_000,
  })
}

function MetricCards({ jobs }: { jobs: Awaited<ReturnType<typeof jobsApi.list>> }) {
  const total = jobs.length
  const completed = jobs.filter((j) => j.status === 'completed').length
  const failedOrCancelled = jobs.filter((j) => j.status === 'failed' || j.status === 'cancelled').length
  const recognisedPersons = jobs.reduce((sum, j) => sum + (j.personCount ?? 0), 0)

  const items = [
    { label: 'Toplam Video', value: total },
    { label: 'Tamamlanan İş', value: completed },
    { label: 'Tanınan Kişi', value: recognisedPersons },
    { label: 'Başarısız / İptal', value: failedOrCancelled },
  ]

  if (total === 0) {
    return (
      <div className={styles.metrics}>
        {items.map((m) => (
          <Card key={m.label} className={styles.metricCard}>
            <div className={styles.metricValue}>-</div>
            <div className={styles.metricLabel}>{m.label}</div>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className={styles.metrics}>
      {items.map((m) => (
        <Card key={m.label} className={styles.metricCard}>
          <div className={styles.metricValue}>{m.value}</div>
          <div className={styles.metricLabel}>{m.label}</div>
        </Card>
      ))}
    </div>
  )
}

function JobsTable({ jobs }: { jobs: Awaited<ReturnType<typeof jobsApi.list>> }) {
  if (jobs.length === 0) {
    return (
      <EmptyState
        title="Henüz işlem yok"
        message="Analiz edilmiş video bulunmuyor. Yeni bir video ekleyerek başlayın."
        action={
          <Link to="/videos/new">
            <Button>
              <Plus size={16} aria-hidden="true" />
              Yeni Video Analizi
            </Button>
          </Link>
        }
      />
    )
  }

  return (
    <table className={styles.table}>
      <thead>
        <tr>
          <th>Durum</th>
          <th>Video Adı</th>
          <th>Oluşturulma</th>
          <th>Süre</th>
          <th>Kişi</th>
          <th>İlerleme</th>
          <th>Detay</th>
        </tr>
      </thead>
      <tbody>
        {jobs.map((job) => (
          <tr key={job.jobId}>
            <td>
              <StatusBadge status={job.status} />
            </td>
            <td className="truncate" style={{ maxWidth: 240 }}>
              {job.videoName ?? 'Bilinmeyen video'}
            </td>
            <td>{new Date(job.createdAt).toLocaleString('tr-TR')}</td>
            <td>{job.durationSeconds ? formatDuration(job.durationSeconds) : '-'}</td>
            <td>{job.personCount ?? '-'}</td>
            <td>
              <div className={styles.progressCell}>
                <div className={styles.progressBar} aria-hidden="true">
                  <div
                    className={styles.progressFill}
                    style={{ width: `${job.progressPercent ?? 0}%` }}
                  />
                </div>
                <span>{job.progressPercent ?? 0}%</span>
              </div>
            </td>
            <td>
              <Link to={`/videos/jobs/${job.jobId}`} className={styles.link}>
                Görüntüle
              </Link>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function DashboardPage() {
  const { data, isLoading, error, refetch } = useJobs()

  return (
    <div>
      <header className={styles.header}>
        <h1 className={styles.title}>Video Analiz Merkezi</h1>
        <p className={styles.subtitle}>
          Yüklenen videoların yüz analiz sonuçlarını takip edin ve yeni analizler
          başlatın.
        </p>
      </header>

      {isLoading ? (
        <div className={styles.metrics}>
          <Skeleton height={96} />
          <Skeleton height={96} />
          <Skeleton height={96} />
          <Skeleton height={96} />
        </div>
      ) : data ? (
        <MetricCards jobs={data} />
      ) : null}

      <Card>
        <CardHeader
          title="Son İşlemler"
          action={
            <Link to="/videos/new">
              <Button>
                <Plus size={16} aria-hidden="true" />
                Yeni Video Analizi
              </Button>
            </Link>
          }
        />
        <CardBody>
          {error ? (
            <ErrorState
              title="İşlemler yüklenemedi"
              message="Sunucudan işlem listesi alınamadı. Lütfen tekrar deneyin."
              onRetry={refetch}
            />
          ) : isLoading ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
              <Skeleton height={40} />
              <Skeleton height={40} />
              <Skeleton height={40} />
            </div>
          ) : data ? (
            <JobsTable jobs={data} />
          ) : null}
        </CardBody>
      </Card>
    </div>
  )
}
