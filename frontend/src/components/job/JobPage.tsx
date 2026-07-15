import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft } from 'lucide-react'
import { jobsApi } from '@/api/jobs.ts'
import Button from '@/components/ui/Button.tsx'
import { Card, CardBody } from '@/components/ui/Card.tsx'
import StatusBadge from '@/components/ui/StatusBadge.tsx'
import ErrorState from '@/components/ui/ErrorState.tsx'
import Skeleton from '@/components/ui/Skeleton.tsx'
import { formatDuration } from '@/lib/durationFormat.ts'
import { isTerminalStatus } from '@/lib/statusMapping.ts'
import ProgressStepper from './ProgressStepper.tsx'
import CancelDialog from './CancelDialog.tsx'
import ResultPage from '../result/ResultPage.tsx'
import styles from './job.module.css'

function ProgressMetrics({ progress }: { progress: NonNullable<ReturnType<typeof useJob>['data']>['progress'] }) {
  if (!progress) return null
  const items = [
    { label: 'İlerleme', value: `${progress.percent}%` },
    { label: 'İşlenen Kare', value: progress.processedFrames ?? '-' },
    { label: 'Tespit Edilen Yüz', value: progress.detectedFaces ?? '-' },
    { label: 'Geçen Süre', value: progress.elapsedSeconds ? formatDuration(progress.elapsedSeconds) : '-' },
  ]

  return (
    <div className={styles.metricGrid}>
      {items.map((m) => (
        <Card key={m.label} className={styles.metricCard}>
          <div className={styles.metricValue}>{m.value}</div>
          <div className={styles.metricLabel}>{m.label}</div>
        </Card>
      ))}
    </div>
  )
}

function useJob(jobId: string) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => jobsApi.get(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (!status || isTerminalStatus(status)) return false
      return 2_000
    },
    enabled: !!jobId,
  })
}

export default function JobPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data, isLoading, error, refetch } = useJob(jobId ?? '')

  const cancelMutation = useMutation({
    mutationFn: (id: string) => jobsApi.cancel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] })
    },
  })

  if (isLoading) {
    return (
      <div>
        <Skeleton height={28} width={300} />
        <Skeleton height={120} style={{ marginTop: 'var(--space-5)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <ErrorState
        title="İşlem yüklenemedi"
        message="İşlem bilgisi alınamadı. Lütfen tekrar deneyin."
        onRetry={refetch}
      />
    )
  }

  if (data.status === 'completed') {
    return <ResultPage job={data} />
  }

  return (
    <div>
      <Button variant="ghost" size="small" onClick={() => navigate('/')} style={{ marginBottom: 'var(--space-4)' }}>
        <ArrowLeft size={16} aria-hidden="true" />
        Geri
      </Button>
      <header className={styles.header}>
        <h1 className={styles.title}>{data.videoName ?? 'Video Analizi'}</h1>
        <p className={styles.subtitle}>
          İşlem: <code>{data.jobId}</code> · <StatusBadge status={data.status} />
        </p>
      </header>

      <Card>
        <CardBody>
          <ProgressStepper status={data.status} progress={data.progress} />
          <ProgressMetrics progress={data.progress} />

          {data.status === 'failed' ? (
            <ErrorState
              title="İşlem başarısız oldu"
              message={data.error?.message ?? 'İşlem sırasında beklenmedik bir hata oluştu.'}
              code={data.error?.code}
            />
          ) : data.status === 'cancelled' ? (
            <ErrorState title="İşlem iptal edildi" message="Bu işlem kullanıcı tarafından iptal edildi." />
          ) : null}

          <div className={styles.actions}>
            {!isTerminalStatus(data.status) ? (
              <CancelDialog
                jobId={data.jobId}
                onConfirm={() => cancelMutation.mutate(data.jobId)}
                disabled={cancelMutation.isPending}
              />
            ) : null}
          </div>
        </CardBody>
      </Card>
    </div>
  )
}
