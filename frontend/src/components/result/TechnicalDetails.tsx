import { Download } from 'lucide-react'
import type { VideoJob } from '@/api/contracts.ts'
import Button from '@/components/ui/Button.tsx'
import styles from './technicalDetails.module.css'

interface TechnicalDetailsProps {
  job: VideoJob
}

export default function TechnicalDetails({ job }: TechnicalDetailsProps) {
  const handleDownload = () => {
    if (!job.result) return
    const blob = new Blob([JSON.stringify(job.result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${job.jobId}_result.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const audit = job.result?.audit ?? {}
  const mappings = (audit as { rawTrackerMappings?: Record<string, unknown> }).rawTrackerMappings ?? {}

  return (
    <details className={styles.details}>
      <summary className={styles.summary}>Teknik Detaylar ve Audit</summary>
      <div className={styles.body}>
        <div className={styles.grid}>
          <Field label="Job ID" value={job.jobId} />
          <Field label="Process ID" value={job.processId} />
          <Field label="Worker ID" value={job.progress?.workerId ?? '-'} />
          <Field label="GPU UUID" value={job.progress?.gpuUuid ?? '-'} />
          <Field label="Model (Detector)" value={(audit as { modelVersions?: { detector?: string } }).modelVersions?.detector ?? '-'} />
          <Field label="Model (Recognizer)" value={(audit as { modelVersions?: { recognizer?: string } }).modelVersions?.recognizer ?? '-'} />
          <Field label="Toplam Kare" value={String(job.video?.totalFrames ?? '-')} />
          <Field label="İşlenen Kare" value={String(job.video?.processedFrames ?? '-')} />
          <Field label="Örnekleme" value={job.video?.samplingRate ?? '-'} />
        </div>

        <div className={styles.field}>
          <div className={styles.fieldLabel}>Raw Tracker → Canonical Track Mapping</div>
          <pre>{JSON.stringify(mappings, null, 2)}</pre>
        </div>

        <div className={styles.download}>
          <Button variant="secondary" onClick={handleDownload} disabled={!job.result}>
            <Download size={14} aria-hidden="true" />
            Sonuç JSON&apos;unu İndir
          </Button>
        </div>
      </div>
    </details>
  )
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.field}>
      <div className={styles.fieldLabel}>{label}</div>
      <div className={styles.fieldValue}>{value}</div>
    </div>
  )
}
