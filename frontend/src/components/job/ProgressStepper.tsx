import { Check } from 'lucide-react'
import type { VideoJobStatus, JobProgress } from '@/api/contracts.ts'
import styles from './job.module.css'

const steps = [
  { key: 'upload', label: 'Yükleme' },
  { key: 'validate', label: 'Doğrulama' },
  { key: 'decode', label: 'GPU Decode' },
  { key: 'detect', label: 'Yüz Tespiti' },
  { key: 'track', label: 'Takip' },
  { key: 'recognize', label: 'Tanıma' },
  { key: 'reconcile', label: 'Birleştirme' },
  { key: 'render', label: 'Sonuç Videosu' },
  { key: 'complete', label: 'Tamamlandı' },
]

function statusToIndex(status: VideoJobStatus, stage?: string): number {
  switch (status) {
    case 'pending':
      return -1
    case 'uploading':
      return 0
    case 'validating':
      return 1
    case 'queued':
      return 2
    case 'processing':
      if (stage?.includes('tespit') || stage?.includes('takip')) return 3
      if (stage?.includes('tanıma')) return 5
      return 4
    case 'finalizing':
      return 6
    case 'rendering':
      return 7
    case 'completed':
      return 8
    case 'failed':
    case 'cancelled':
      return -1
    default:
      return -1
  }
}

interface ProgressStepperProps {
  status: VideoJobStatus
  progress?: JobProgress
}

export default function ProgressStepper({ status, progress }: ProgressStepperProps) {
  const activeIndex = statusToIndex(status, progress?.stage)

  return (
    <div className={styles.stepper} role="list" aria-label="İşlem aşamaları">
      {steps.map((step, index) => {
        const completed = activeIndex > index
        const active = activeIndex === index
        const stepClasses = [styles.step, completed ? styles.completed : '', active ? styles.active : '']
          .filter(Boolean)
          .join(' ')

        return (
          <div key={step.key} className={stepClasses} role="listitem" aria-current={active ? 'step' : undefined}>
            <div className={styles.badge}>
              {completed ? <Check size={14} aria-hidden="true" /> : index + 1}
            </div>
            <span className={styles.label}>{step.label}</span>
          </div>
        )
      })}
    </div>
  )
}
