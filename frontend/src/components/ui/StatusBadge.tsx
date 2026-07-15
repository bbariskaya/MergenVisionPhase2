import {
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Clock,
  UploadCloud,
} from 'lucide-react'
import type { VideoJobStatus, IdentityStatus } from '@/api/contracts.ts'
import styles from './statusBadge.module.css'

interface StatusBadgeProps {
  status: VideoJobStatus | IdentityStatus
  showLabel?: boolean
}

const jobStatusConfig: Record<
  VideoJobStatus,
  { label: string; icon: typeof Clock; className: string }
> = {
  pending: { label: 'Bekliyor', icon: Clock, className: styles.pending },
  uploading: { label: 'Yükleniyor', icon: UploadCloud, className: styles.uploading },
  validating: { label: 'Doğrulanıyor', icon: Loader2, className: styles.validating },
  queued: { label: 'Sırada', icon: Clock, className: styles.queued },
  processing: { label: 'İşleniyor', icon: Loader2, className: styles.processing },
  finalizing: { label: 'Sonuçlandırılıyor', icon: Loader2, className: styles.finalizing },
  rendering: { label: 'Video Çıktısı', icon: Loader2, className: styles.rendering },
  completed: { label: 'Tamamlandı', icon: CheckCircle2, className: styles.completed },
  failed: { label: 'Başarısız', icon: XCircle, className: styles.failed },
  cancelled: { label: 'İptal Edildi', icon: AlertCircle, className: styles.cancelled },
}

const identityStatusConfig: Record<
  IdentityStatus,
  { label: string; className: string }
> = {
  known: { label: 'Tanındı', className: styles.completed },
  anonymous: { label: 'Anonim', className: styles.unknown },
  new_anonymous: { label: 'Yeni Anonim', className: styles.processing },
  unknown: { label: 'Bilinmiyor', className: styles.unknown },
}

function isJobStatus(status: string): status is VideoJobStatus {
  return status in jobStatusConfig
}

export default function StatusBadge({
  status,
  showLabel = true,
}: StatusBadgeProps) {
  if (isJobStatus(status)) {
    const config = jobStatusConfig[status]
    const Icon = config.icon
    return (
      <span
        className={`${styles.badge} ${config.className}`}
        data-status={status}
        aria-label={`Durum: ${config.label}`}
      >
        <Icon size={14} aria-hidden="true" />
        {showLabel ? <span>{config.label}</span> : <span className="sr-only">{config.label}</span>}
      </span>
    )
  }

  const config = identityStatusConfig[status]
  return (
    <span
      className={`${styles.badge} ${config.className}`}
      data-status={status}
      aria-label={`Kimlik durumu: ${config.label}`}
    >
      <span className={styles.dot} aria-hidden="true" />
      {showLabel ? <span>{config.label}</span> : <span className="sr-only">{config.label}</span>}
    </span>
  )
}
