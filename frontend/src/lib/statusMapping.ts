import type { VideoJobStatus, IdentityStatus } from '@/api/contracts.ts'

export const jobStatusLabel: Record<VideoJobStatus, string> = {
  pending: 'Bekliyor',
  uploading: 'Yükleniyor',
  validating: 'Doğrulanıyor',
  queued: 'Sırada',
  processing: 'İşleniyor',
  finalizing: 'Sonuçlandırılıyor',
  rendering: 'Video Çıktısı',
  completed: 'Tamamlandı',
  failed: 'Başarısız',
  cancelled: 'İptal Edildi',
}

export const identityStatusLabel: Record<IdentityStatus, string> = {
  known: 'Tanındı',
  anonymous: 'Anonim',
  new_anonymous: 'Yeni Anonim',
  unknown: 'Bilinmiyor',
}

export function isTerminalStatus(status: VideoJobStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}
