import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { UploadCloud, FileVideo, ChevronDown, ChevronUp, X } from 'lucide-react'
import { videosApi } from '@/api/videos.ts'
import Button from '@/components/ui/Button.tsx'
import { Card, CardBody } from '@/components/ui/Card.tsx'
import Input from '@/components/ui/Input.tsx'
import Select from '@/components/ui/Select.tsx'
import ErrorState from '@/components/ui/ErrorState.tsx'
import { normalizeError } from '@/api/errors.ts'
import styles from './upload.module.css'

const ACCEPTED_TYPES = [
  'video/mp4',
  'video/quicktime',
  'video/x-msvideo',
  'video/avi',
  'video/webm',
]

const MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024 // 2 GB

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Number((bytes / k ** i).toFixed(1))} ${sizes[i]}`
}

function DropZone({
  file,
  error,
  onFileSelect,
  onClear,
}: {
  file: File | null
  error?: string
  onFileSelect: (file: File) => void
  onClear: () => void
}) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const validateAndSet = useCallback(
    (candidate: File) => {
      if (!ACCEPTED_TYPES.includes(candidate.type)) {
        return 'Desteklenmeyen dosya formatı. Lütfen MP4, MOV, AVI veya WEBM yükleyin.'
      }
      if (candidate.size > MAX_FILE_BYTES) {
        return `Dosya boyutu ${formatBytes(MAX_FILE_BYTES)} limitini aşıyor.`
      }
      onFileSelect(candidate)
      return undefined
    },
    [onFileSelect],
  )

  const handleDropFile = (candidate: File | undefined) => {
    if (!candidate) return
    validateAndSet(candidate)
    setIsDragging(false)
  }

  return (
    <div
      className={`${styles.dropZone} ${isDragging ? styles.active : ''} ${
        error && !file ? styles.dropZoneError : ''
      }`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        handleDropFile(e.dataTransfer.files?.[0])
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
      aria-label="Video dosyası seçmek için tıklayın veya sürükleyip bırakın"
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(',')}
        onChange={(e) => handleDropFile(e.target.files?.[0])}
        className="sr-only"
      />
      {file ? (
        <>
          <FileVideo size={40} aria-hidden="true" />
          <div className={styles.fileName}>{file.name}</div>
          <div className={styles.fileMeta}>
            {formatBytes(file.size)} · {file.type || 'video'}
          </div>
          <Button
            type="button"
            variant="ghost"
            size="small"
            onClick={(e) => {
              e.stopPropagation()
              onClear()
            }}
            style={{ marginTop: 'var(--space-3)' }}
          >
            <X size={14} aria-hidden="true" />
            Kaldır
          </Button>
        </>
      ) : (
        <>
          <UploadCloud size={40} aria-hidden="true" />
          <div className={styles.fileName}>Video dosyası sürükleyin veya seçin</div>
          <div className={styles.hint}>MP4, MOV, AVI, WEBM · max {formatBytes(MAX_FILE_BYTES)}</div>
        </>
      )}
    </div>
  )
}

export default function UploadPage() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string>()
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [formError, setFormError] = useState<string>()

  const [samplingRate, setSamplingRate] = useState('every_5th_frame')
  const [minFaceSize, setMinFaceSize] = useState(80)
  const [profile, setProfile] = useState<'accuracy' | 'balanced' | 'fast'>('balanced')

  const mutation = useMutation({
    mutationFn: (payload: { file: File }) =>
      videosApi.recognize(payload.file, {
        samplingRate,
        minFaceSize,
        profile,
      }),
    onSuccess: (job) => {
      navigate(`/videos/jobs/${job.jobId}`)
    },
    onError: (error) => {
      const normalized = normalizeError(error)
      setFormError(normalized.message)
    },
  })

  const handleFileSelect = (selected: File) => {
    setFile(selected)
    setFileError(undefined)
  }

  const handleClear = () => {
    setFile(null)
    setFileError(undefined)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setFormError(undefined)
    if (!file) {
      setFileError('Lütfen bir video dosyası seçin.')
      return
    }
    mutation.mutate({ file })
  }

  return (
    <div>
      <header className={styles.header}>
        <h1 className={styles.title}>Yeni Video Analizi</h1>
        <p className={styles.subtitle}>
          Analiz edilecek videoyu yükleyin ve işlem parametrelerini seçin.
        </p>
      </header>

      <Card>
        <CardBody>
          {formError ? (
            <div style={{ marginBottom: 'var(--space-5)' }}>
              <ErrorState
                title="Yükleme başarısız"
                message={formError}
              />
            </div>
          ) : null}

          <form onSubmit={handleSubmit}>
            <DropZone
              file={file}
              error={fileError}
              onFileSelect={handleFileSelect}
              onClear={handleClear}
            />
            {fileError ? (
              <p role="alert" style={{ color: 'var(--color-danger)', fontSize: 12, marginTop: 'var(--space-2)' }}>
                {fileError}
              </p>
            ) : null}

            <div className={styles.formGrid}>
              <Select
                label="Örnekleme Oranı"
                value={samplingRate}
                onChange={(e) => setSamplingRate(e.target.value)}
                options={[
                  { value: 'every_1st_frame', label: 'Her kare' },
                  { value: 'every_3rd_frame', label: 'Her 3. kare' },
                  { value: 'every_5th_frame', label: 'Her 5. kare' },
                  { value: 'every_10th_frame', label: 'Her 10. kare' },
                ]}
              />
              <Select
                label="İşlem Profili"
                value={profile}
                onChange={(e) => setProfile(e.target.value as typeof profile)}
                options={[
                  { value: 'accuracy', label: 'Doğruluk' },
                  { value: 'balanced', label: 'Dengeli' },
                  { value: 'fast', label: 'Hızlı' },
                ]}
              />
              <div>
                <Input
                  label="Minimum Yüz Boyutu (piksel)"
                  type="number"
                  min={32}
                  max={512}
                  value={minFaceSize}
                  onChange={(e) => setMinFaceSize(Number(e.target.value))}
                />
              </div>
            </div>

            <div className={styles.advanced}>
              <button
                type="button"
                className={styles.advancedHeader}
                onClick={() => setAdvancedOpen((v) => !v)}
                aria-expanded={advancedOpen}
              >
                <span>Gelişmiş Ayarlar</span>
                {advancedOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
              {advancedOpen ? (
                <div className={styles.advancedBody}>
                  <Input label="Maksimum İşlem Süresi (dk)" type="number" disabled defaultValue={30} />
                  <Input label="Galeri Eşiği" type="number" disabled defaultValue={0.6} />
                </div>
              ) : null}
            </div>

            <div className={styles.actions}>
              <Button
                type="button"
                variant="secondary"
                onClick={() => navigate('/')}
                disabled={mutation.isPending}
              >
                İptal
              </Button>
              <Button type="submit" disabled={mutation.isPending || !file}>
                <UploadCloud size={16} aria-hidden="true" />
                {mutation.isPending ? 'Gönderiliyor...' : 'Analizi Başlat'}
              </Button>
            </div>
          </form>
        </CardBody>
      </Card>
    </div>
  )
}
