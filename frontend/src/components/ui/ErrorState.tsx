import { AlertTriangle, RefreshCw } from 'lucide-react'
import Button from './Button.tsx'
import styles from './errorState.module.css'

interface ErrorStateProps {
  title?: string
  message?: string
  code?: string
  onRetry?: () => void
}

export default function ErrorState({
  title = 'Bir hata oluştu',
  message = 'İşleminiz sırasında beklenmedik bir sorun oluştu. Lütfen tekrar deneyin.',
  code,
  onRetry,
}: ErrorStateProps) {
  return (
    <div className={styles.error} role="alert">
      <div className={styles.icon}>
        <AlertTriangle size={40} aria-hidden="true" />
      </div>
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.message}>{message}</p>
      {code ? <div className={styles.code}>Kod: {code}</div> : null}
      {onRetry ? (
        <Button onClick={onRetry} variant="secondary">
          <RefreshCw size={14} aria-hidden="true" />
          Tekrar Dene
        </Button>
      ) : null}
    </div>
  )
}
