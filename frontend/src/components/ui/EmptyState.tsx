import type { ReactNode } from 'react'
import { Inbox } from 'lucide-react'
import styles from './emptyState.module.css'

interface EmptyStateProps {
  title?: string
  message?: string
  icon?: ReactNode
  action?: ReactNode
}

export default function EmptyState({
  title = 'Henüz veri yok',
  message = 'Bu alanda görüntülenecek bir içerik bulunamadı.',
  icon,
  action,
}: EmptyStateProps) {
  return (
    <div className={styles.empty} role="status">
      <div className={styles.icon}>{icon ?? <Inbox size={40} aria-hidden="true" />}</div>
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.message}>{message}</p>
      {action ? <div style={{ marginTop: 'var(--space-5)' }}>{action}</div> : null}
    </div>
  )
}
