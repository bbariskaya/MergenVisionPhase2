import type { ReactNode } from 'react'
import styles from './card.module.css'

interface CardProps {
  children: ReactNode
  className?: string
}

interface CardHeaderProps {
  title: ReactNode
  subtitle?: ReactNode
  action?: ReactNode
}

export function Card({ children, className = '' }: CardProps) {
  return <div className={`${styles.card} ${className}`}>{children}</div>
}

export function CardHeader({ title, subtitle, action }: CardHeaderProps) {
  return (
    <div className={styles.header}>
      <div>
        <h3 className={styles.title}>{title}</h3>
        {subtitle ? <p className={styles.subtitle}>{subtitle}</p> : null}
      </div>
      {action ? <div>{action}</div> : null}
    </div>
  )
}

export function CardBody({ children }: { children: ReactNode }) {
  return <div className={styles.body}>{children}</div>
}
