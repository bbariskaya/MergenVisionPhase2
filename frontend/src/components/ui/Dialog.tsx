import { useEffect, useRef, type ReactNode } from 'react'
import { X } from 'lucide-react'
import styles from './dialog.module.css'

interface DialogProps {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  footer?: ReactNode
}

export default function Dialog({ open, onClose, title, children, footer }: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) {
      dialog.showModal()
    } else if (!open && dialog.open) {
      dialog.close()
    }
  }, [open])

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    const handleCancel = (event: Event) => {
      event.preventDefault()
      onClose()
    }
    dialog.addEventListener('cancel', handleCancel)
    return () => dialog.removeEventListener('cancel', handleCancel)
  }, [onClose])

  return (
    <dialog ref={ref} className={styles.dialog} onClose={onClose}>
      <div className={styles.header}>
        <h2 className={styles.title}>{title}</h2>
        <button
          type="button"
          className={styles.close}
          onClick={onClose}
          aria-label="Kapat"
        >
          <X size={18} aria-hidden="true" />
        </button>
      </div>
      <div className={styles.body}>{children}</div>
      {footer ? <div className={styles.footer}>{footer}</div> : null}
    </dialog>
  )
}
