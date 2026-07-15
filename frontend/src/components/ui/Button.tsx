import type { ButtonHTMLAttributes, ReactNode } from 'react'
import styles from './button.module.css'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
  size?: 'default' | 'small'
  fullWidth?: boolean
  children: ReactNode
}

export default function Button({
  variant = 'primary',
  size = 'default',
  fullWidth = false,
  className = '',
  children,
  ...rest
}: ButtonProps) {
  const classes = [
    styles.button,
    styles[variant],
    size === 'small' ? styles.small : '',
    fullWidth ? styles.fullWidth : '',
    className,
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button type="button" className={classes} {...rest}>
      {children}
    </button>
  )
}
