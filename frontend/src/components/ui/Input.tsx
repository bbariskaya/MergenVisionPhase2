import { forwardRef, type InputHTMLAttributes } from 'react'
import styles from './input.module.css'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, className = '', ...rest }, ref) => {
    return (
      <div className={styles.field}>
        {label ? (
          <label className={styles.label} htmlFor={rest.id}>
            {label}
          </label>
        ) : null}
        <input
          ref={ref}
          className={`${styles.input} ${error ? styles.error : ''} ${className}`}
          aria-invalid={!!error}
          aria-describedby={error ? `${rest.id}-error` : undefined}
          {...rest}
        />
        {error ? (
          <p id={`${rest.id}-error`} className={styles.errorText} role="alert">
            {error}
          </p>
        ) : null}
      </div>
    )
  },
)

Input.displayName = 'Input'
export default Input
