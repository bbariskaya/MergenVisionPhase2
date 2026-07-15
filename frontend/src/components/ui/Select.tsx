import { forwardRef, type SelectHTMLAttributes } from 'react'
import { ChevronDown } from 'lucide-react'
import styles from './input.module.css'

interface Option {
  value: string
  label: string
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  error?: string
  options: Option[]
}

const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, error, options, className = '', ...rest }, ref) => {
    return (
      <div className={styles.field}>
        {label ? (
          <label className={styles.label} htmlFor={rest.id}>
            {label}
          </label>
        ) : null}
        <div className={styles.selectWrapper}>
          <select
            ref={ref}
            className={`${styles.select} ${error ? styles.error : ''} ${className}`}
            aria-invalid={!!error}
            aria-describedby={error ? `${rest.id}-error` : undefined}
            {...rest}
          >
            {options.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <ChevronDown size={16} className={styles.chevron} aria-hidden="true" />
        </div>
        {error ? (
          <p id={`${rest.id}-error`} className={styles.errorText} role="alert">
            {error}
          </p>
        ) : null}
      </div>
    )
  },
)

Select.displayName = 'Select'
export default Select
