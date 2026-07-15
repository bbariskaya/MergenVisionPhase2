import styles from './skeleton.module.css'

interface SkeletonProps {
  width?: string | number
  height?: string | number
  variant?: 'rect' | 'circle' | 'text'
  className?: string
  style?: React.CSSProperties
}

export default function Skeleton({
  width = '100%',
  height = '1em',
  variant = 'rect',
  className = '',
  style,
}: SkeletonProps) {
  const baseStyle: React.CSSProperties = {
    width: typeof width === 'number' ? `${width}px` : width,
    height: typeof height === 'number' ? `${height}px` : height,
    ...style,
  }
  const variantClass = variant === 'circle' ? styles.circle : variant === 'text' ? styles.text : ''
  return <div className={`${styles.skeleton} ${variantClass} ${className}`} style={baseStyle} aria-hidden="true" />
}
