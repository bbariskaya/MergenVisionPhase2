import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import StatusBadge from './StatusBadge.tsx'

describe('StatusBadge', () => {
  it('renders job status label', () => {
    render(<StatusBadge status="completed" />)
    expect(screen.getByText('Tamamlandı')).toBeInTheDocument()
  })

  it('renders identity status label', () => {
    render(<StatusBadge status="new_anonymous" />)
    expect(screen.getByText('Yeni Anonim')).toBeInTheDocument()
  })

  it('hides label when showLabel is false', () => {
    render(<StatusBadge status="failed" showLabel={false} />)
    expect(screen.getByLabelText('Durum: Başarısız')).toBeInTheDocument()
    expect(screen.getByText('Başarısız')).toHaveClass('sr-only')
  })
})
