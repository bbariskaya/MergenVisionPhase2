import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ProgressStepper from './ProgressStepper.tsx'

describe('ProgressStepper', () => {
  it('marks upload as active when uploading', () => {
    render(<ProgressStepper status="uploading" progress={{ percent: 5, stage: 'Yükleniyor' }} />)
    const active = screen.getAllByRole('listitem').find((el) => el.className.includes('active'))
    expect(active).toHaveTextContent('Yükleme')
  })

  it('marks earlier steps completed when processing', () => {
    render(<ProgressStepper status="processing" progress={{ percent: 70, stage: 'Yüz tanıma' }} />)
    const items = screen.getAllByRole('listitem')
    const classes = items.map((el) => el.className)
    expect(classes[0]).toContain('completed')
    expect(classes[1]).toContain('completed')
    expect(classes[5]).toContain('active')
  })
})
