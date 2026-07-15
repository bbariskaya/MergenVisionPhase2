import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import CancelDialog from './CancelDialog.tsx'

describe('CancelDialog', () => {
  it('opens dialog and calls onConfirm', async () => {
    const onConfirm = vi.fn()
    render(<CancelDialog jobId="job_123" onConfirm={onConfirm} />)

    await userEvent.click(screen.getByRole('button', { name: /İşlemi İptal Et/i }))
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /Evet, İptal Et/i }))
    expect(onConfirm).toHaveBeenCalledOnce()
  })

  it('closes dialog when back button is clicked', async () => {
    render(<CancelDialog jobId="job_123" onConfirm={vi.fn()} />)
    await userEvent.click(screen.getByRole('button', { name: /İşlemi İptal Et/i }))
    await userEvent.click(screen.getByRole('button', { name: /Geri/i }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
})
