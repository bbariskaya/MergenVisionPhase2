import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import PersonList from './PersonList.tsx'
import type { CanonicalPerson } from '@/api/contracts.ts'

const persons: CanonicalPerson[] = [
  {
    faceId: 'face_1',
    trackId: 'track_1',
    status: 'known',
    name: 'Phoebe',
    metadata: {},
    firstSeen: 1,
    lastSeen: 10,
    totalDuration: 9,
    confidence: 0.97,
    appearances: [{ start: 1, end: 10, startFrame: 25, endFrame: 250 }],
    detections: [],
  },
  {
    faceId: 'face_2',
    trackId: 'track_2',
    status: 'unknown',
    name: null,
    metadata: {},
    firstSeen: 5,
    lastSeen: 8,
    totalDuration: 3,
    similarity: 0.36,
    appearances: [{ start: 5, end: 8, startFrame: 125, endFrame: 200 }],
    detections: [],
  },
]

describe('PersonList', () => {
  it('renders all persons by default', () => {
    render(<PersonList persons={persons} selectedTrackId={null} onSelect={vi.fn()} />)
    expect(screen.getByText('Phoebe')).toBeInTheDocument()
    expect(screen.getByText('track_2')).toBeInTheDocument()
  })

  it('filters by identity status', async () => {
    render(<PersonList persons={persons} selectedTrackId={null} onSelect={vi.fn()} />)
    await userEvent.click(screen.getByRole('button', { name: /Bilinmiyor/i }))
    expect(screen.queryByText('Phoebe')).not.toBeInTheDocument()
    expect(screen.getByText('track_2')).toBeInTheDocument()
  })

  it('calls onSelect when a card is clicked', async () => {
    const onSelect = vi.fn()
    render(<PersonList persons={persons} selectedTrackId={null} onSelect={onSelect} />)
    await userEvent.click(screen.getByText('Phoebe'))
    expect(onSelect).toHaveBeenCalledWith('track_1')
  })

  it('displays calibrated confidence for known person', () => {
    render(<PersonList persons={persons} selectedTrackId={null} onSelect={vi.fn()} />)
    expect(screen.getByText(/Eşleşme Güveni: 97%/i)).toBeInTheDocument()
  })

  it('displays raw similarity for unknown person', () => {
    render(<PersonList persons={persons} selectedTrackId={null} onSelect={vi.fn()} />)
    expect(screen.getByText(/Benzerlik: 0.36/i)).toBeInTheDocument()
  })
})
