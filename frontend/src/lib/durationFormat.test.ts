import { describe, it, expect } from 'vitest'
import { formatDuration, formatDurationMs } from './durationFormat.ts'

describe('formatDuration', () => {
  it('formats seconds only', () => {
    expect(formatDuration(42)).toBe('00:42')
  })

  it('formats minutes and seconds', () => {
    expect(formatDuration(125)).toBe('02:05')
  })

  it('formats hours', () => {
    expect(formatDuration(3661)).toBe('1:01:01')
  })

  it('handles invalid input', () => {
    expect(formatDuration(NaN)).toBe('00:00')
  })
})

describe('formatDurationMs', () => {
  it('converts milliseconds to mm:ss', () => {
    expect(formatDurationMs(125000)).toBe('02:05')
  })
})
