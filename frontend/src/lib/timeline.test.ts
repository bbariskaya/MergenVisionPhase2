import { describe, it, expect } from 'vitest'
import { timeToPercent } from './timeline.ts'

describe('timeToPercent', () => {
  it('maps midpoint to 50%', () => {
    expect(timeToPercent(21.74, 43.48)).toBeCloseTo(50, 1)
  })

  it('clamps negative time to 0%', () => {
    expect(timeToPercent(-5, 43.48)).toBe(0)
  })

  it('clamps time beyond duration to 100%', () => {
    expect(timeToPercent(100, 43.48)).toBe(100)
  })

  it('returns 0 when duration is zero', () => {
    expect(timeToPercent(5, 0)).toBe(0)
  })
})
