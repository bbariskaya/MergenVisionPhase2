import { describe, it, expect } from 'vitest'
import { computeLetterbox, sourceToDisplay, displayToSource } from './bboxTransform.ts'

describe('computeLetterbox', () => {
  it('scales to fit width when container is narrower', () => {
    const lb = computeLetterbox(1920, 1080, 960, 540)
    expect(lb.scale).toBe(0.5)
    expect(lb.offsetX).toBe(0)
    expect(lb.offsetY).toBe(0)
  })

  it('centers content with pillarbox when container is wider', () => {
    const lb = computeLetterbox(1280, 720, 1600, 720)
    expect(lb.scale).toBe(1)
    expect(lb.offsetX).toBe(160)
    expect(lb.offsetY).toBe(0)
  })

  it('centers content with letterbox when container is taller', () => {
    const lb = computeLetterbox(1280, 720, 1280, 900)
    expect(lb.scale).toBe(1)
    expect(lb.offsetX).toBe(0)
    expect(lb.offsetY).toBe(90)
  })

  it('returns safe defaults for zero dimensions', () => {
    const lb = computeLetterbox(0, 0, 100, 100)
    expect(lb.scale).toBe(1)
    expect(lb.offsetX).toBe(0)
    expect(lb.offsetY).toBe(0)
  })
})

describe('sourceToDisplay', () => {
  it('maps original-resolution bbox to displayed canvas coordinates', () => {
    const lb = { scale: 0.5, offsetX: 100, offsetY: 20 }
    const box = { x: 200, y: 100, width: 160, height: 160 }
    expect(sourceToDisplay(box, lb)).toEqual({
      x: 200,
      y: 70,
      width: 80,
      height: 80,
    })
  })
})

describe('displayToSource', () => {
  it('inverts the letterbox transform', () => {
    const lb = { scale: 0.5, offsetX: 100, offsetY: 20 }
    expect(displayToSource(200, 70, lb)).toEqual({ x: 200, y: 100 })
  })
})
