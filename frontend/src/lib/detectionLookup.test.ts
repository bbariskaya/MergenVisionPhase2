import { describe, it, expect } from 'vitest'
import { findDetectionAtTime, findAllDetectionsAroundTime } from './detectionLookup.ts'

describe('findDetectionAtTime', () => {
  const detections = [
    { frame: 10, timestamp: 0.4, boundingBox: { x: 0, y: 0, width: 10, height: 10 }, confidence: 0.9 },
    { frame: 20, timestamp: 0.8, boundingBox: { x: 1, y: 1, width: 10, height: 10 }, confidence: 0.9 },
    { frame: 30, timestamp: 1.2, boundingBox: { x: 2, y: 2, width: 10, height: 10 }, confidence: 0.9 },
    { frame: 40, timestamp: 1.6, boundingBox: { x: 3, y: 3, width: 10, height: 10 }, confidence: 0.9 },
  ]

  it('returns exact match', () => {
    expect(findDetectionAtTime(detections, 1.2)?.frame).toBe(30)
  })

  it('returns nearest detection before time when no exact match', () => {
    expect(findDetectionAtTime(detections, 1.0)?.frame).toBe(20)
  })

  it('returns nearest detection when time is closer to next sample', () => {
    expect(findDetectionAtTime(detections, 1.45)?.frame).toBe(40)
  })

  it('returns last detection when time exceeds range', () => {
    expect(findDetectionAtTime(detections, 10)?.frame).toBe(40)
  })

  it('returns null for empty list', () => {
    expect(findDetectionAtTime([], 1)).toBeNull()
  })
})

describe('findAllDetectionsAroundTime', () => {
  const persons = [
    {
      trackId: 't1',
      detections: [
        { frame: 10, timestamp: 0.4, boundingBox: { x: 0, y: 0, width: 10, height: 10 } },
      ],
    },
    { trackId: 't2', detections: [{ frame: 100, timestamp: 4.0, boundingBox: { x: 0, y: 0, width: 10, height: 10 } }] },
  ]

  it('includes only detections within tolerance', () => {
    const result = findAllDetectionsAroundTime(persons, 0.5, 0.2)
    expect(result).toHaveLength(1)
    expect(result[0]?.trackId).toBe('t1')
  })
})
