import type { FrameDetection } from '@/api/contracts.ts'

export function findDetectionAtTime(
  detections: FrameDetection[],
  time: number,
): FrameDetection | null {
  if (!detections.length) return null

  let left = 0
  let right = detections.length - 1

  while (left < right) {
    const mid = Math.floor((left + right + 1) / 2)
    if (detections[mid]!.timestamp <= time) {
      left = mid
    } else {
      right = mid - 1
    }
  }

  const candidate = detections[left]!
  if (left + 1 < detections.length) {
    const next = detections[left + 1]!
    if (Math.abs(next.timestamp - time) < Math.abs(candidate.timestamp - time)) {
      return next
    }
  }
  return candidate
}

export function findAllDetectionsAroundTime(
  persons: { trackId: string; detections: FrameDetection[] }[],
  time: number,
  tolerance = 0.2,
): Array<{ trackId: string; detection: FrameDetection }> {
  return persons
    .map((p) => {
      const detection = findDetectionAtTime(p.detections, time)
      if (!detection) return null
      if (Math.abs(detection.timestamp - time) <= tolerance) {
        return { trackId: p.trackId, detection }
      }
      return null
    })
    .filter((item): item is NonNullable<typeof item> => item !== null)
}
