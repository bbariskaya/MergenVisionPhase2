export function timeToPercent(time: number, duration: number): number {
  if (duration <= 0) return 0
  return Math.min(100, Math.max(0, (time / duration) * 100))
}

export function secondsToPixel(time: number, duration: number, containerWidth: number): number {
  if (duration <= 0 || containerWidth <= 0) return 0
  return (time / duration) * containerWidth
}
