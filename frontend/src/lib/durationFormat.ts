export function formatDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return '00:00'
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = Math.floor(totalSeconds % 60)

  const paddedMinutes = String(minutes).padStart(2, '0')
  const paddedSeconds = String(seconds).padStart(2, '0')

  if (hours > 0) {
    return `${hours}:${paddedMinutes}:${paddedSeconds}`
  }
  return `${paddedMinutes}:${paddedSeconds}`
}

export function formatDurationMs(ms: number): string {
  return formatDuration(ms / 1000)
}
