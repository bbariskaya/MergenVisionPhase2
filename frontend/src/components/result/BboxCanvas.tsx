import { useEffect, useRef, useCallback } from 'react'
import type { CanonicalPerson } from '@/api/contracts.ts'
import { computeLetterbox, sourceToDisplay } from '@/lib/bboxTransform.ts'
import { findDetectionAtTime } from '@/lib/detectionLookup.ts'
import styles from './bboxCanvas.module.css'

interface BboxCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement>
  persons: CanonicalPerson[]
  selectedTrackId: string | null
  showLabels: boolean
  showUnknown: boolean
  onSelectPerson: (trackId: string) => void
}

const statusColor: Record<CanonicalPerson['status'], string> = {
  known: '#1e5aa8',
  anonymous: '#6b7280',
  new_anonymous: '#047857',
  unknown: '#9ca3af',
}

export default function BboxCanvas({
  videoRef,
  persons,
  selectedTrackId,
  showLabels,
  showUnknown,
  onSelectPerson,
}: BboxCanvasProps) {
  const wrapperRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const dimensionsRef = useRef({ width: 0, height: 0 })

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const wrapper = wrapperRef.current
    const video = videoRef.current
    if (!canvas || !wrapper || !video) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { width, height } = wrapper.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
    }
    ctx.resetTransform()
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)

    dimensionsRef.current = { width, height }

    const sourceWidth = video.videoWidth
    const sourceHeight = video.videoHeight
    if (!sourceWidth || !sourceHeight || !Number.isFinite(video.currentTime)) return

    const letterbox = computeLetterbox(sourceWidth, sourceHeight, width, height)

    persons.forEach((person) => {
      if (!showUnknown && person.status === 'unknown') return
      const isSelected = selectedTrackId === person.trackId
      const isDimmed = selectedTrackId && !isSelected

      const detection = findDetectionAtTime(person.detections, video.currentTime)
      if (!detection) return
      if (Math.abs(detection.timestamp - video.currentTime) > 0.5) return

      const box = sourceToDisplay(detection.boundingBox, letterbox)
      const color = statusColor[person.status]
      const lineWidth = isSelected ? 3 : 2
      const alpha = isDimmed ? 0.35 : 1

      ctx.globalAlpha = alpha
      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.strokeRect(box.x, box.y, box.width, box.height)

      if (showLabels) {
        const label = person.name ?? person.trackId
        const shortLabel = label.length > 24 ? `${label.slice(0, 24)}…` : label
        const padding = 4
        ctx.font = '12px ui-sans-serif, system-ui, sans-serif'
        const textMetrics = ctx.measureText(shortLabel)
        const textWidth = textMetrics.width
        const textHeight = 14
        const labelY = box.y > textHeight + padding * 2 ? box.y - textHeight - padding * 2 : box.y + box.height + padding

        ctx.fillStyle = color
        ctx.fillRect(box.x, labelY, textWidth + padding * 2, textHeight + padding * 2)
        ctx.fillStyle = '#ffffff'
        ctx.fillText(shortLabel, box.x + padding, labelY + textHeight)
      }

      ctx.globalAlpha = 1
    })
  }, [persons, selectedTrackId, showLabels, showUnknown, videoRef])

  useEffect(() => {
    const video = videoRef.current
    const wrapper = wrapperRef.current
    if (!video || !wrapper) return

    draw()
    const handleTime = () => draw()
    video.addEventListener('timeupdate', handleTime)
    video.addEventListener('seeked', handleTime)
    video.addEventListener('loadedmetadata', handleTime)
    video.addEventListener('play', handleTime)
    video.addEventListener('pause', handleTime)

    const animation = requestAnimationFrame(function loop() {
      draw()
      requestAnimationFrame(loop)
    })

    const observer = new ResizeObserver(() => draw())
    observer.observe(wrapper)

    return () => {
      video.removeEventListener('timeupdate', handleTime)
      video.removeEventListener('seeked', handleTime)
      video.removeEventListener('loadedmetadata', handleTime)
      video.removeEventListener('play', handleTime)
      video.removeEventListener('pause', handleTime)
      cancelAnimationFrame(animation)
      observer.disconnect()
    }
  }, [draw, videoRef])

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const wrapper = wrapperRef.current
    const video = videoRef.current
    if (!wrapper || !video) return
    const rect = wrapper.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const letterbox = computeLetterbox(video.videoWidth, video.videoHeight, rect.width, rect.height)

    for (let i = persons.length - 1; i >= 0; i--) {
      const person = persons[i]!
      const detection = findDetectionAtTime(person.detections, video.currentTime)
      if (!detection) continue
      const box = sourceToDisplay(detection.boundingBox, letterbox)
      if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) {
        onSelectPerson(person.trackId)
        return
      }
    }
  }

  return (
    <div ref={wrapperRef} className={styles.canvasWrapper}>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        onClick={handleClick}
        role="img"
        aria-label="Video üzerindeki yüz kutuları"
      />
    </div>
  )
}
