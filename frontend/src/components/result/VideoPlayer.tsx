import {
  useEffect,
  useRef,
  useState,
  useCallback,
  forwardRef,
  useImperativeHandle,
} from 'react'
import {
  Play,
  Pause,
  Maximize,
  Minimize,
  Eye,
  EyeOff,
  Tag,
  UserX,
} from 'lucide-react'
import type { CanonicalPerson } from '@/api/contracts.ts'
import { formatDuration } from '@/lib/durationFormat.ts'
import BboxCanvas from './BboxCanvas.tsx'
import styles from './videoPlayer.module.css'

const SOURCE_VIDEO_URL = '/mock-videos/friendsshort.mp4'

export interface VideoPlayerHandle {
  seek: (time: number) => void
}

interface VideoPlayerProps {
  persons: CanonicalPerson[]
  selectedTrackId: string | null
  onSelectPerson: (trackId: string) => void
  onTimeUpdate?: (time: number) => void
}

const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(
  ({ persons, selectedTrackId, onSelectPerson, onTimeUpdate }, ref) => {
    const playerRef = useRef<HTMLDivElement>(null)
    const videoRef = useRef<HTMLVideoElement>(null)
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)
    const [isFullscreen, setIsFullscreen] = useState(false)
    const [showBboxes, setShowBboxes] = useState(true)
    const [showLabels, setShowLabels] = useState(true)
    const [showUnknown, setShowUnknown] = useState(true)

    useImperativeHandle(ref, () => ({
      seek: (time: number) => {
        const video = videoRef.current
        if (!video || !Number.isFinite(video.duration)) return
        video.currentTime = Math.max(0, Math.min(video.duration, time))
      },
    }))

    useEffect(() => {
      const video = videoRef.current
      if (!video) return
      const handleLoaded = () => setDuration(video.duration)
      const handleTime = () => {
        setCurrentTime(video.currentTime)
        onTimeUpdate?.(video.currentTime)
      }
      const handlePlay = () => setIsPlaying(true)
      const handlePause = () => setIsPlaying(false)
      video.addEventListener('loadedmetadata', handleLoaded)
      video.addEventListener('timeupdate', handleTime)
      video.addEventListener('play', handlePlay)
      video.addEventListener('pause', handlePause)
      if (video.readyState >= 1) handleLoaded()
    return () => {
      video.removeEventListener('loadedmetadata', handleLoaded)
      video.removeEventListener('timeupdate', handleTime)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
    }
  }, [onTimeUpdate])

  useEffect(() => {
      const handler = () => setIsFullscreen(!!document.fullscreenElement)
      document.addEventListener('fullscreenchange', handler)
      return () => document.removeEventListener('fullscreenchange', handler)
    }, [])

    const togglePlay = useCallback(() => {
      const video = videoRef.current
      if (!video) return
      if (video.paused) video.play()
      else video.pause()
    }, [])

    const seek = useCallback((fraction: number) => {
      const video = videoRef.current
      if (!video || !Number.isFinite(video.duration)) return
      video.currentTime = Math.max(0, Math.min(video.duration, video.duration * fraction))
    }, [])

    const toggleFullscreen = useCallback(() => {
      const player = playerRef.current
      if (!player) return
      if (!document.fullscreenElement) player.requestFullscreen()
      else document.exitFullscreen()
    }, [])

    const handleSeekClick = (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
      seek(fraction)
    }

    return (
      <div ref={playerRef} className={styles.player}>
        <video
          ref={videoRef}
          className={styles.video}
          src={SOURCE_VIDEO_URL}
          onClick={togglePlay}
          preload="metadata"
          aria-label="Analiz edilen video"
        />
        {showBboxes ? (
          <BboxCanvas
            videoRef={videoRef}
            persons={persons}
            selectedTrackId={selectedTrackId}
            showLabels={showLabels}
            showUnknown={showUnknown}
            onSelectPerson={onSelectPerson}
          />
        ) : null}
        <div className={styles.controls}>
          <div
            className={styles.seekBar}
            onClick={handleSeekClick}
            role="slider"
            aria-valuemin={0}
            aria-valuemax={duration || 0}
            aria-valuenow={currentTime}
            aria-label="Video konumu"
          >
            <div
              className={styles.seekFill}
              style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
            />
          </div>
          <div className={styles.controlRow}>
            <div className={styles.controlGroup}>
              <button
                type="button"
                className={styles.iconButton}
                onClick={togglePlay}
                aria-label={isPlaying ? 'Duraklat' : 'Oynat'}
              >
                {isPlaying ? <Pause size={18} /> : <Play size={18} />}
              </button>
              <span className={styles.time}>
                {formatDuration(currentTime)} / {formatDuration(duration)}
              </span>
              <select
                className={styles.select}
                aria-label="Oynatma hızı"
                onChange={(e) => {
                  if (videoRef.current) videoRef.current.playbackRate = Number(e.target.value)
                }}
                defaultValue={1}
              >
                <option value={0.5}>0.5x</option>
                <option value={1}>1x</option>
                <option value={1.5}>1.5x</option>
                <option value={2}>2x</option>
              </select>
            </div>
            <div className={styles.toggleGroup}>
              <button
                type="button"
                className={`${styles.toggle} ${showBboxes ? styles.activeToggle : ''}`}
                onClick={() => setShowBboxes((v) => !v)}
                aria-pressed={showBboxes}
              >
                {showBboxes ? <Eye size={12} /> : <EyeOff size={12} />}
                BBox
              </button>
              <button
                type="button"
                className={`${styles.toggle} ${showLabels ? styles.activeToggle : ''}`}
                onClick={() => setShowLabels((v) => !v)}
                aria-pressed={showLabels}
              >
                <Tag size={12} />
                Etiket
              </button>
              <button
                type="button"
                className={`${styles.toggle} ${showUnknown ? styles.activeToggle : ''}`}
                onClick={() => setShowUnknown((v) => !v)}
                aria-pressed={showUnknown}
              >
                <UserX size={12} />
                Bilinmeyen
              </button>
              <button
                type="button"
                className={styles.iconButton}
                onClick={toggleFullscreen}
                aria-label={isFullscreen ? 'Tam ekrandan çık' : 'Tam ekran'}
              >
                {isFullscreen ? <Minimize size={18} /> : <Maximize size={18} />}
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  },
)

VideoPlayer.displayName = 'VideoPlayer'
export default VideoPlayer
