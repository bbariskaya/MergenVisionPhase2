import { useState, useRef, useCallback, useMemo } from 'react'
import { ArrowLeft, Users } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import type { VideoJob } from '@/api/contracts.ts'
import Button from '@/components/ui/Button.tsx'
import { Card, CardBody, CardHeader } from '@/components/ui/Card.tsx'
import StatusBadge from '@/components/ui/StatusBadge.tsx'
import EmptyState from '@/components/ui/EmptyState.tsx'
import VideoPlayer, { type VideoPlayerHandle } from './VideoPlayer.tsx'
import PersonList from './PersonList.tsx'
import Timeline from './Timeline.tsx'
import TechnicalDetails from './TechnicalDetails.tsx'
import { formatDuration } from '@/lib/durationFormat.ts'
import styles from './resultPage.module.css'

interface ResultPageProps {
  job: VideoJob
}

export default function ResultPage({ job }: ResultPageProps) {
  const navigate = useNavigate()
  const videoRef = useRef<VideoPlayerHandle>(null)
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState(0)

  const persons = useMemo(() => job.result?.persons ?? [], [job.result?.persons])
  const duration = job.video?.duration ?? 0

  const handleSelectPerson = useCallback(
    (trackId: string) => {
      setSelectedTrackId(trackId)
      const person = persons.find((p) => p.trackId === trackId)
      if (person && videoRef.current) {
        const target = person.firstSeen
        videoRef.current.seek(target)
        setCurrentTime(target)
      }
    },
    [persons],
  )

  const handleTimelineSegmentClick = (trackId: string, time: number) => {
    setSelectedTrackId(trackId)
    if (videoRef.current) {
      videoRef.current.seek(time)
      setCurrentTime(time)
    }
  }

  if (job.status === 'completed' && persons.length === 0) {
    return (
      <div>
        <Button variant="ghost" size="small" onClick={() => navigate('/')} style={{ marginBottom: 'var(--space-4)' }}>
          <ArrowLeft size={16} aria-hidden="true" />
          Geri
        </Button>
        <Card>
          <CardBody>
            <div className={styles.noFaceState}>
              <EmptyState
                icon={<Users size={40} aria-hidden="true" />}
                title="Yüz bulunamadı"
                message="Video başarıyla işlendi ancak içerisinde tanımlanabilir bir yüz tespit edilmedi."
              />
            </div>
          </CardBody>
        </Card>
      </div>
    )
  }

  return (
    <div>
      <Button variant="ghost" size="small" onClick={() => navigate('/')} style={{ marginBottom: 'var(--space-4)' }}>
        <ArrowLeft size={16} aria-hidden="true" />
        Geri
      </Button>

      <header className={styles.header}>
        <h1 className={styles.title}>{job.videoName ?? 'Video Analizi Sonucu'}</h1>
        <div className={styles.subtitle}>
          <StatusBadge status={job.status} />
          <span>
            {job.result?.personCount ?? 0} kişi · {formatDuration(job.video?.duration ?? 0)} ·{' '}
            {job.video?.width}x{job.video?.height}
          </span>
          <Link to={`/faces`} style={{ fontWeight: 600 }}>
            Yüz geçmişine git
          </Link>
        </div>
      </header>

      <div className={styles.layout}>
        <div>
          <VideoPlayer
            ref={videoRef}
            persons={persons}
            selectedTrackId={selectedTrackId}
            onSelectPerson={handleSelectPerson}
            onTimeUpdate={setCurrentTime}
          />
        </div>
        <Card className={styles.personPanel}>
          <CardHeader title="Kişiler" subtitle={`${persons.length} canonical track`} />
          <CardBody>
            <PersonList persons={persons} selectedTrackId={selectedTrackId} onSelect={handleSelectPerson} />
          </CardBody>
        </Card>
      </div>

      <Card className={styles.timelineSection}>
        <CardHeader title="Görünüm Zaman Çizelgesi" />
        <CardBody>
          <Timeline
            duration={duration}
            persons={persons}
            selectedTrackId={selectedTrackId}
            currentTime={currentTime}
            onSegmentClick={handleTimelineSegmentClick}
          />
        </CardBody>
      </Card>

      <TechnicalDetails job={job} />
    </div>
  )
}
