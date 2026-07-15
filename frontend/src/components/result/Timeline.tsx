import { timeToPercent } from '@/lib/timeline.ts'
import type { CanonicalPerson, IdentityStatus } from '@/api/contracts.ts'
import styles from './timeline.module.css'

interface TimelineProps {
  duration: number
  persons: CanonicalPerson[]
  selectedTrackId: string | null
  currentTime: number
  onSegmentClick: (trackId: string, time: number) => void
}

const statusClass: Record<IdentityStatus, string> = {
  known: styles.known,
  anonymous: styles.anonymous,
  new_anonymous: styles.new_anonymous,
  unknown: styles.unknown,
}

export default function Timeline({
  duration,
  persons,
  selectedTrackId,
  currentTime,
  onSegmentClick,
}: TimelineProps) {
  const playheadPercent = timeToPercent(currentTime, duration)

  return (
    <div className={styles.timeline} aria-label="Zaman çizelgesi">
      {persons.map((person) => (
        <div key={person.trackId} className={styles.row}>
          <div className={styles.label}>
            <span
              className={styles.labelText}
              style={{ opacity: selectedTrackId && selectedTrackId !== person.trackId ? 0.5 : 1 }}
            >
              {person.name ?? person.trackId}
            </span>
          </div>
          <div className={styles.track}>
            {person.appearances.map((app, idx) => {
              const left = timeToPercent(app.start, duration)
              const width = timeToPercent(app.end, duration) - left
              return (
                <button
                  key={idx}
                  type="button"
                  className={`${styles.segment} ${statusClass[person.status]}`}
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                    opacity: selectedTrackId && selectedTrackId !== person.trackId ? 0.35 : 1,
                  }}
                  onClick={() => onSegmentClick(person.trackId, app.start)}
                  aria-label={`${person.name ?? person.trackId}, ${app.start.toFixed(1)} - ${app.end.toFixed(1)} saniye`}
                  title={`${app.start.toFixed(2)}s - ${app.end.toFixed(2)}s`}
                />
              )
            })}
            <div className={styles.playhead} style={{ left: `${playheadPercent}%` }} aria-hidden="true" />
          </div>
        </div>
      ))}
    </div>
  )
}
