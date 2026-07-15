import { useState, useMemo } from 'react'
import { User } from 'lucide-react'
import type { CanonicalPerson, IdentityStatus } from '@/api/contracts.ts'
import StatusBadge from '@/components/ui/StatusBadge.tsx'
import { formatDuration } from '@/lib/durationFormat.ts'
import styles from './personList.module.css'

const filters: { key: IdentityStatus | 'all'; label: string }[] = [
  { key: 'all', label: 'Tümü' },
  { key: 'known', label: 'Tanındı' },
  { key: 'anonymous', label: 'Anonim' },
  { key: 'new_anonymous', label: 'Yeni Anonim' },
  { key: 'unknown', label: 'Bilinmiyor' },
]

interface PersonListProps {
  persons: CanonicalPerson[]
  selectedTrackId: string | null
  onSelect: (trackId: string) => void
}

export default function PersonList({ persons, selectedTrackId, onSelect }: PersonListProps) {
  const [filter, setFilter] = useState<IdentityStatus | 'all'>('all')

  const visible = useMemo(() => {
    if (filter === 'all') return persons
    return persons.filter((p) => p.status === filter)
  }, [persons, filter])

  return (
    <div className={styles.list} role="listbox" aria-label="Tespit edilen kişiler">
      <div className={styles.filters} role="group" aria-label="Kişi filtresi">
        {filters.map((f) => (
          <button
            key={f.key}
            type="button"
            className={`${styles.filterButton} ${filter === f.key ? styles.activeFilter : ''}`}
            onClick={() => setFilter(f.key)}
            aria-pressed={filter === f.key}
          >
            {f.label}
          </button>
        ))}
      </div>
      {visible.map((person) => {
        const confidenceLabel =
          person.confidence !== undefined
            ? `Eşleşme Güveni: ${Math.round(person.confidence * 100)}%`
            : person.similarity !== undefined
              ? `Benzerlik: ${person.similarity.toFixed(2)}`
              : null

        return (
          <div
            key={person.trackId}
            className={`${styles.card} ${selectedTrackId === person.trackId ? styles.selected : ''}`}
            onClick={() => onSelect(person.trackId)}
            role="option"
            aria-selected={selectedTrackId === person.trackId}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelect(person.trackId)
              }
            }}
          >
            <div className={styles.thumbnail} aria-hidden="true">
              <User size={20} />
            </div>
            <div className={styles.content}>
              <div className={styles.name}>{person.name ?? person.trackId}</div>
              <div className={styles.meta}>
                <StatusBadge status={person.status} showLabel />
                <span style={{ margin: '0 var(--space-1)' }}>·</span>
                <span>{person.appearances.length} görünüm</span>
                <span style={{ margin: '0 var(--space-1)' }}>·</span>
                <span>{formatDuration(person.totalDuration)}</span>
              </div>
              {confidenceLabel ? <div className={styles.confidence}>{confidenceLabel}</div> : null}
            </div>
          </div>
        )
      })}
    </div>
  )
}
