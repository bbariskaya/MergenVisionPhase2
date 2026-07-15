import { describe, it, expect } from 'vitest'
import { jobStatusLabel, identityStatusLabel, isTerminalStatus } from './statusMapping.ts'

describe('jobStatusLabel', () => {
  it('labels completed status', () => {
    expect(jobStatusLabel.completed).toBe('Tamamlandı')
  })

  it('labels processing status', () => {
    expect(jobStatusLabel.processing).toBe('İşleniyor')
  })
})

describe('identityStatusLabel', () => {
  it('labels known identities', () => {
    expect(identityStatusLabel.known).toBe('Tanındı')
  })

  it('labels new anonymous identities', () => {
    expect(identityStatusLabel.new_anonymous).toBe('Yeni Anonim')
  })
})

describe('isTerminalStatus', () => {
  it('returns true for completed', () => {
    expect(isTerminalStatus('completed')).toBe(true)
  })

  it('returns false for processing', () => {
    expect(isTerminalStatus('processing')).toBe(false)
  })
})
