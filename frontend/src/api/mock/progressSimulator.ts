import type { VideoJob } from '@/api/contracts.ts'

type JobUpdater = (jobId: string, patch: Partial<VideoJob>) => void

let updateJob: JobUpdater | null = null
const timers = new Map<string, number[]>()

export const SIMULATION_STEP_MS = 450

export function registerJobUpdater(updater: JobUpdater) {
  updateJob = updater
}

function getUpdater(): JobUpdater {
  if (!updateJob) throw new Error('Job updater not registered')
  return updateJob
}

function clearJobTimers(jobId: string) {
  const ids = timers.get(jobId) ?? []
  ids.forEach((id) => window.clearTimeout(id))
  timers.delete(jobId)
}

function schedule(jobId: string, fn: () => void, delay: number) {
  const id = window.setTimeout(fn, delay)
  const ids = timers.get(jobId) ?? []
  ids.push(id)
  timers.set(jobId, ids)
}

const stages: Array<{ status: VideoJob['status']; stage: string; progress: number }> = [
  { status: 'uploading', stage: 'Yükleniyor', progress: 10 },
  { status: 'validating', stage: 'Dosya doğrulanıyor', progress: 25 },
  { status: 'queued', stage: 'GPU kuyruğunda', progress: 35 },
  { status: 'processing', stage: 'Yüz tespiti ve takip', progress: 55 },
  { status: 'processing', stage: 'Yüz tanıma', progress: 75 },
  { status: 'finalizing', stage: 'Kimlikler birleştiriliyor', progress: 88 },
  { status: 'rendering', stage: 'Sonuç videosu hazırlanıyor', progress: 97 },
]

interface SimulationOptions {
  duration?: number
  totalFrames?: number
  processedFrames?: number
  completedJob?: VideoJob
}

export function startJobSimulation(jobId: string, options: SimulationOptions = {}) {
  clearJobTimers(jobId)
  const updater = getUpdater()
  const duration = options.duration ?? 60
  const totalFrames = options.totalFrames ?? Math.round(duration * 25)
  const processedFrames = options.processedFrames ?? Math.round(totalFrames / 5)

  stages.forEach((step, index) => {
    schedule(
      jobId,
      () => {
        updater(jobId, {
          status: step.status,
          progress: {
            percent: step.progress,
            stage: step.stage,
            processedFrames: Math.round((processedFrames * step.progress) / 100),
            decodedFrames: Math.round((totalFrames * step.progress) / 100),
            detectedFaces: step.progress > 40 ? 5 : 0,
            currentTracklets: step.progress > 50 ? 5 : 0,
            elapsedSeconds: Math.round(((index + 1) * duration) / (stages.length + 1)),
            workerId: 'worker-mock-01',
            gpuUuid: 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
          },
        })
      },
      (index + 1) * SIMULATION_STEP_MS,
    )
  })

  if (options.completedJob) {
    schedule(
      jobId,
      () => {
        completeJobSimulation(jobId, options.completedJob!)
      },
      (stages.length + 1) * SIMULATION_STEP_MS,
    )
  }
}

export function completeJobSimulation(jobId: string, completedJob: VideoJob) {
  clearJobTimers(jobId)
  const updater = getUpdater()
  updater(jobId, {
    status: 'completed',
    progress: completedJob.progress,
    result: completedJob.result,
    video: completedJob.video,
  })
}

export function cancelJobSimulation(jobId: string) {
  clearJobTimers(jobId)
  const updater = getUpdater()
  updater(jobId, {
    status: 'cancelled',
    progress: { percent: 0, stage: 'İptal edildi' },
  })
}

export function resetSimulations() {
  timers.forEach((ids) => ids.forEach((id) => window.clearTimeout(id)))
  timers.clear()
}
