import {
  completedFriendsJob,
  failedJob,
  cancelledJob,
  noFaceJob,
  faceAppearancesFixture,
  initialJobList,
  MOCK_DATA_TAG,
} from './fixtures.ts'
import {
  startJobSimulation,
  cancelJobSimulation,
  resetSimulations,
  registerJobUpdater,
} from './progressSimulator.ts'
import type {
  VideoJob,
  JobListItem,
} from '@/api/contracts.ts'
import type { RequestOptions } from '@/api/client.ts'
import { ApiError, ValidationError } from '@/api/errors.ts'

function deepClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

const store = {
  jobs: new Map<string, VideoJob>(),
  list: [] as JobListItem[],
}

function seedStore() {
  store.jobs.clear()
  store.list = deepClone(initialJobList)
  initialJobsList().forEach((job) => store.jobs.set(job.jobId, deepClone(job)))
}

function initialJobsList(): VideoJob[] {
  return [completedFriendsJob, failedJob, cancelledJob, noFaceJob]
}

function jobToListItem(job: VideoJob): JobListItem {
  return {
    jobId: job.jobId,
    status: job.status,
    videoName: job.videoName,
    createdAt: job.createdAt,
    durationSeconds: job.video?.duration,
    personCount: job.result?.personCount,
    progressPercent: job.progress?.percent,
  }
}

function updateJob(jobId: string, patch: Partial<VideoJob>) {
  const existing = store.jobs.get(jobId)
  if (!existing) return
  const updated = { ...existing, ...patch }
  if (patch.progress && existing.progress) {
    updated.progress = { ...existing.progress, ...patch.progress }
  }
  store.jobs.set(jobId, updated)
  const index = store.list.findIndex((item) => item.jobId === jobId)
  if (index >= 0) {
    store.list[index] = jobToListItem(updated)
  } else {
    store.list.unshift(jobToListItem(updated))
  }
}

registerJobUpdater(updateJob)
seedStore()

function extractPathParams(path: string, pattern: string): Record<string, string> | null {
  const regex = new RegExp(`^${pattern.replace(/:\w+/g, '([^/]+)')}$`)
  const match = path.match(regex)
  if (!match) return null
  const keys = [...pattern.matchAll(/:(\w+)/g)].map((m) => m[1])
  return Object.fromEntries(keys.map((key, idx) => [key, match[idx + 1]]))
}

function createJobId(): string {
  return `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}

function createProcessId(): string {
  return `proc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}

async function smallDelay(ms = 120): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

export const mockApiClient = {
  async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    await smallDelay()

    if (path === '/videos/recognize' && options.method === 'POST') {
      if (!(options.body instanceof FormData)) {
        throw new ValidationError('Video dosyası eksik')
      }
      const file = options.body.get('video')
      if (!(file instanceof File)) {
        throw new ValidationError('Video dosyası eksik')
      }
      const jobId = createJobId()
      const processId = createProcessId()
      const videoName = file.name

      const newJob: VideoJob = {
        jobId,
        processId,
        status: 'uploading',
        createdAt: new Date().toISOString(),
        videoName,
        progress: {
          percent: 5,
          stage: 'Yükleniyor',
          processedFrames: 0,
          decodedFrames: 0,
          detectedFaces: 0,
          currentTracklets: 0,
          elapsedSeconds: 0,
          workerId: 'worker-mock-01',
          gpuUuid: 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        },
      }
      store.jobs.set(jobId, newJob)
      store.list.unshift(jobToListItem(newJob))

      const completedJob = deepClone(completedFriendsJob)
      completedJob.jobId = jobId
      completedJob.processId = processId
      completedJob.videoName = videoName
      if (completedJob.result) {
        completedJob.result.jobId = jobId
        completedJob.result.processId = processId
      }

      startJobSimulation(jobId, {
        duration: completedFriendsJob.video?.duration,
        totalFrames: completedFriendsJob.video?.totalFrames,
        processedFrames: completedFriendsJob.video?.processedFrames,
        completedJob,
      })

      return { ...newJob, _mock: MOCK_DATA_TAG } as unknown as T
    }

    if (path === '/videos/jobs' && options.method !== 'POST') {
      return deepClone(store.list) as unknown as T
    }

    const jobParams = extractPathParams(path, '/videos/jobs/:jobId')
    if (jobParams && !path.includes('/result')) {
      const { jobId } = jobParams
      const job = store.jobs.get(jobId)
      if (!job) throw new ApiError('İşlem bulunamadı', 404, { code: 'JOB_NOT_FOUND', message: 'İşlem bulunamadı' })

      if (options.method === 'DELETE') {
        if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
          throw new ApiError('İşlem durdurulamaz', 409, { code: 'JOB_NOT_CANCELLABLE', message: 'Bu işlem iptal edilemez' })
        }
        cancelJobSimulation(jobId)
        return deepClone(store.jobs.get(jobId)) as unknown as T
      }

      return deepClone(job) as unknown as T
    }

    const resultParams = extractPathParams(path, '/videos/jobs/:jobId/result')
    if (resultParams) {
      const { jobId } = resultParams
      const job = store.jobs.get(jobId)
      if (!job) throw new ApiError('İşlem bulunamadı', 404, { code: 'JOB_NOT_FOUND', message: 'İşlem bulunamadı' })
      if (job.status !== 'completed') {
        throw new ApiError('Sonuç henüz hazır değil', 409, { code: 'RESULT_NOT_READY', message: 'Sonuç henüz hazır değil' })
      }
      return deepClone(job) as unknown as T
    }

    const faceParams = extractPathParams(path, '/faces/:faceId/appearances')
    if (faceParams) {
      const { faceId } = faceParams
      const data = faceAppearancesFixture[faceId]
      if (!data) throw new ApiError('Yüz bulunamadı', 404, { code: 'FACE_NOT_FOUND', message: 'Yüz bulunamadı' })
      return deepClone(data) as unknown as T
    }

    throw new ApiError('Bilinmeyen endpoint', 404, { code: 'NOT_FOUND', message: 'Bilinmeyen endpoint' })
  },

  reset() {
    resetSimulations()
    seedStore()
  },
}
