export type VideoJobStatus =
  | 'pending'
  | 'uploading'
  | 'validating'
  | 'queued'
  | 'processing'
  | 'finalizing'
  | 'rendering'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type IdentityStatus = 'known' | 'anonymous' | 'new_anonymous' | 'unknown'

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface FrameDetection {
  frame: number
  timestamp: number
  boundingBox: BoundingBox
  confidence?: number
}

export interface AppearanceInterval {
  start: number
  end: number
  startFrame: number
  endFrame: number
}

export interface RecognitionEvidence {
  bestMatchFaceId: string
  similarity: number
  calibratedConfidence?: number
  runnerUpFaceId?: string
  runnerUpSimilarity?: number
  margin?: number
}

export interface CanonicalPerson {
  faceId: string
  trackId: string
  status: IdentityStatus
  name: string | null
  metadata: Record<string, unknown>
  firstSeen: number
  lastSeen: number
  totalDuration: number
  confidence?: number
  similarity?: number
  margin?: number
  appearances: AppearanceInterval[]
  detections: FrameDetection[]
  evidence?: RecognitionEvidence
}

export interface VideoMetadata {
  duration: number
  fps: number
  width: number
  height: number
  totalFrames: number
  processedFrames: number
  samplingRate: string
}

export interface JobProgress {
  percent: number
  stage: string
  decodedFrames?: number
  processedFrames?: number
  detectedFaces?: number
  currentTracklets?: number
  elapsedSeconds?: number
  workerId?: string
  gpuUuid?: string
  currentTimestamp?: number
}

export interface ProcessStage {
  key: string
  label: string
  description?: string
}

export interface VideoResult {
  jobId: string
  processId: string
  status: Extract<VideoJobStatus, 'completed'>
  video: VideoMetadata
  personCount: number
  persons: CanonicalPerson[]
  audit?: Record<string, unknown>
}

export interface ApiErrorBody {
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface VideoJob {
  jobId: string
  processId: string
  status: VideoJobStatus
  createdAt: string
  videoName?: string
  video?: VideoMetadata
  progress?: JobProgress
  result?: VideoResult
  error?: ApiErrorBody
}

export interface CreateJobRequest {
  samplingRate?: string
  minFaceSize?: number
  profile?: 'accuracy' | 'balanced' | 'fast'
}

export interface JobListItem {
  jobId: string
  status: VideoJobStatus
  videoName?: string
  createdAt: string
  durationSeconds?: number
  personCount?: number
  progressPercent?: number
}

export interface FaceAppearance {
  jobId: string
  videoName?: string
  start: number
  end: number
  startFrame: number
  endFrame: number
  status: IdentityStatus
}

export interface FaceAppearancesResponse {
  faceId: string
  name: string | null
  status: IdentityStatus
  totalVideos: number
  appearances: FaceAppearance[]
}

export interface ApiHealth {
  status: 'ok' | 'degraded' | 'unavailable'
  version?: string
  dependencies?: Record<string, 'ok' | 'degraded' | 'unavailable'>
}
