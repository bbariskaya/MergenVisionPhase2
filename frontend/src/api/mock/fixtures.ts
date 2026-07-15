import type {
  VideoJob,
  VideoResult,
  CanonicalPerson,
  FrameDetection,
  AppearanceInterval,
  JobListItem,
  FaceAppearancesResponse,
} from '@/api/contracts.ts'

export const MOCK_DATA_TAG = 'MOCK_DATA'

function detectionsForAppearance(
  interval: AppearanceInterval,
  baseBox: { x: number; y: number; width: number; height: number },
  count = 5,
): FrameDetection[] {
  const detections: FrameDetection[] = []
  const duration = interval.end - interval.start
  const frameDuration = interval.endFrame - interval.startFrame
  for (let i = 0; i < count; i++) {
    const t = interval.start + (duration * i) / (count - 1 || 1)
    const frame = interval.startFrame + Math.round((frameDuration * i) / (count - 1 || 1))
    const jitter = Math.sin(i * 1.3) * 4
    detections.push({
      frame,
      timestamp: Number(t.toFixed(3)),
      boundingBox: {
        x: Math.round(baseBox.x + jitter),
        y: Math.round(baseBox.y + jitter * 0.5),
        width: baseBox.width,
        height: baseBox.height,
      },
      confidence: 0.88 + (i % 3) * 0.03,
    })
  }
  return detections
}

function makePerson(
  faceId: string,
  trackId: string,
  status: CanonicalPerson['status'],
  name: string | null,
  appearances: AppearanceInterval[],
  baseBox: { x: number; y: number; width: number; height: number },
  confidence?: number,
  similarity?: number,
  margin?: number,
): CanonicalPerson {
  const detections = appearances.flatMap((a) => detectionsForAppearance(a, baseBox))
  return {
    faceId,
    trackId,
    status,
    name,
    metadata: status === 'known' ? { department: 'Cast' } : {},
    firstSeen: appearances[0]?.start ?? 0,
    lastSeen: appearances[appearances.length - 1]?.end ?? 0,
    totalDuration: Number(
      appearances.reduce((sum, a) => sum + (a.end - a.start), 0).toFixed(3),
    ),
    confidence,
    similarity,
    margin,
    appearances,
    detections,
    evidence: {
      bestMatchFaceId: faceId,
      similarity: similarity ?? 0.5,
      calibratedConfidence: confidence,
      runnerUpFaceId: 'face_other',
      runnerUpSimilarity: (similarity ?? 0.5) - (margin ?? 0.1),
      margin: margin ?? 0.1,
    },
  }
}

export const friendsResult: VideoResult = {
  jobId: 'job_friends_demo_001',
  processId: 'proc_friends_demo_001',
  status: 'completed',
  video: {
    duration: 43.48,
    fps: 25,
    width: 1280,
    height: 720,
    totalFrames: 1087,
    processedFrames: 217,
    samplingRate: 'every_5th_frame',
  },
  personCount: 5,
  persons: [
    makePerson(
      'face_phoebe_001',
      'video_person_0001',
      'known',
      'Phoebe Buffay',
      [
        { start: 1.2, end: 8.4, startFrame: 30, endFrame: 210 },
        { start: 22.6, end: 28.0, startFrame: 565, endFrame: 700 },
      ],
      { x: 420, y: 180, width: 160, height: 160 },
      0.97,
      0.91,
      0.24,
    ),
    makePerson(
      'face_rachel_002',
      'video_person_0002',
      'known',
      'Rachel Green',
      [
        { start: 3.6, end: 14.2, startFrame: 90, endFrame: 355 },
      ],
      { x: 740, y: 160, width: 150, height: 150 },
      0.96,
      0.89,
      0.21,
    ),
    makePerson(
      'face_chandler_003',
      'video_person_0003',
      'known',
      'Chandler Bing',
      [
        { start: 7.0, end: 18.5, startFrame: 175, endFrame: 462 },
      ],
      { x: 180, y: 220, width: 170, height: 170 },
      0.95,
      0.87,
      0.19,
    ),
    makePerson(
      'face_anon_prev_004',
      'video_person_0004',
      'anonymous',
      null,
      [
        { start: 16.0, end: 23.5, startFrame: 400, endFrame: 587 },
      ],
      { x: 950, y: 260, width: 140, height: 140 },
      0.72,
      0.68,
      0.08,
    ),
    makePerson(
      'face_unknown_005',
      'video_person_0005',
      'unknown',
      null,
      [
        { start: 30.0, end: 35.0, startFrame: 750, endFrame: 875 },
      ],
      { x: 520, y: 300, width: 130, height: 130 },
      undefined,
      0.36,
      undefined,
    ),
  ],
  audit: {
    modelVersions: {
      detector: 'RetinaFace-R50',
      recognizer: 'GlintR100',
      tracker: 'NvDCF',
    },
    workerId: 'worker-mock-01',
    gpuUuid: 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    rawTrackerMappings: {
      t004: { trackletId: 'tl_001', canonicalTrackId: 'video_person_0002', faceId: 'face_rachel_002' },
      t019: { trackletId: 'tl_007', canonicalTrackId: 'video_person_0002', faceId: 'face_rachel_002' },
      t031: { trackletId: 'tl_012', canonicalTrackId: 'video_person_0002', faceId: 'face_rachel_002' },
      t001: { trackletId: 'tl_002', canonicalTrackId: 'video_person_0001', faceId: 'face_phoebe_001' },
      t015: { trackletId: 'tl_008', canonicalTrackId: 'video_person_0001', faceId: 'face_phoebe_001' },
    },
  },
}

export const completedFriendsJob: VideoJob = {
  jobId: friendsResult.jobId,
  processId: friendsResult.processId,
  status: 'completed',
  createdAt: '2026-07-15T08:30:00.000Z',
  videoName: 'friendsshort.mp4',
  video: friendsResult.video,
  progress: {
    percent: 100,
    stage: 'completed',
    processedFrames: friendsResult.video.processedFrames,
    detectedFaces: 5,
    currentTracklets: 5,
    elapsedSeconds: 42,
    workerId: 'worker-mock-01',
    gpuUuid: 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
  },
  result: friendsResult,
}

export const noFaceResult: VideoResult = {
  jobId: 'job_noface_demo_002',
  processId: 'proc_noface_demo_002',
  status: 'completed',
  video: {
    duration: 12.0,
    fps: 25,
    width: 1920,
    height: 1080,
    totalFrames: 300,
    processedFrames: 60,
    samplingRate: 'every_5th_frame',
  },
  personCount: 0,
  persons: [],
}

export const noFaceJob: VideoJob = {
  jobId: noFaceResult.jobId,
  processId: noFaceResult.processId,
  status: 'completed',
  createdAt: '2026-07-15T09:10:00.000Z',
  videoName: 'empty_scene.mp4',
  video: noFaceResult.video,
  progress: { percent: 100, stage: 'completed', processedFrames: 60 },
  result: noFaceResult,
}

export const failedJob: VideoJob = {
  jobId: 'job_failed_demo_003',
  processId: 'proc_failed_demo_003',
  status: 'failed',
  createdAt: '2026-07-15T09:20:00.000Z',
  videoName: 'corrupt.mp4',
  progress: { percent: 12, stage: 'validating', processedFrames: 0 },
  error: {
    code: 'VIDEO_DECODE_ERROR',
    message: 'Video dosyası açılamadı; container veya codec desteklenmiyor.',
  },
}

export const cancelledJob: VideoJob = {
  jobId: 'job_cancelled_demo_004',
  processId: 'proc_cancelled_demo_004',
  status: 'cancelled',
  createdAt: '2026-07-15T09:25:00.000Z',
  videoName: 'long_video.mp4',
  progress: { percent: 34, stage: 'processing', processedFrames: 120 },
}

export const initialJobList: JobListItem[] = [
  {
    jobId: completedFriendsJob.jobId,
    status: 'completed',
    videoName: completedFriendsJob.videoName,
    createdAt: completedFriendsJob.createdAt,
    durationSeconds: completedFriendsJob.video?.duration,
    personCount: friendsResult.personCount,
    progressPercent: 100,
  },
  {
    jobId: noFaceJob.jobId,
    status: 'completed',
    videoName: noFaceJob.videoName,
    createdAt: noFaceJob.createdAt,
    durationSeconds: noFaceJob.video?.duration,
    personCount: 0,
    progressPercent: 100,
  },
  {
    jobId: failedJob.jobId,
    status: 'failed',
    videoName: failedJob.videoName,
    createdAt: failedJob.createdAt,
    progressPercent: 12,
  },
  {
    jobId: cancelledJob.jobId,
    status: 'cancelled',
    videoName: cancelledJob.videoName,
    createdAt: cancelledJob.createdAt,
    progressPercent: 34,
  },
]

export const initialJobs: Record<string, VideoJob> = {
  [completedFriendsJob.jobId]: completedFriendsJob,
  [noFaceJob.jobId]: noFaceJob,
  [failedJob.jobId]: failedJob,
  [cancelledJob.jobId]: cancelledJob,
}

export const faceAppearancesFixture: Record<string, FaceAppearancesResponse> = {
  'face_phoebe_001': {
    faceId: 'face_phoebe_001',
    name: 'Phoebe Buffay',
    status: 'known',
    totalVideos: 3,
    appearances: [
      { jobId: 'job_friends_demo_001', videoName: 'friendsshort.mp4', start: 1.2, end: 8.4, startFrame: 30, endFrame: 210, status: 'known' },
      { jobId: 'job_demo_010', videoName: 'friends_s02_e04_clip.mp4', start: 14.5, end: 22.0, startFrame: 348, endFrame: 528, status: 'known' },
      { jobId: 'job_demo_015', videoName: 'central_perk_b_roll.mp4', start: 3.0, end: 6.5, startFrame: 72, endFrame: 156, status: 'known' },
    ],
  },
  'face_rachel_002': {
    faceId: 'face_rachel_002',
    name: 'Rachel Green',
    status: 'known',
    totalVideos: 1,
    appearances: [
      { jobId: 'job_friends_demo_001', videoName: 'friendsshort.mp4', start: 3.6, end: 14.2, startFrame: 90, endFrame: 355, status: 'known' },
    ],
  },
  'face_unknown_005': {
    faceId: 'face_unknown_005',
    name: null,
    status: 'unknown',
    totalVideos: 1,
    appearances: [
      { jobId: 'job_friends_demo_001', videoName: 'friendsshort.mp4', start: 30.0, end: 35.0, startFrame: 750, endFrame: 875, status: 'unknown' },
    ],
  },
}
