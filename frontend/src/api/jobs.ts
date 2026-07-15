import { apiClient } from './client.ts'
import type { VideoJob, JobListItem } from './contracts.ts'

export const jobsApi = {
  get: (jobId: string): Promise<VideoJob> => apiClient.get<VideoJob>(`/videos/jobs/${jobId}`),

  getResult: (jobId: string): Promise<VideoJob> =>
    apiClient.get<VideoJob>(`/videos/jobs/${jobId}/result`),

  cancel: (jobId: string): Promise<VideoJob> =>
    apiClient.delete<VideoJob>(`/videos/jobs/${jobId}`),

  list: (): Promise<JobListItem[]> => apiClient.get<JobListItem[]>('/videos/jobs'),
}
