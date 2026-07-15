import { apiClient } from './client.ts'
import type { VideoJob, CreateJobRequest } from './contracts.ts'

export const videosApi = {
  recognize: async (file: File, params: CreateJobRequest = {}): Promise<VideoJob> => {
    const formData = new FormData()
    formData.append('video', file)
    if (params.samplingRate) formData.append('samplingRate', params.samplingRate)
    if (params.minFaceSize) formData.append('minFaceSize', String(params.minFaceSize))
    if (params.profile) formData.append('profile', params.profile)

    return apiClient.post<VideoJob>('/videos/recognize', formData)
  },
}
