import { apiClient } from './client.ts'
import type { FaceAppearancesResponse } from './contracts.ts'

export const facesApi = {
  appearances: (faceId: string): Promise<FaceAppearancesResponse> =>
    apiClient.get<FaceAppearancesResponse>(`/faces/${faceId}/appearances`),
}
