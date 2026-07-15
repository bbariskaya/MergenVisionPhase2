import { ApiError, NetworkError } from './errors.ts'
import type { ApiErrorBody } from './contracts.ts'
import { mockApiClient } from './mock/index.ts'

function getBaseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
}

function isMockEnabled(): boolean {
  return import.meta.env.VITE_USE_MOCK_API === 'true'
}

async function parseErrorBody(response: Response): Promise<ApiErrorBody | undefined> {
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    try {
      return (await response.json()) as ApiErrorBody
    } catch {
      return undefined
    }
  }
  return undefined
}

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE'
  body?: BodyInit | FormData | Record<string, unknown>
  headers?: Record<string, string>
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  if (isMockEnabled()) {
    return mockApiClient.request<T>(path, options)
  }

  const baseUrl = getBaseUrl()
  const url = `${baseUrl}${path}`
  const init: RequestInit = {
    method: options.method || 'GET',
    headers: {},
  }

  if (options.body instanceof FormData) {
    init.body = options.body
  } else if (options.body) {
    ;(init.headers as Record<string, string>)['Content-Type'] = 'application/json'
    init.body = JSON.stringify(options.body)
  }

  if (options.headers) {
    init.headers = { ...(init.headers as Record<string, string>), ...options.headers }
  }

  try {
    const response = await fetch(url, init)
    if (!response.ok) {
      const body = await parseErrorBody(response)
      throw new ApiError(
        body?.message || `HTTP ${response.status}`,
        response.status,
        body,
      )
    }

    if (response.status === 204) {
      return undefined as T
    }

    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      return (await response.json()) as T
    }

    return (await response.blob()) as unknown as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error instanceof TypeError) throw new NetworkError()
    throw error
  }
}

export const apiClient = {
  request,
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: RequestOptions['body']) =>
    request<T>(path, { method: 'POST', body }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
}
