import type { ApiErrorBody } from './contracts.ts'

export class ApiError extends Error {
  status: number
  body?: ApiErrorBody

  constructor(message: string, status: number, body?: ApiErrorBody) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

export class NetworkError extends Error {
  constructor(message = 'Ağ bağlantısı kurulamadı') {
    super(message)
    this.name = 'NetworkError'
  }
}

export class ValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ValidationError'
  }
}

export function normalizeError(error: unknown): { message: string; code?: string } {
  if (error instanceof ApiError) {
    return {
      message: error.body?.message ?? error.message,
      code: error.body?.code ?? `HTTP_${error.status}`,
    }
  }
  if (error instanceof NetworkError) {
    return { message: error.message, code: 'NETWORK_ERROR' }
  }
  if (error instanceof ValidationError) {
    return { message: error.message, code: 'VALIDATION_ERROR' }
  }
  if (error instanceof Error) {
    return { message: error.message, code: 'UNKNOWN_ERROR' }
  }
  return { message: 'Bilinmeyen bir hata oluştu', code: 'UNKNOWN_ERROR' }
}
