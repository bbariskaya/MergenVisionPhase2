import type { BoundingBox } from '@/api/contracts.ts'

export interface Letterbox {
  scale: number
  offsetX: number
  offsetY: number
}

export interface DisplayBoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export function computeLetterbox(
  sourceWidth: number,
  sourceHeight: number,
  containerWidth: number,
  containerHeight: number,
): Letterbox {
  if (sourceWidth <= 0 || sourceHeight <= 0 || containerWidth <= 0 || containerHeight <= 0) {
    return { scale: 1, offsetX: 0, offsetY: 0 }
  }
  const scale = Math.min(containerWidth / sourceWidth, containerHeight / sourceHeight)
  const offsetX = (containerWidth - sourceWidth * scale) / 2
  const offsetY = (containerHeight - sourceHeight * scale) / 2
  return { scale, offsetX, offsetY }
}

export function sourceToDisplay(
  box: BoundingBox,
  letterbox: Letterbox,
): DisplayBoundingBox {
  return {
    x: letterbox.offsetX + box.x * letterbox.scale,
    y: letterbox.offsetY + box.y * letterbox.scale,
    width: box.width * letterbox.scale,
    height: box.height * letterbox.scale,
  }
}

export function displayToSource(
  displayX: number,
  displayY: number,
  letterbox: Letterbox,
): { x: number; y: number } {
  return {
    x: (displayX - letterbox.offsetX) / letterbox.scale,
    y: (displayY - letterbox.offsetY) / letterbox.scale,
  }
}
