import { test, expect } from '@playwright/test'

test.describe('MergenVision internal demo', { tag: '@demo' }, () => {
  test('dashboard shows seeded completed Friends job', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Video Analiz Merkezi' })).toBeVisible()
    await expect(page.getByText('friendsshort.mp4').first()).toBeVisible()
    await page.screenshot({ path: 'artifacts/screenshots/01-dashboard.png' })
  })

  test('completed job result page renders video, persons and timeline', async ({ page }) => {
    await page.goto('/videos/jobs/job_friends_demo_001')
    await expect(page.getByRole('heading', { name: 'friendsshort.mp4' })).toBeVisible()
    await expect(page.locator('video')).toBeVisible()
    await expect(page.getByRole('option', { name: /Phoebe Buffay/ })).toBeVisible()
    await expect(page.getByRole('option', { name: /Rachel Green/ })).toBeVisible()
    await expect(page.getByRole('option', { name: /Chandler Bing/ })).toBeVisible()

    await page.getByRole('option', { name: /Rachel Green/ }).click()
    await expect(page.getByRole('option', { name: /Rachel Green/ })).toHaveAttribute('aria-selected', 'true')

    await page.screenshot({ path: 'artifacts/screenshots/02-result-page.png' })
  })

  test('face appearances page shows identity history', async ({ page }) => {
    await page.goto('/faces/face_phoebe_001')
    await expect(page.getByRole('heading', { name: 'Phoebe Buffay' })).toBeVisible()
    await expect(page.getByText(/3 videoda görünüyor/)).toBeVisible()
    await page.screenshot({ path: 'artifacts/screenshots/03-face-page.png' })
  })

  test('upload video and wait for mock completion', async ({ page }) => {
    await page.goto('/videos/new')
    await expect(page.getByRole('heading', { name: 'Yeni Video Analizi' })).toBeVisible()

    const fileChooserPromise = page.waitForEvent('filechooser')
    await page.getByRole('button', { name: /Video dosyası seçmek için/i }).click()
    const fileChooser = await fileChooserPromise
    await fileChooser.setFiles('../test_videos/friendsshort.mp4')

    await expect(page.getByText(/friendsshort\.mp4/)).toBeVisible()
    await page.getByRole('button', { name: 'Analizi Başlat' }).click()

    await page.waitForURL(/\/videos\/jobs\/job_/)
    await expect(page.getByRole('heading', { name: /friendsshort\.mp4/ })).toBeVisible()
    await expect(page.getByRole('option', { name: /Phoebe Buffay/ })).toBeVisible({ timeout: 30_000 })

    await page.screenshot({ path: 'artifacts/screenshots/04-upload-complete.png' })
  })
})
