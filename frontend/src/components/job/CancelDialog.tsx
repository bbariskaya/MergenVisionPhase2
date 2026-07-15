import { useState } from 'react'
import Dialog from '@/components/ui/Dialog.tsx'
import Button from '@/components/ui/Button.tsx'

interface CancelDialogProps {
  jobId: string
  onConfirm: () => void
  disabled?: boolean
}

export default function CancelDialog({ jobId, onConfirm, disabled }: CancelDialogProps) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <Button variant="danger" onClick={() => setOpen(true)} disabled={disabled}>
        İşlemi İptal Et
      </Button>
      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        title="İşlemi iptal etmek istiyor musunuz?"
        footer={
          <>
            <Button variant="secondary" onClick={() => setOpen(false)}>
              Geri
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                setOpen(false)
                onConfirm()
              }}
            >
              Evet, İptal Et
            </Button>
          </>
        }
      >
        <p>
          <strong>{jobId}</strong> numaralı işlem durdurulacak. GPU üzerindeki
          kaynaklar temizlenecek ve sonuç üretilmeyecek.
        </p>
      </Dialog>
    </>
  )
}
