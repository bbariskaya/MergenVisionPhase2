import '@testing-library/jest-dom/vitest'

if (typeof window !== 'undefined' && !window.HTMLDialogElement.prototype.showModal) {
  window.HTMLDialogElement.prototype.showModal = function () {
    this.setAttribute('open', '')
  }
  window.HTMLDialogElement.prototype.close = function () {
    this.removeAttribute('open')
  }
}
