// detailmenu.js — handle konfirmasi hapus & modal close

(function () {
  const openModal = (el) => { if (el) { el.style.display = 'flex'; el.setAttribute('aria-hidden', 'false'); } };
  const closeModal = (el) => { if (el) { el.style.display = 'none'; el.setAttribute('aria-hidden', 'true'); } };

  document.addEventListener('DOMContentLoaded', () => {
    const deleteBtn   = document.getElementById('btn-delete');
    const deleteForm  = document.getElementById('deleteForm');
    const modal       = document.getElementById('deleteConfirmModal');
    const ok          = document.getElementById('modal-confirm-delete');
    const cancel      = document.getElementById('modal-cancel-delete');

    if (deleteBtn) deleteBtn.addEventListener('click', () => openModal(modal));
    if (cancel)    cancel.addEventListener('click', () => closeModal(modal));
    if (ok)        ok.addEventListener('click', () => { if (deleteForm) deleteForm.submit(); });

    // klik di luar konten menutup modal
    if (modal) {
      modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(modal); });
    }
    // ESC menutup
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal(modal);
    });
  });
})();
