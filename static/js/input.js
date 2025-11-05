// static/js/back.js
document.addEventListener('DOMContentLoaded', function () {
  const backBtn = document.getElementById('back-btn');
  if (!backBtn) return;

  // Jika halaman punya modal discard (contoh: menu_add), biarkan script halamannya yang handle
  if (document.getElementById('discardChangesModal')) return;

  backBtn.addEventListener('click', function (e) {
    e.preventDefault();

    // Balik ke halaman sebelumnya kalau ada referrer dalam origin yang sama
    try {
      const hasRef = document.referrer && new URL(document.referrer).origin === location.origin;
      if (hasRef) {
        history.back();
        return;
      }
    } catch (_) {}

    // Fallback ke URL yang dikasih lewat data-fallback
    const fallback = backBtn.getAttribute('data-fallback');
    if (fallback) {
      location.href = fallback;
    } else {
      // fallback terakhir: root
      location.href = '/';
    }
  });
});
