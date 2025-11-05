// static/js/riwayat.js
document.addEventListener('DOMContentLoaded', function () {
  const modal = document.getElementById('detail-modal');
  const modalBody = document.getElementById('modal-body');
  const detailButtons = document.querySelectorAll('.btn-detail');
  const closeButton = document.querySelector('.modal-close-btn');

  const openModal = () => { modal.style.display = 'flex'; };
  const closeModal = () => { modal.style.display = 'none'; modalBody.innerHTML = '<p>Memuat data...</p>'; };

  detailButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const prefId = btn.dataset.prefId;
      openModal();

      fetch(`/admin/riwayat/detail/${prefId}`)
        .then(res => { if (!res.ok) throw new Error(res.status); return res.json(); })
        .then(result => {
          if (!result.success) {
            modalBody.innerHTML = `<p>Error: ${result.message}</p>`;
            return;
          }

          const p = result.data.preferences;
          const recs = result.data.recommendations || [];

          let recHtml = '<p>Tidak ada rekomendasi.</p>';
          if (recs.length) {
            recHtml = `
              <table class="rec-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Menu</th>
                    <th>Similarity</th>
                    <th>Final Score</th>
                  </tr>
                </thead>
                <tbody>
                  ${recs.map(r => `
                    <tr>
                      <td>${r.rank_position}</td>
                      <td>${r.nama_minuman}</td>
                      <td>${(Number(r.similarity_score) * 100).toFixed(2)}%</td>
                      <td>${(Number(r.final_score) * 100).toFixed(2)}%</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            `;
          }

          modalBody.innerHTML = `
            <div class="detail-section">
              <h4>Preferensi</h4>
              <div class="detail-grid">
                <div class="detail-item"><strong>Nama</strong><span>${p.nama_customer || '-'}</span></div>
                <div class="detail-item"><strong>Mood</strong><span>${p.mood || '-'}</span></div>
                <div class="detail-item"><strong>Rasa</strong><span>${p.rasa || '-'}</span></div>
                <div class="detail-item"><strong>Tekstur</strong><span>${p.tekstur || '-'}</span></div>
                <div class="detail-item"><strong>Kafein</strong><span>${p.kafein || '-'}</span></div>
                <div class="detail-item"><strong>Suhu</strong><span>${p.suhu || '-'}</span></div>
                <div class="detail-item"><strong>Budget</strong><span>${p.budget || '-'}</span></div>
              </div>
            </div>
            <div class="detail-section">
              <h4>Rekomendasi</h4>
              ${recHtml}
            </div>
          `;
        })
        .catch(() => {
          modalBody.innerHTML = `<p>Terjadi kesalahan saat memuat data.</p>`;
        });
    });
  });

  if (closeButton) closeButton.addEventListener('click', closeModal);
  if (modal) modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
});
