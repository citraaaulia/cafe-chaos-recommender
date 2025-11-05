document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('toggle-password-btn');
  const passwordField = document.getElementById('password');
  const iconImg = document.getElementById('toggle-icon');

  toggleBtn.addEventListener('click', () => {
    if (passwordField.type === 'password') {
      passwordField.type = 'text';
      iconImg.src = '/static/icons/hide.svg';
      iconImg.alt = 'Sembunyikan';
    } else {
      passwordField.type = 'password';
      iconImg.src = '/static/icons/show.svg';
      iconImg.alt = 'Tampilkan';
    }
  });
});
