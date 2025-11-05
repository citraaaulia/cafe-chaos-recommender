document.addEventListener('DOMContentLoaded', function () {
  const radios = document.querySelectorAll('.option input[type="radio"]');

  radios.forEach(radio => {
    if (radio.checked) {
      radio.closest('.option').classList.add('selected');
    }

    radio.addEventListener('change', () => {
      document.querySelectorAll('.option').forEach(label => label.classList.remove('selected'));
      radio.closest('.option').classList.add('selected');
    });
  });
});
