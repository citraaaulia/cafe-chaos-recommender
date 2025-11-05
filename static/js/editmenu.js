// =========================
// editmenu.js — selaras dgn addmenu.js (error + dropdown arrow)
// =========================

// --- GLOBAL ---
let isFormDirty = false;
const form = document.getElementById('editMenuForm');

// default cancel url; bisa di-override dengan data-fallback di back button
let cancelURL = '/admin/menu';
const backBtnEl = document.getElementById('back-btn');
if (backBtnEl && backBtnEl.getAttribute('data-fallback')) {
  cancelURL = backBtnEl.getAttribute('data-fallback');
}

// label untuk template pesan error (samakan dengan addmenu.js)
const LABELS = {
  nama_minuman: 'Nama',
  harga: 'Harga',
  kafein_score: 'Kafein Score',
  rasa_manis: 'Rasa Manis',
  rasa_pahit: 'Rasa Pahit',
  rasa_gurih: 'Rasa Gurih',
  rasa_asam: 'Rasa Asam',
  temperatur_opsi: 'Temperatur',
  tekstur: 'Tekstur',
  carbonated_score: 'Carbonated',
  sweetness_level: 'Sweetness Level'
};

// definisi grup checkbox/radio
const GROUP_DEFS = [
  { name: 'categories',       errorId: 'error-kategori',     label: 'Kategori',       required: true  },
  { name: 'main_ingredients', errorId: 'error-bahan_utama',  label: 'Bahan Utama',    required: true  },
  { name: 'toppings',         errorId: 'error-topping',      label: 'Topping',        required: false },
  { name: 'aromas',           errorId: 'error-aroma',        label: 'Aroma',          required: true  },
  { name: 'flavour_notes',    errorId: 'error-flavor_notes', label: 'Flavour Notes',  required: true  },
  { name: 'availability',     errorId: 'error-availability', label: 'Availability',   required: true  }
];

// ---------- helpers error ----------
function getOrCreateErrorEl(id, afterEl) {
  let el = document.getElementById(id);
  if (!el && afterEl) {
    el = document.createElement('div');
    el.id = id;
    el.className = 'error-message';
    afterEl.parentNode.insertBefore(el, afterEl.nextSibling);
  }
  return el;
}

function setError(id, msg, afterEl) {
  const el = getOrCreateErrorEl(id, afterEl);
  if (!el) return;
  el.textContent = msg || '';
  if (msg) el.classList.add('show'); else el.classList.remove('show');
}

// ---------- validasi field "single" (input/select biasa) ----------
function validateSingles() {
  let ok = true;

  const SINGLE_FIELDS = [
    'nama_minuman','harga',
    'kafein_score','rasa_manis','rasa_pahit','rasa_gurih','rasa_asam',
    'temperatur_opsi','tekstur','carbonated_score','sweetness_level'
  ];

  SINGLE_FIELDS.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;

    let msg = '';
    if (el.required && el.validity.valueMissing) {
      msg = `Data ${LABELS[id]} wajib diisi.`;
    } else if ((id === 'kafein_score' || id === 'rasa_manis' || id === 'rasa_pahit' || id === 'rasa_gurih' || id === 'rasa_asam')
      && (el.validity.rangeUnderflow || el.validity.rangeOverflow)) {
      msg = `${LABELS[id]} harus antara ${el.min}–${el.max}.`;
    } else if (el.validity.stepMismatch) {
      msg = `${LABELS[id]} gunakan kelipatan ${el.step}.`;
    }

    setError(`error-${id}`, msg, el);
    if (msg) ok = false;
  });

  return ok;
}

// ---------- validasi grup (checkbox/radio & dropdown-custom) ----------
function validateGroups() {
  let ok = true;

  GROUP_DEFS.forEach(g => {
    const inputs = form ? form.querySelectorAll(`input[name="${g.name}"]`) : [];
    const anyChecked = Array.from(inputs).some(i => i.checked);

    // cari container yang pas untuk menaruh error
    const container =
      (inputs[0] && (inputs[0].closest('.dropdown-check-list') ||
                     inputs[0].closest('.ketersediaan') ||
                     inputs[0].closest('.form-group') ||
                     inputs[0].closest('.form-grid') ||
                     inputs[0].parentElement)) || null;

    const msg = (g.required && !anyChecked) ? `Data ${g.label} wajib diisi.` : '';
    setError(g.errorId, msg, container);
    if (msg) ok = false;
  });

  return ok;
}

// ---------- submit flow ----------
function validateAndSubmit() {
  const btn = document.getElementById('btn-simpan');
  if (btn) btn.disabled = true;

  const singlesOK = validateSingles();
  const groupsOK  = validateGroups();

  if (!form.checkValidity() || !singlesOK || !groupsOK) {
    form.reportValidity();
    if (btn) btn.disabled = false;
    return;
  }
  showModalEl(document.getElementById('confirmationModal'));
}

function confirmSave() {
  isFormDirty = false;
  if (form) form.submit();
}

function handleCancelAction(e) {
  e.preventDefault();
  if (e && e.stopImmediatePropagation) e.stopImmediatePropagation();
  checkDirty();
  if (isFormDirty) showModalEl(document.getElementById('discardChangesModal'));
  else window.location.href = cancelURL;
}

// ---------- MODAL helpers ----------
function showModalEl(el){ if (el) { el.style.display = 'flex'; el.setAttribute('aria-hidden','false'); } }
function hideModalEl(el){ if (el) { el.style.display = 'none'; el.setAttribute('aria-hidden','true'); } }

window.closeModal        = () => hideModalEl(document.getElementById('confirmationModal'));
window.closeDiscardModal = () => hideModalEl(document.getElementById('discardChangesModal'));
window.confirmSave       = confirmSave;
window.confirmDiscard    = () => { isFormDirty = false; window.location.href = cancelURL; };

function wireModalDismiss() {
  document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', e => { if (e.target === modal) hideModalEl(modal); });
    const content = modal.querySelector('.modal-content');
    if (content) content.addEventListener('click', e => e.stopPropagation());
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal').forEach(m => {
        if (getComputedStyle(m).display !== 'none') hideModalEl(m);
      });
    }
  });
}

// ====== Snapshot & Detect Dirty (tetap) ======
function snapshotForm(frm) {
  if (!frm) return '';
  const data = {};
  frm.querySelectorAll('input, select, textarea').forEach(el => {
    if (!el.name) return;
    if (el.type === 'checkbox') {
      if (!data[el.name]) data[el.name] = [];
      if (el.checked) data[el.name].push(el.value);
    } else if (el.type === 'radio') {
      if (el.checked) data[el.name] = el.value;
      else if (data[el.name] === undefined) data[el.name] = null;
    } else if (el.type === 'file') {
      data[el.name] = (el.files && el.files.length) ? 'HAS_FILE' : '';
    } else {
      data[el.name] = el.value;
    }
  });
  Object.keys(data).forEach(k => { if (Array.isArray(data[k])) data[k].sort(); });
  return JSON.stringify(data);
}

let __initialSnapshot = null;
function checkDirty() {
  if (!form) { isFormDirty = false; return false; }
  const now = snapshotForm(form);
  isFormDirty = (__initialSnapshot !== null && now !== __initialSnapshot);
  return isFormDirty;
}

// ---------- ENHANCED DROPDOWN (ikon panah & perilaku disamakan dgn addmenu.js) ----------
class EnhancedDropdown {
  constructor(container) {
    if (container.__enhancedDropdown) return;
    container.__enhancedDropdown = true;

    this.container = container;
    this.anchor = container.querySelector('.anchor');
    this.itemsList = container.querySelector('.items');
    this.inputs = container.querySelectorAll('input[type="checkbox"], input[type="radio"]');
    this.listItems = container.querySelectorAll('.items li');

    const onlyRadio = Array.from(this.inputs).every(i => i.type === 'radio');
    this.isSingle = this.container.classList.contains('single') || onlyRadio;

    this.isOpen = false;
    this.selectedItems = new Map();
    this.originalAnchorText = this.anchor.textContent.trim();

    this.setupAnchorStructure();
    this.bindEvents();
    this.loadInitialSelection();        // tidak menandai dirty
    this.updateDisplay();
  }

  setupAnchorStructure() {
    const originalText = this.anchor.textContent.trim();
    this.anchor.innerHTML = '';

    const selectedContainer = document.createElement('div');
    selectedContainer.className = 'selected-tags-container';

    const placeholderText = document.createElement('span');
    placeholderText.className = 'placeholder-text';
    placeholderText.textContent = originalText;

    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.className = 'search-input';
    searchInput.placeholder = 'Cari...';
    searchInput.style.display = 'none';

    const dropdownIcon = document.createElement('div');
    dropdownIcon.className = 'dropdown-icon';
    dropdownIcon.innerHTML = `
      <svg class="arrow-down" width="12" height="8" viewBox="0 0 12 8" fill="none">
        <path d="M1 1L6 6L11 1" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <svg class="arrow-up" width="12" height="8" viewBox="0 0 12 8" fill="none" style="display:none;">
        <path d="M1 7L6 2L11 7" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    `;

    selectedContainer.appendChild(placeholderText);
    selectedContainer.appendChild(searchInput);
    this.anchor.appendChild(selectedContainer);
    this.anchor.appendChild(dropdownIcon);

    this.selectedContainer = selectedContainer;
    this.placeholderText = placeholderText;
    this.searchInput = searchInput;
    this.arrowDown = dropdownIcon.querySelector('.arrow-down');
    this.arrowUp   = dropdownIcon.querySelector('.arrow-up');
  }

  bindEvents() {
    this.anchor.addEventListener('click', (e) => {
      if (!this.searchInput.contains(e.target) && !e.target.classList.contains('remove-tag')) {
        this.toggleDropdown();
      }
    });

    this.searchInput.addEventListener('input', (e) => this.filterItems(e.target.value));
    this.searchInput.addEventListener('focus', () => { if (!this.isOpen) this.openDropdown(); });

    this.inputs.forEach(input => {
      input.addEventListener('change', (e) => this.handleInputChange(e.target));
    });

    document.addEventListener('click', (e) => { if (!this.container.contains(e.target)) this.closeDropdown(); });
    this.itemsList.addEventListener('click', (e) => e.stopPropagation());

    this.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' || e.key === 'Enter') { e.preventDefault(); this.closeDropdown(); }
    });
  }

  loadInitialSelection() {
    const checked = this.container.querySelectorAll('input[type="checkbox"]:checked, input[type="radio"]:checked');
    checked.forEach(inp => this.handleInputChange(inp, true)); // true = initial
  }

  toggleDropdown() { this.isOpen ? this.closeDropdown() : this.openDropdown(); }

  openDropdown() {
    this.isOpen = true;
    this.container.classList.add('visible', 'active');
    this.itemsList.classList.add('visible');
    this.searchInput.style.display = 'block';
    if (this.selectedItems.size === 0) this.placeholderText.style.display = 'none';
    this.arrowDown.style.display = 'none';
    this.arrowUp.style.display = 'block';
    setTimeout(() => this.searchInput.focus(), 10);
    this.filterItems('');
  }

  closeDropdown() {
    this.isOpen = false;
    this.container.classList.remove('visible', 'active');
    this.itemsList.classList.remove('visible');
    this.searchInput.value = '';
    this.filterItems('');
    this.arrowDown.style.display = 'block';
    this.arrowUp.style.display = 'none';
    this.searchInput.style.display = 'none';
    this.updateDisplay();
  }

  handleInputChange(input, isInitial = false) {
    const value = input.value;
    const li = input.closest('li');
    const label = li && li.querySelector('label') ? li.querySelector('label').textContent : '';

    if (input.type === 'radio' || this.isSingle) {
      Array.from(this.inputs).filter(i => i.name === input.name).forEach(i => this.selectedItems.delete(i.value));
      if (input.checked) this.selectedItems.set(value, { value, label, input });
      if (!isInitial) setTimeout(() => this.closeDropdown(), 120);
    } else {
      if (input.checked) this.selectedItems.set(value, { value, label, input });
      else this.selectedItems.delete(value);
    }

    this.listItems.forEach(node => node.classList.remove('selected'));
    if (li && input.checked) li.classList.add('selected');

    this.updateDisplay();
    if (!isInitial) { checkDirty(); validateGroups(); } // bersihkan error saat pilih
  }

  updateDisplay() {
    Array.from(this.selectedContainer.querySelectorAll('.selected-tag')).forEach(tag => tag.remove());
    const showSearch = this.isOpen;

    if (this.selectedItems.size === 0) {
      if (!this.placeholderText.parentNode) {
        this.selectedContainer.insertBefore(this.placeholderText, this.selectedContainer.firstChild);
      }
      this.placeholderText.style.display = showSearch ? 'none' : 'block';
      this.placeholderText.textContent = this.originalAnchorText;
      this.searchInput.style.display = showSearch ? 'block' : 'none';
      return;
    }

    if (this.isSingle) {
      const first = this.selectedItems.values().next().value;
      if (!this.placeholderText.parentNode) {
        this.selectedContainer.insertBefore(this.placeholderText, this.selectedContainer.firstChild);
      }
      this.placeholderText.style.display = 'block';
      this.placeholderText.textContent = first.label;
      this.searchInput.style.display = showSearch ? 'block' : 'none';
      return;
    }

    if (this.placeholderText.parentNode) {
      this.placeholderText.parentNode.removeChild(this.placeholderText);
    }
    this.searchInput.style.display = showSearch ? 'block' : 'none';

    this.selectedItems.forEach(item => {
      const tag = document.createElement('span');
      tag.className = 'selected-tag';
      tag.innerHTML = `${item.label} <button type="button" class="remove-tag" data-value="${item.value}">&times;</button>`;
      tag.querySelector('.remove-tag').addEventListener('click', (e) => {
        e.stopPropagation();
        this.removeItem(item.value);
      });
      this.selectedContainer.insertBefore(tag, this.searchInput);
    });
  }

  removeItem(value) {
    const item = this.selectedItems.get(value);
    if (item?.input) item.input.checked = false;
    this.selectedItems.delete(value);
    this.updateDisplay();
    checkDirty();
    validateGroups();
  }

  filterItems(term) {
    const q = (term || '').toLowerCase().trim();
    let visible = 0;
    this.listItems.forEach(li => {
      if (li.querySelector('a')) return;
      const lab = li.querySelector('label'); if (!lab) return;
      const ok = lab.textContent.toLowerCase().includes(q);
      li.classList.toggle('hidden', !ok);
      if (ok) visible++;
    });
    this.updateEmptyState(visible, q);
  }

  updateEmptyState(count, q) {
    let empty = this.itemsList.querySelector('.empty-state');
    if (count === 0 && q) {
      if (!empty) {
        empty = document.createElement('li');
        empty.className = 'empty-state';
        empty.textContent = 'Tidak ada item yang ditemukan';
        this.itemsList.appendChild(empty);
      }
      empty.classList.add('show');
    } else if (empty) {
      empty.classList.remove('show');
    }
  }
}

// ---------- INIT ----------
document.addEventListener('DOMContentLoaded', () => {
  // Inisialisasi dropdown
  document.querySelectorAll('.dropdown-check-list').forEach(container => {
    if (!container.__enhancedDropdown) new EnhancedDropdown(container);
  });

  // Tombol utama
  const btnSimpan   = document.getElementById('btn-simpan');
  const btnCancel   = document.getElementById('btn-cancel');
  const backBtn     = document.getElementById('back-btn');
  const btnConfirm  = document.getElementById('modal-confirm');
  const btnCancelM1 = document.getElementById('modal-cancel');
  const btnCancelM2 = document.getElementById('modal-cancel2');

  if (btnSimpan)   btnSimpan.addEventListener('click', validateAndSubmit);
  if (btnConfirm)  btnConfirm.addEventListener('click', confirmSave);
  if (btnCancel)   btnCancel.addEventListener('click', handleCancelAction);
  if (backBtn)     backBtn.addEventListener('click', handleCancelAction);

  // tombol "Tidak"
  btnCancelM1?.addEventListener('click', () => hideModalEl(document.getElementById('confirmationModal')));
  btnCancelM2?.addEventListener('click', () => hideModalEl(document.getElementById('discardChangesModal')));

  wireModalDismiss();

  // revalidate agar pesan hilang ketika user mengisi
  if (form) {
    form.querySelectorAll('input, select, textarea').forEach(el => {
      el.addEventListener('input',  () => { checkDirty(); validateSingles(); validateGroups(); });
      el.addEventListener('change', () => { checkDirty(); validateSingles(); validateGroups(); });
    });
  }

  // snapshot awal
  __initialSnapshot = snapshotForm(form);
  isFormDirty = false;
});
