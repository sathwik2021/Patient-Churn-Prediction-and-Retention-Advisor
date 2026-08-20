const API_BASE = '';
const TOKEN_KEY = 'patient_churn_token';
const COHORT_STORAGE_KEY = 'patient_churn_cohort_results';
const LEGACY_COHORT_STORAGE_KEY = COHORT_STORAGE_KEY;

const api = (path, options = {}) => fetch(`${API_BASE}${path}`, {
  ...options,
  headers: {
    ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
    ...(options.headers || {}),
    Authorization: `Bearer ${localStorage.getItem(TOKEN_KEY) || ''}`
  }
});

const page = document.body.dataset.page;
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
let cohortRows = [];
let cohortPage = 1;
let currentUserId = null;
const COHORT_PAGE_SIZE = 50;

function getCohortStorageKey() {
  return currentUserId ? `${COHORT_STORAGE_KEY}:${currentUserId}` : null;
}

function persistCohortResults(stats) {
  const storageKey = getCohortStorageKey();
  if (!storageKey) return;
  try {
    localStorage.setItem(storageKey, JSON.stringify({ rows: cohortRows, stats }));
  } catch (error) {
    console.warn('Cohort results could not be saved locally.', error);
  }
}

function restoreCohortResults() {
  const storageKey = getCohortStorageKey();
  if (!storageKey) return false;
  try {
    const stored = localStorage.getItem(storageKey);
    if (!stored) return false;
    const saved = JSON.parse(stored);
    if (!Array.isArray(saved.rows) || !saved.rows.length || !saved.stats) return false;
    cohortRows = saved.rows;
    $('#cohort-results')?.classList.remove('hidden');
    $('#cohort-total').textContent = saved.stats.total;
    $('#cohort-high').textContent = saved.stats.high;
    $('#cohort-medium').textContent = saved.stats.medium;
    $('#cohort-low').textContent = saved.stats.low;
    renderCohortVisualization(saved.stats);
    renderCohortRows();
    return true;
  } catch (error) {
    localStorage.removeItem(storageKey);
    return false;
  }
}

function renderCohortVisualization(stats) {
  const total = Number(stats.total) || 0;
  const levels = [
    ['high', Number(stats.high) || 0],
    ['medium', Number(stats.medium) || 0],
    ['low', Number(stats.low) || 0]
  ];
  levels.forEach(([level, count]) => {
    const percentage = total ? (count / total) * 100 : 0;
    const bar = $(`#cohort-chart-${level}`);
    const value = $(`#cohort-chart-${level}-value`);
    if (bar) bar.style.setProperty('--bar-height', `${percentage}%`);
    if (value) value.textContent = total ? `${Math.round(percentage)}%` : '—';
  });
}

function redirectToLogin() {
  window.location.href = '/frontend/login.html';
}

function redirectToDashboard() {
  window.location.href = '/frontend/dashboard.html';
}

async function requireAuth() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    redirectToLogin();
    return null;
  }
  try {
    const response = await api('/api/auth/me');
    if (!response.ok) throw new Error('Session expired');
    const data = await response.json();
    currentUserId = data.user.id;
    // Remove results saved by older versions under a shared browser key.
    localStorage.removeItem(LEGACY_COHORT_STORAGE_KEY);
    document.querySelectorAll('[data-user-name]').forEach((node) => { node.textContent = data.user.name; });
    document.querySelectorAll('[data-user-email]').forEach((node) => { node.textContent = data.user.email; });
    document.querySelectorAll('[data-user-initial]').forEach((node) => { node.textContent = data.user.name.charAt(0).toUpperCase(); });
    return data.user;
  } catch (error) {
    localStorage.removeItem(TOKEN_KEY);
    redirectToLogin();
    return null;
  }
}

async function signout() {
  try { await api('/api/auth/signout', { method: 'POST' }); } catch (error) { /* local session is still cleared */ }
  localStorage.removeItem(TOKEN_KEY);
  currentUserId = null;
  redirectToLogin();
}

function setLoginMode(mode) {
  const signup = mode === 'signup';
  $('#login-name-field')?.classList.toggle('hidden', !signup);
  $('#login-name')?.toggleAttribute('required', signup);
  $('#tab-signin')?.classList.toggle('active', !signup);
  $('#tab-signup')?.classList.toggle('active', signup);
  if ($('#login-subtitle')) $('#login-subtitle').textContent = signup ? 'Create a secure workspace for your care team' : 'Sign in to your patient retention workspace';
  if ($('#login-submit-btn')) $('#login-submit-btn').textContent = signup ? 'Create account' : 'Sign in to workspace';
}

async function handleLogin(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const mode = $('#tab-signup')?.classList.contains('active') ? 'signup' : 'signin';
  const button = $('#login-submit-btn');
  const error = $('#login-error');
  const formData = new FormData(form);
  const body = mode === 'signup'
    ? { name: formData.get('name'), email: formData.get('email'), password: formData.get('password') }
    : { email: formData.get('email'), password: formData.get('password') };
  button.disabled = true;
  button.textContent = 'Checking your workspace...';
  error?.classList.add('hidden');
  try {
    const response = await api(`/api/auth/${mode}`, { method: 'POST', body: JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Authentication failed');
    localStorage.setItem(TOKEN_KEY, data.token);
    redirectToDashboard();
  } catch (requestError) {
    if (error) { error.textContent = requestError.message; error.classList.remove('hidden'); }
  } finally {
    button.disabled = false;
    button.textContent = mode === 'signup' ? 'Create account' : 'Sign in to workspace';
  }
}

function formatDate(value) {
  if (!value) return 'Recently';
  const normalized = String(value).includes('T') ? String(value) : String(value).replace(' ', 'T');
  const utcValue = /(?:Z|[+-]\d{2}:?\d{2})$/.test(normalized) ? normalized : `${normalized}Z`;
  const date = new Date(utcValue);
  return Number.isNaN(date.getTime()) ? 'Recently' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

function riskClass(level) { return `risk-${String(level || 'low').toLowerCase()}`; }
function riskBadge(level) { return `<span class="risk-badge ${riskClass(level)}">${escapeHtml(level)} risk</span>`; }

function renderShell() {
  document.querySelectorAll('[data-nav]').forEach((link) => {
    link.classList.toggle('active', link.dataset.nav === page);
  });
  document.querySelectorAll('[data-signout]').forEach((button) => button.addEventListener('click', signout));
}

async function loadDashboard() {
  const response = await api('/api/user/analytics');
  if (!response.ok) return;
  const data = await response.json();
  const values = { evaluated: data.total_evaluated, average: `${data.avg_churn}%`, high: data.high_risk_count, medium: data.medium_risk_count, low: data.low_risk_count };
  Object.entries(values).forEach(([key, value]) => { const node = $(`[data-stat="${key}"]`); if (node) node.textContent = value; });
  const historyResponse = await api('/api/history');
  if (!historyResponse.ok) return;
  const history = (await historyResponse.json()).history.slice(0, 5);
  const list = $('#recent-history');
  if (!list) return;
  list.innerHTML = history.length ? history.map((item) => `<li><div><strong>${escapeHtml(item.primary_reason)}</strong><small>${formatDate(item.created_at)}</small></div><span>${(item.probability * 100).toFixed(1)}%</span></li>`).join('') : '<li class="empty-row">No predictions yet. Start your first assessment.</li>';
}

async function handleSinglePrediction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $('#predict-submit');
  const error = $('#predict-error');
  const result = $('#prediction-result');
  const payload = Object.fromEntries(new FormData(form).entries());
  ['age', 'tenure_months', 'referrals_made', 'visits_last_year', 'missed_appointments', 'days_since_last_visit', 'portal_usage', 'billing_issues'].forEach((field) => { payload[field] = Number.parseInt(payload[field], 10); });
  ['overall_satisfaction', 'wait_time_satisfaction', 'staff_satisfaction', 'provider_rating', 'avg_out_of_pocket_cost', 'distance_to_facility'].forEach((field) => { payload[field] = Number.parseFloat(payload[field]); });
  button.disabled = true;
  button.textContent = 'Generating assessment...';
  error?.classList.add('hidden');
  try {
    const response = await api('/api/predict', { method: 'POST', body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Assessment failed');
    result.classList.remove('hidden');
    result.innerHTML = `<div class="result-head"><div><span class="eyebrow">ASSESSMENT COMPLETE</span><h2>${data.percentage}% churn probability</h2></div>${riskBadge(data.risk_level)}</div><div class="result-reason"><span>Primary churn reason</span><strong>${escapeHtml(data.primary_churn_reason)}</strong></div><div class="result-advice"><span>Retention advice</span><p>${escapeHtml(data.retention_advice)}</p></div>`;
    result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (requestError) {
    if (error) { error.textContent = requestError.message; error.classList.remove('hidden'); }
  } finally { button.disabled = false; button.textContent = 'Generate risk assessment'; }
}

async function handleCohortUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const status = $('#cohort-status');
  const error = $('#cohort-error');
  status?.classList.remove('hidden'); error?.classList.add('hidden');
  const formData = new FormData(); formData.append('file', file);
  try {
    const response = await api('/api/batch-predict', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Upload failed');
    cohortRows = data.results.map((item, index) => ({ ...item, record_id: `REC-${String(index + 1).padStart(4, '0')}` }));
    cohortPage = 1;
    persistCohortResults({ total: data.total, high: data.high_risk, medium: data.medium_risk, low: data.low_risk });
    $('#cohort-results')?.classList.remove('hidden');
    $('#cohort-total').textContent = data.total; $('#cohort-high').textContent = data.high_risk; $('#cohort-medium').textContent = data.medium_risk; $('#cohort-low').textContent = data.low_risk;
    renderCohortVisualization({ total: data.total, high: data.high_risk, medium: data.medium_risk, low: data.low_risk });
    renderCohortRows();
  } catch (requestError) { if (error) { error.textContent = requestError.message; error.classList.remove('hidden'); } }
  finally { status?.classList.add('hidden'); event.target.value = ''; }
}

function renderCohortRows() {
  const query = ($('#record-filter')?.value || '').trim().toLowerCase();
  const rows = cohortRows.filter((item) => item.record_id.toLowerCase().includes(query) || String(item.patient_id || '').toLowerCase().includes(query));
  const body = $('#cohort-table-body');
  const totalPages = Math.ceil(rows.length / COHORT_PAGE_SIZE) || 1;
  cohortPage = Math.min(cohortPage, totalPages);
  const pageRows = rows.slice((cohortPage - 1) * COHORT_PAGE_SIZE, cohortPage * COHORT_PAGE_SIZE);
  if (!rows.length) body.innerHTML = '<tr><td colspan="7" class="empty-row">No records match this filter.</td></tr>';
  else body.innerHTML = pageRows.map((item) => `<tr><td><strong>${escapeHtml(item.record_id)}</strong></td><td>${escapeHtml(item.patient_id)}</td><td>${item.percentage}%</td><td>${riskBadge(item.risk_level)}</td><td>${escapeHtml(item.primary_churn_reason)}</td><td>${escapeHtml(item.retention_advice)}</td><td><button class="button button-secondary patient-view-button" type="button" data-patient-id="${escapeHtml(item.patient_id)}">View</button></td></tr>`).join('');
  body.querySelectorAll('[data-patient-id]').forEach((button) => button.addEventListener('click', () => openPatientFromCohort(button.dataset.patientId)));
  const pagination = $('#cohort-pagination');
  if (pagination) {
    pagination.classList.toggle('hidden', totalPages <= 1);
    $('#cohort-page-label').textContent = `Page ${cohortPage} of ${totalPages} · ${rows.length} records`;
    $('#cohort-prev').disabled = cohortPage === 1;
    $('#cohort-next').disabled = cohortPage === totalPages;
  }
}

function openPatientFromCohort(patientId) {
  const patient = cohortRows.find((item) => String(item.patient_id) === String(patientId));
  if (patient) localStorage.setItem('patient_churn_selected_patient', JSON.stringify(patient));
  window.location.href = 'predict.html';
}

function loadSelectedPatient() {
  const raw = localStorage.getItem('patient_churn_selected_patient');
  if (!raw) return;
  try {
    const patient = JSON.parse(raw);
    const attributes = patient.attributes || {};
    const norm = (value) => String(value ?? '').toLowerCase().replace(/[\s_-]/g, '');
    const fieldMap = { age: 'Age', gender: 'Gender', state: 'State', specialty: 'Specialty', insurance_type: 'Insurance_Type', tenure_months: 'Tenure_Months', referrals_made: 'Referrals_Made', visits_last_year: 'Visits_Last_Year', missed_appointments: 'Missed_Appointments', days_since_last_visit: 'Days_Since_Last_Visit', overall_satisfaction: 'Overall_Satisfaction', wait_time_satisfaction: 'Wait_Time_Satisfaction', staff_satisfaction: 'Staff_Satisfaction', provider_rating: 'Provider_Rating', avg_out_of_pocket_cost: 'Avg_Out_Of_Pocket_Cost', billing_issues: 'Billing_Issues', portal_usage: 'Portal_Usage', distance_to_facility: 'Distance_To_Facility_Miles' };
    Object.entries(fieldMap).forEach(([field, column]) => {
      const match = Object.keys(attributes).find((key) => norm(key) === norm(column));
      if (match !== undefined && attributes[match] !== undefined) {
        const input = document.querySelector(`[name="${field}"]`);
        if (input) input.value = attributes[match];
      }
    });
    localStorage.removeItem('patient_churn_selected_patient');
  } catch (error) { localStorage.removeItem('patient_churn_selected_patient'); }
}

function downloadCohortResults() {
  const rows = cohortRows.map((item) => `<tr><td>${escapeHtml(item.record_id)}</td><td>${escapeHtml(item.patient_id)}</td><td>${item.percentage}%</td><td>${escapeHtml(item.risk_level)}</td><td>${escapeHtml(item.primary_churn_reason)}</td><td>${escapeHtml(item.retention_advice)}</td></tr>`).join('');
  const printWindow = window.open('', '_blank', 'width=1200,height=800');
  if (!printWindow) return;
  printWindow.document.write(`<!doctype html><html><head><title>Patient Churn Cohort Results</title><style>body{font-family:Arial,sans-serif;color:#0f172a;padding:28px}h1{font-size:24px}p{color:#64748b}table{width:100%;border-collapse:collapse;font-size:11px}th,td{padding:10px;border:1px solid #cbd5e1;text-align:left;vertical-align:top}th{background:#0f172a;color:#fff}tr:nth-child(even){background:#f8fafc}@media print{body{padding:0}}</style></head><body><h1>Patient Churn Prediction and Retention Advisor</h1><p>Cohort risk assessment generated ${new Date().toLocaleString()}</p><table><thead><tr><th>Record ID</th><th>Patient ID</th><th>Probability</th><th>Risk</th><th>Primary reason</th><th>Retention advice</th></tr></thead><tbody>${rows}</tbody></table></body></html>`);
  printWindow.document.close();
  printWindow.focus();
  printWindow.print();
}

async function loadHistory() {
  const response = await api('/api/history');
  if (!response.ok) return;
  const records = (await response.json()).history;
  const body = $('#history-body');
  if (!records.length) { body.innerHTML = '<tr><td colspan="5" class="empty-row">No prediction history yet.</td></tr>'; return; }
  body.innerHTML = records.map((item) => `<tr><td>${formatDate(item.created_at)}</td><td>${(item.probability * 100).toFixed(1)}%</td><td>${riskBadge(item.risk_level)}</td><td>${escapeHtml(item.primary_reason)}</td><td>${escapeHtml(item.retention_advice)}</td></tr>`).join('');
}

async function loadAnalytics() {
  const response = await api('/api/user/analytics');
  if (!response.ok) return;
  const data = await response.json();
  const values = { evaluated: data.total_evaluated, average: `${data.avg_churn}%`, high: data.high_risk_count, medium: data.medium_risk_count, low: data.low_risk_count };
  Object.entries(values).forEach(([key, value]) => { const node = $(`[data-stat="${key}"]`); if (node) node.textContent = value; });
  const cohortBody = $('#cohort-history');
  if (!data.cohort_uploads.length) { cohortBody.innerHTML = '<div class="empty-state">No cohort uploads yet.</div>'; return; }
  cohortBody.innerHTML = data.cohort_uploads.map((item) => `<article class="cohort-history-card"><div><strong>${escapeHtml(item.filename)}</strong><small>${formatDate(item.created_at)}</small></div><span>${item.total_patients} patients</span><div class="cohort-risk-line"><b>${item.high_risk} high</b><b>${item.medium_risk} medium</b><b>${item.low_risk} low</b></div></article>`).join('');
}

document.addEventListener('DOMContentLoaded', async () => {
  if (page === 'login') {
    if (localStorage.getItem(TOKEN_KEY)) { redirectToDashboard(); return; }
    setLoginMode(new URLSearchParams(window.location.search).get('mode') === 'signup' ? 'signup' : 'signin');
    $('#login-form')?.addEventListener('submit', handleLogin);
    $('#tab-signin')?.addEventListener('click', () => setLoginMode('signin'));
    $('#tab-signup')?.addEventListener('click', () => setLoginMode('signup'));
    return;
  }
  if (document.body.dataset.protected === 'true') {
    const user = await requireAuth();
    if (!user) return;
    renderShell();
    if (page === 'dashboard') await loadDashboard();
    if (page === 'predict') { loadSelectedPatient(); $('#predict-form')?.addEventListener('submit', handleSinglePrediction); }
    if (page === 'cohort') { restoreCohortResults(); $('#cohort-file')?.addEventListener('change', handleCohortUpload); }
    if (page === 'cohort') { $('#record-filter')?.addEventListener('input', () => { cohortPage = 1; renderCohortRows(); }); $('#cohort-prev')?.addEventListener('click', () => { cohortPage--; renderCohortRows(); }); $('#cohort-next')?.addEventListener('click', () => { cohortPage++; renderCohortRows(); }); $('#download-cohort')?.addEventListener('click', downloadCohortResults); }
    if (page === 'history') await loadHistory();
    if (page === 'analytics') await loadAnalytics();
  }
});
