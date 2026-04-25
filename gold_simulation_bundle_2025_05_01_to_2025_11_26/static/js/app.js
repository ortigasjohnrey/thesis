let runId = null;
let latestChart = { dates: [], actual: [], predicted: [], absolute_error: [] };
let revealedRows = [];
let cumulativeMetrics = { rmse: null, r2: null, rows: 0 };
let chartData = { dates: [], actual: [], predicted: [], pad: null };
const REVEAL_DELAY_MS = 700;

const $ = (id) => document.getElementById(id);

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function money(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  return Number(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fixed(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  return Number(x).toFixed(d);
}

function setAlert(id, msg, kind = 'neutral') {
  const el = $(id);
  el.className = `alert ${kind}`;
  el.textContent = msg;
}

function setButtons({ started = false, finished = false } = {}) {
  $('nextBtn').disabled = !started || finished;
  $('resetBtn').disabled = !started;
  $('downloadBtn').disabled = revealedRows.length === 0;
}

async function apiJson(url, options = {}) {
  const res = await fetch(url, options);
  let data;
  try { data = await res.json(); } catch { data = { ok: false, error: await res.text() }; }
  if (!res.ok || data.ok === false) throw new Error(data.error || `Request failed: ${res.status}`);
  return data;
}

async function checkStatus() {
  try {
    const data = await apiJson('/api/status');
    if (data.new_data_exists) {
      setAlert('message', 'Ready: data files found. Enter a date and click Load Simulation.', 'good');
    } else {
      setAlert('message', 'Missing data files. Please configure your CSV file path.', 'bad');
    }
  } catch (err) {
    setAlert('message', err.message, 'bad');
  }
}

async function startSimulation() {
  $('startBtn').disabled = true;
  setAlert('message', 'Loading model and data...', 'neutral');
  try {
    const payload = {
      start_date: $('startDate').value,
      mode: 'anchor'
    };
    const data = await apiJson('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    runId = data.run_id;
    latestChart = data.chart;
    revealedRows = [];
    cumulativeMetrics = data.metrics;
    $('logBody').innerHTML = '';
    updateMetrics(null, data.metrics, data.remaining_rows);
    drawChart(latestChart);
    setButtons({ started: true, finished: false });
    let msg = `Loaded ${data.total_rows} forecast rows. First forecast: ${data.first_forecast_date}.`;
    if (data.start_note) msg += ` Note: ${data.start_note}`;
    setAlert('message', msg, data.start_note ? 'warn' : 'good');
  } catch (err) {
    runId = null;
    setButtons({ started: false });
    setAlert('message', err.message, 'bad');
  } finally {
    $('startBtn').disabled = false;
  }
}

async function nextDayPredict() {
  if (!runId) return;
  $('nextBtn').disabled = true;
  setAlert('message', 'Computing and preparing reveal...', 'neutral');
  try {
    const data = await apiJson('/api/next', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: runId })
    });
    if (data.current_row) {
      await wait(REVEAL_DELAY_MS);
    }
    if (data.current_row) {
      revealedRows.push(data.current_row);
      cumulativeMetrics = data.metrics;
      prependLog(data.current_row, data.metrics);
      updateMetrics(data.current_row, data.metrics, data.remaining_rows);
      latestChart = data.chart;
      drawChart(latestChart);
      setAlert('message', data.finished ? 'Simulation finished. You reached the final forecast row.' : 'Forecast revealed. Metrics updated.', data.finished ? 'neutral' : 'good');
    } else {
      setAlert('message', data.message || 'Simulation finished.', 'neutral');
    }
    setButtons({ started: true, finished: data.finished });
  } catch (err) {
    setAlert('message', err.message, 'bad');
    setButtons({ started: true, finished: false });
  }
}

async function resetRun() {
  if (!runId) return;
  try {
    const data = await apiJson('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: runId })
    });
    latestChart = data.chart;
    revealedRows = [];
    cumulativeMetrics = data.metrics;
    $('logBody').innerHTML = '';
    updateMetrics(null, data.metrics, data.remaining_rows);
    drawChart(latestChart);
    setButtons({ started: true, finished: false });
    setAlert('message', 'Reset. Click Next Day Predict to begin.', 'neutral');
  } catch (err) {
    setAlert('message', err.message, 'bad');
  }
}

function updateMetrics(row, metrics = {}, remaining = null) {
  $('mAnchor').textContent = row ? row.anchor_date : '—';
  $('mDate').textContent = row ? row.forecast_date : '—';
  $('mPred').textContent = row ? money(row.predicted_price) : '—';
  $('mActual').textContent = row ? money(row.actual_price) : '—';
  $('mErr').textContent = row ? money(row.absolute_error) : '—';
  $('mRmse').textContent = metrics && metrics.rmse !== null ? money(metrics.rmse) : '—';
  $('mR2').textContent = metrics && metrics.r2 !== null ? fixed(metrics.r2, 4) : (metrics && metrics.rows === 1 ? 'N/A' : '—');
}

function prependLog(row, metrics = {}) {
  const tr = document.createElement('tr');
  const r2Val = metrics && metrics.r2 !== null ? fixed(metrics.r2, 4) : (metrics && metrics.rows === 1 ? 'N/A' : '—');
  const rmseVal = metrics && metrics.rmse !== null ? money(metrics.rmse) : '—';
  tr.innerHTML = `
    <td>${row.anchor_date}</td>
    <td>${row.forecast_date}</td>
    <td>${money(row.predicted_price)}</td>
    <td>${money(row.actual_price)}</td>
    <td>${r2Val}</td>
    <td>${rmseVal}</td>
  `;
  $('logBody').prepend(tr);
}

function downloadCsv() {
  if (revealedRows.length === 0) return;
  const cols = ['anchor_date', 'forecast_date', 'predicted_price', 'actual_price', 'absolute_error', 'squared_error'];
  const escape = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
  const csv = [cols.join(',')].concat(revealedRows.map(row => cols.map(c => escape(row[c])).join(','))).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'gold_simulation_log.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function handleChartHover(e) {
  const canvas = $('chart');
  const tooltip = $('chartTooltip');
  
  if (!chartData.dates || !chartData.dates.length) {
    tooltip.style.display = 'none';
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  const w = canvas.width;
  const h = canvas.height;
  const pad = chartData.pad;
  
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  
  // Check if mouse is within the plot area
  if (mouseX < pad.left || mouseX > w - pad.right || mouseY < pad.top || mouseY > h - pad.bottom) {
    tooltip.style.display = 'none';
    return;
  }
  
  // Find closest data point
  const dates = chartData.dates;
  const actual = chartData.actual;
  const predicted = chartData.predicted;
  
  const allVals = actual.concat(predicted).filter(v => Number.isFinite(v));
  let minY = Math.min(...allVals);
  let maxY = Math.max(...allVals);
  const span = maxY - minY || 1;
  minY -= span * 0.1;
  maxY += span * 0.1;
  
  // Calculate x position corresponding to date index
  let closestIdx = 0;
  let closestDist = Infinity;
  
  for (let i = 0; i < dates.length; i++) {
    const xx = dates.length === 1 ? pad.left + plotW / 2 : pad.left + i * plotW / (dates.length - 1);
    const dist = Math.abs(mouseX - xx);
    if (dist < closestDist) {
      closestDist = dist;
      closestIdx = i;
    }
  }
  
  // Only show tooltip if within reasonable distance
  const xx = dates.length === 1 ? pad.left + plotW / 2 : pad.left + closestIdx * plotW / (dates.length - 1);
  if (closestDist > 30) {
    tooltip.style.display = 'none';
    return;
  }
  
  const actualVal = actual[closestIdx];
  const predictedVal = predicted[closestIdx];
  const date = dates[closestIdx];
  
  tooltip.innerHTML = `
    <div><strong>Date:</strong> ${date}</div>
    <div><strong>Actual:</strong> <span style="color: #808080;">$${actualVal ? Number(actualVal).toFixed(2) : 'N/A'}</span></div>
    <div><strong>Predicted:</strong> <span style="color: #ff4444;">$${predictedVal ? Number(predictedVal).toFixed(2) : 'N/A'}</span></div>
  `;
  
  tooltip.style.display = 'block';
  tooltip.style.left = (mouseX + 10) + 'px';
  tooltip.style.top = (mouseY - 60) + 'px';
}

function drawChart(chart) {
  const canvas = $('chart');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const pad = { left: 70, right: 40, top: 40, bottom: 60 };

  // Store chart data for hover interactions
  chartData = { dates: chart.dates || [], actual: (chart.actual || []).map(Number), predicted: (chart.predicted || []).map(Number), pad };

  // Dark background
  ctx.fillStyle = '#141414';
  ctx.fillRect(0, 0, w, h);

  const dates = chart.dates || [];
  const actual = (chart.actual || []).map(Number);
  const predicted = (chart.predicted || []).map(Number);

  // Title
  ctx.fillStyle = '#f5f5f5';
  ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
  ctx.fillText('Actual vs Predicted Gold Price', pad.left, 24);

  // Axes
  ctx.strokeStyle = '#2d2d2d';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, h - pad.bottom);
  ctx.lineTo(w - pad.right, h - pad.bottom);
  ctx.stroke();

  if (!dates.length) {
    ctx.fillStyle = '#a8a8a8';
    ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
    ctx.fillText('No forecast revealed yet', pad.left + 20, h / 2);
    return;
  }

  const allVals = actual.concat(predicted).filter(v => Number.isFinite(v));
  let minY = Math.min(...allVals);
  let maxY = Math.max(...allVals);
  const span = maxY - minY || 1;
  minY -= span * 0.1;
  maxY += span * 0.1;

  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const x = (i) => dates.length === 1 ? pad.left + plotW / 2 : pad.left + i * plotW / (dates.length - 1);
  const y = (v) => h - pad.bottom - ((v - minY) / (maxY - minY)) * plotH;

  // Grid and Y-axis labels
  ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
  for (let i = 0; i <= 4; i++) {
    const value = minY + i * (maxY - minY) / 4;
    const yy = y(value);
    ctx.strokeStyle = '#252525';
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(w - pad.right, yy);
    ctx.stroke();
    ctx.fillStyle = '#a8a8a8';
    ctx.textAlign = 'right';
    ctx.fillText(value.toFixed(0), pad.left - 10, yy + 4);
    ctx.textAlign = 'left';
  }

  // Draw lines
  function drawLine(values, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    values.forEach((v, i) => {
      if (Number.isFinite(v)) {
        const xx = x(i), yy = y(v);
        if (i === 0 || !Number.isFinite(values[i - 1])) ctx.moveTo(xx, yy);
        else ctx.lineTo(xx, yy);
      }
    });
    ctx.stroke();
    
    // Points
    values.forEach((v, i) => {
      if (Number.isFinite(v)) {
        ctx.beginPath();
        ctx.arc(x(i), y(v), 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    });
  }

  // Actual = gray, Predicted = red
  drawLine(actual, '#808080');
  drawLine(predicted, '#ff4444');

  // Legend
  ctx.font = 'bold 13px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
  ctx.textAlign = 'right';
  ctx.fillStyle = '#808080';
  ctx.fillText('Actual', w - 50, 24);
  ctx.fillStyle = '#ff4444';
  ctx.fillText('Predicted', w - 50, 42);
  ctx.textAlign = 'left';

  // Date labels
  ctx.fillStyle = '#a8a8a8';
  ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
  ctx.fillText(dates[0], pad.left, h - 20);
  ctx.textAlign = 'right';
  ctx.fillText(dates[dates.length - 1], w - pad.right, h - 20);
  ctx.textAlign = 'left';
}

window.addEventListener('load', () => {
  checkStatus();
  drawChart(latestChart);
  const canvas = $('chart');
  canvas.addEventListener('mousemove', handleChartHover);
  canvas.addEventListener('mouseleave', () => {
    $('chartTooltip').style.display = 'none';
  });
  $('startBtn').addEventListener('click', startSimulation);
  $('nextBtn').addEventListener('click', nextDayPredict);
  $('resetBtn').addEventListener('click', resetRun);
  $('downloadBtn').addEventListener('click', downloadCsv);
});
