const metricNames = {
  score: 'EgoPath Score',
  a1_ba: 'Point Trav. BA', a1_f1: 'Point Trav. F1',
  b1_ba: 'Embodied Trav. BA', b1_f1: 'Embodied Trav. F1',
  a2_vpr: 'Point Path VPR', a2_sr: 'Point Path SR', a2_spl: 'Point Path SPL',
  b2_vpr: 'Embodied Path VPR', b2_sr: 'Embodied Path SR', b2_spl: 'Embodied Path SPL',
  c_vpr: 'Intent Path VPR', c_sr: 'Intent Path SR', c_spl: 'Intent Path SPL'
};
const overview = ['score', 'a1_ba', 'b1_ba', 'a2_sr', 'b2_sr', 'c_sr'];
const table = document.getElementById('model-table');
const metricView = document.getElementById('metric-view');
const statusLine = document.getElementById('board-status');
const settingButtons = document.querySelectorAll('[data-setting]');
let models = [];
let setting = 'Zero-shot';
let sortMetric = 'score';
let descending = true;

function renderBoard() {
  const group = models.filter(model => model.setting === setting);
  const view = metricView.value;
  const columns = view === 'overall' ? overview : ['score', ...Object.keys(metricNames).filter(key => key.startsWith(view + '_'))];
  const order = group.slice().sort((a, b) => Number(b[sortMetric]) - Number(a[sortMetric]));
  const header = document.createElement('tr');
  for (const label of ['Rank', 'Model']) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = label;
    header.append(th);
  }
  for (const key of columns) {
    const th = document.createElement('th');
    th.scope = 'col';
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.metric = key;
    button.textContent = metricNames[key];
    const active = key === sortMetric;
    if (active) th.setAttribute('aria-sort', descending ? 'descending' : 'ascending');
    button.setAttribute('aria-label', 'Sort by ' + metricNames[key] + (active && descending ? ', ascending' : ', descending'));
    th.append(button);
    header.append(th);
  }
  table.tHead.replaceChildren(header);

  const rows = descending ? order : order.slice().reverse();
  table.tBodies[0].replaceChildren(...rows.map(model => {
    const row = document.createElement('tr');
    const rank = order.findIndex(item => item[sortMetric] === model[sortMetric]) + 1;
    if (rank === 1) row.className = 'first-place';
    row.insertCell().textContent = rank;
    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = model.model;
    row.append(name);
    for (const key of columns) {
      const cell = row.insertCell();
      cell.textContent = model[key];
      if (key === sortMetric) cell.className = 'sorted-value';
    }
    return row;
  }));
  const groupName = setting === 'Zero-shot' ? '10 zero-shot models' : 'Qwen3.5-4B before and after EgoPath-SFT';
  statusLine.textContent = groupName + ' · Sorted by ' + metricNames[sortMetric] + (descending ? ' (highest first)' : ' (lowest first)');
}

function changeView() {
  const view = metricView.value;
  sortMetric = view === 'overall' ? 'score' : view + (view === 'a1' || view === 'b1' ? '_ba' : '_sr');
  descending = true;
  if (models.length) renderBoard();
}
metricView.addEventListener('change', changeView);
for (const button of settingButtons) {
  button.addEventListener('click', () => {
    setting = button.dataset.setting;
    for (const tab of settingButtons) tab.setAttribute('aria-pressed', String(tab === button));
    changeView();
  });
}
table.tHead.addEventListener('click', event => {
  const button = event.target.closest('[data-metric]');
  if (!button) return;
  descending = button.dataset.metric === sortMetric ? !descending : true;
  sortMetric = button.dataset.metric;
  renderBoard();
  table.tHead.querySelector('[data-metric="' + sortMetric + '"]').focus({preventScroll: true});
});

fetch('leaderboard.csv').then(response => {
  if (!response.ok) throw new Error('Leaderboard unavailable');
  return response.text();
}).then(csv => {
  // The published table contains plain model names and decimal numbers.
  const [header, ...rows] = csv.trim().split(/\r?\n/);
  const fields = header.split(',');
  models = rows.map(line => Object.fromEntries(line.split(',').map((value, index) => [fields[index], value])));
  const leaders = models.filter(row => row.setting === 'Zero-shot')
    .sort((a, b) => Number(b.score) - Number(a.score)).slice(0, 3);
  const podium = document.getElementById('top-models');
  for (const model of leaders) {
    const item = document.createElement('li');
    const score = document.createElement('strong');
    score.textContent = model.score;
    const name = document.createElement('span');
    name.textContent = model.model;
    item.append(score, name);
    podium.append(item);
  }
  podium.parentElement.hidden = false;
  renderBoard();
}).catch(() => {
  statusLine.textContent = 'The table could not be loaded. Use Download CSV to read the results.';
});
