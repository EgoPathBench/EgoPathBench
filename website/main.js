const taskDetails = {
  a1: {agent: 'POINT AGENT', count: 146, title: 'Where can a point agent go?', description: 'Select every visible candidate that is traversable when the agent has no body width. This tests waypoint-to-scene correspondence and candidate feasibility.', output: 'An unordered set of traversable waypoint IDs.', check: 'Candidate feasibility: balanced accuracy and F1.'},
  b1: {agent: 'EMBODIED AGENT · 0.6 M DIAMETER', count: 146, title: 'Where can a robot fit?', description: 'Use the same visual candidate space while accounting for a physical body. A location that works for a point agent may have insufficient clearance for the robot.', output: 'An unordered set of robot-traversable waypoint IDs.', check: 'Embodied candidate feasibility: balanced accuracy and F1.'},
  a2: {agent: 'POINT AGENT', count: 309, title: 'Can the selected steps reach the goal?', description: 'Ground an explicit target and compose an ordered route for a point agent. Each adjacent pair must form a legal transition, and the endpoint must reach an acceptable goal region.', output: 'An ordered route of visible waypoint IDs.', check: 'Route validity, goal-reaching success, and path efficiency.'},
  b2: {agent: 'EMBODIED AGENT · 0.6 M DIAMETER', count: 309, title: 'Can the whole route accommodate a body?', description: 'Keep the scene, view, and target fixed, but plan with the robot footprint. Every transition along the route must satisfy embodied geometry.', output: 'An ordered route of visible waypoint IDs.', check: 'Embodied route validity, goal-reaching success, and path efficiency.'},
  c: {agent: 'EMBODIED AGENT + INTENT', count: 201, title: 'Can intent become a feasible route?', description: 'Resolve an intended object from the user’s need and a visual cue, then plan an embodied route to it. Goal grounding and route feasibility must succeed together.', output: 'An ordered route to the resolved target.', check: 'Intent-grounded embodied route validity, success, and efficiency.'}
};
const tabs = [...document.querySelectorAll('[role="tab"]')];
let examples;
let activeTask = 'a1';
function selectTask(task) {
  activeTask = task;
  const data = taskDetails[task];
  tabs.forEach(tab => {
    const selected = tab.dataset.task === task;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.getElementById('task-panel').setAttribute('aria-labelledby', 'tab-' + task);
  for (const [field, value] of Object.entries(data)) {
    document.getElementById('task-' + field).textContent = field === 'count' ? value + ' benchmark questions' : value;
  }
  const image = document.getElementById('task-image');
  image.src = 'assets/task-' + task + '.png';
  image.alt = data.title + ' Actual benchmark observation with numbered visible waypoints.';
  const example = examples?.find(row => row.task === task);
  document.getElementById('task-prompt').textContent = example ? example.prompt : 'Loading the original benchmark prompt…';
}
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener('click', () => selectTask(tab.dataset.task));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      tabs[next].focus();
      selectTask(tabs[next].dataset.task);
    }
  });
}
fetch('examples.json').then(response => {
  if (!response.ok) throw new Error('Example file unavailable');
  return response.json();
}).then(data => {
  examples = data;
  selectTask(activeTask);
}).catch(() => {
  document.getElementById('task-prompt').textContent = 'The example prompt could not be loaded. Please reload or view the released examples in the repository.';
});

// Values transcribed from the author-selected PDF, Figure 3. Rates use all
// predictions and are parallel diagnostics, not a sequential funnel.
const diagnostics = {
  a2: [96.0, 28.9, 46.8, 16.7],
  b2: [90.4, 4.8, 4.8, 1.0],
  c: [88.0, 5.2, 6.5, 1.4]
};
const diagnosticLabels = ['Legal first edge', 'Goal-consistent endpoint', 'Complete route legality', 'Joint route success'];
function renderDiagnostics() {
  const container = document.getElementById('diagnostic-bars');
  container.replaceChildren();
  diagnostics[document.getElementById('diagnostic-task').value].forEach((value, index) => {
    const row = document.createElement('div');
    row.className = 'diagnostic-row';
    const heading = document.createElement('div');
    heading.className = 'bar-heading';
    const label = document.createElement('span');
    label.textContent = diagnosticLabels[index];
    const number = document.createElement('strong');
    number.textContent = value.toFixed(1) + '%';
    heading.append(label, number);
    const track = document.createElement('div');
    track.className = 'bar-track';
    track.setAttribute('aria-hidden', 'true');
    const bar = document.createElement('i');
    bar.style.width = value + '%';
    track.appendChild(bar);
    row.append(heading, track);
    container.appendChild(row);
  });
}
document.getElementById('diagnostic-task').addEventListener('change', renderDiagnostics);
renderDiagnostics();

// Table 3 of the author-selected July 25 PDF; no superseded aggregate result.
const transferResults = [
  ['VSI-Bench Route Planning · Full', 29.38, 33.51, 2],
  ['VSI-Bench Route Planning · Debiased', 20.18, 24.56, 2],
  ['SpatialEval-VTQA · Full', 61.8, 71.4, 1],
  ['3DSRBench · Full', 58.0, 59.4, 1]
];
for (const [label, base, sft, precision] of transferResults) {
  const row = document.createElement('div');
  row.className = 'transfer-row';
  const heading = document.createElement('div');
  heading.className = 'transfer-title';
  const name = document.createElement('span');
  name.textContent = label;
  const gain = document.createElement('strong');
  gain.textContent = '+' + (sft - base).toFixed(precision) + ' pp';
  heading.append(name, gain);
  const bars = document.createElement('div');
  bars.className = 'paired-bars';
  [base, sft].forEach((value, index) => {
    const track = document.createElement('div');
    track.style.setProperty('--value', value + '%');
    track.setAttribute('aria-label', (index === 0 ? 'Base: ' : 'After SFT: ') + value.toFixed(precision) + '%');
    const bar = document.createElement('i');
    bar.className = index === 0 ? 'base' : 'sft';
    bar.style.width = value + '%';
    const number = document.createElement('b');
    number.textContent = value.toFixed(precision);
    track.append(bar, number);
    bars.appendChild(track);
  });
  row.append(heading, bars);
  document.getElementById('transfer-chart').appendChild(row);
}
