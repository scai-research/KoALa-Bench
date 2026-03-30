const HOME_VIEW = "HOME";

const TASK_ORDER = [
  "K-disentQA",
  "SQA",
  "Instruct",
  "ASR",
  "Translation",
  "LSQA",
];

const OVERALL_BADGE_IMAGES = {
  1: "./images/1st.png",
  2: "./images/2nd.png",
  3: "./images/3rd.png",
};

const ui = {
  taskMenu: document.getElementById("taskMenu"),
  homeView: document.getElementById("homeView"),
  taskView: document.getElementById("taskView"),
};

const state = {
  activeView: HOME_VIEW,
  tasks: [],
  taskMap: {},
  entries: [],
  loading: true,
  error: "",
};

function toNum(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function average(values) {
  const valid = values.filter((value) => value !== null);
  if (!valid.length) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function compareScores(a, b, lowerBetter) {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return lowerBetter ? a - b : b - a;
}

function orderedTasks(tasks) {
  return [...tasks].sort((left, right) => {
    const leftIndex = TASK_ORDER.indexOf(left.id);
    const rightIndex = TASK_ORDER.indexOf(right.id);
    const normalizedLeft = leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex;
    const normalizedRight = rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex;
    if (normalizedLeft === normalizedRight) {
      return left.label.localeCompare(right.label);
    }
    return normalizedLeft - normalizedRight;
  });
}

function buildTaskMap(tasks) {
  return tasks.reduce((accumulator, task) => {
    accumulator[task.id] = task;
    return accumulator;
  }, {});
}

function datasetIds(task) {
  return (task.datasets || []).map((dataset) => dataset.id);
}

function metricValue(entry, taskId, datasetId) {
  if (!entry.tasks || !entry.tasks[taskId]) return null;
  const dataset = entry.tasks[taskId][datasetId];
  return dataset ? toNum(dataset.value) : null;
}

function metricDisplay(entry, taskId, datasetId) {
  if (!entry.tasks || !entry.tasks[taskId]) return "-";
  const dataset = entry.tasks[taskId][datasetId];
  if (!dataset) return "-";
  return dataset.display || (dataset.value === null ? "-" : String(dataset.value));
}

function computeTaskOverall(entry, task) {
  return average(datasetIds(task).map((datasetId) => metricValue(entry, task.id, datasetId)));
}

function normalizeTaskScores(entries, tasks) {
  return tasks.reduce((accumulator, task) => {
    const values = entries
      .map((entry) => entry.taskOverall[task.id])
      .filter((value) => value !== null);

    if (!values.length) {
      accumulator[task.id] = null;
      return accumulator;
    }

    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    accumulator[task.id] = { minValue, maxValue };
    return accumulator;
  }, {});
}

function normalizedScore(value, range, lowerBetter) {
  if (value === null || !range) return null;
  if (range.maxValue === range.minValue) return 100;
  if (lowerBetter) {
    return ((range.maxValue - value) / (range.maxValue - range.minValue)) * 100;
  }
  return ((value - range.minValue) / (range.maxValue - range.minValue)) * 100;
}

function enrichEntries(entries, tasks) {
  const withTaskOverall = entries.map((entry) => {
    const taskOverall = {};
    tasks.forEach((task) => {
      taskOverall[task.id] = computeTaskOverall(entry, task);
    });
    return { ...entry, taskOverall };
  });

  const ranges = normalizeTaskScores(withTaskOverall, tasks);
  return withTaskOverall.map((entry) => {
    const normalizedTaskScores = {};
    tasks.forEach((task) => {
      normalizedTaskScores[task.id] = normalizedScore(
        entry.taskOverall[task.id],
        ranges[task.id],
        task.lowerBetter,
      );
    });

    return {
      ...entry,
      normalizedTaskScores,
      overall: average(tasks.map((task) => normalizedTaskScores[task.id])),
    };
  });
}

function sortOverall(entries) {
  return [...entries]
    .sort((left, right) => compareScores(left.overall, right.overall, false))
    .map((entry, index) => ({ ...entry, rank: index + 1 }));
}

function sortTask(entries, task, datasetId) {
  return [...entries]
    .sort((left, right) => {
      const leftValue =
        datasetId === "Overall"
          ? left.taskOverall[task.id]
          : metricValue(left, task.id, datasetId);
      const rightValue =
        datasetId === "Overall"
          ? right.taskOverall[task.id]
          : metricValue(right, task.id, datasetId);
      return compareScores(leftValue, rightValue, task.lowerBetter);
    })
    .map((entry, index) => ({ ...entry, rank: index + 1 }));
}

function metricClass(lowerBetter, value) {
  if (value === null) return "muted";
  return lowerBetter ? "metric-bad" : "metric-good";
}

function renderMenu() {
  const items = [
    { key: HOME_VIEW, label: "Home", meta: "Overall ranking" },
    ...state.tasks.map((task) => ({
      key: task.id,
      label: task.label,
      meta: `${task.datasets.length} datasets`,
    })),
  ];

  ui.taskMenu.innerHTML = items
    .map(
      (item) => `
        <button type="button" class="${state.activeView === item.key ? "active" : ""}" data-view="${item.key}">
          <span class="menu-label">${item.label}</span>
          <span class="menu-meta">${item.meta}</span>
        </button>
      `,
    )
    .join("");

  ui.taskMenu.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeView = button.dataset.view;
      renderApp();
    });
  });
}

function renderRankStrip(entries) {
  return `
    <section class="section-card card">
      <div class="section-head">
        <h3>Top Ranking</h3>
      </div>
      <div class="rank-strip-list">
        ${entries
          .slice(0, 12)
          .map(
            (entry) => `
              <article class="rank-pill">
                <div class="rank-badge-wrap">
                  ${
                    OVERALL_BADGE_IMAGES[entry.rank]
                      ? `<img class="rank-badge-image" src="${OVERALL_BADGE_IMAGES[entry.rank]}" alt="${entry.rank} place" />`
                      : `<span class="rank-badge">#${entry.rank}</span>`
                  }
                </div>
                <span class="rank-name">${entry.rank_name}</span>
              </article>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderHomeTable(entries) {
  const headerTop = [
    '<th rowspan="2" class="rank-col col-rank">Rank</th>',
    '<th rowspan="2" class="col-rankname">RankName</th>',
    '<th rowspan="2" class="col-model">Model</th>',
    '<th rowspan="2">URL</th>',
    '<th rowspan="2">Overall</th>',
    ...state.tasks.map((task) => `<th class="grouped" colspan="1">${task.label}</th>`),
  ].join("");

  const headerBottom = state.tasks
    .map((task) => `<th>${task.shortMetric}</th>`)
    .join("");

  const rows = entries
    .map((entry) => {
      const urlCell = entry.url
        ? `<a class="url-link" href="${entry.url}" target="_blank" rel="noopener noreferrer" aria-label="External link"><img src="./images/external-link.png" alt="" /></a>`
        : "-";

      const taskCells = state.tasks
        .map((task) => {
          const value = entry.taskOverall[task.id];
          return `<td><span class="${metricClass(task.lowerBetter, value)}">${value === null ? "-" : value.toFixed(2)}</span></td>`;
        })
        .join("");

      return `
        <tr>
          <td class="rank-col col-rank">${entry.rank}</td>
          <td class="col-rankname">${entry.rank_name}</td>
          <td class="col-model">${entry.model || entry.rank_name}</td>
          <td>${urlCell}</td>
          <td>${entry.overall === null ? "-" : entry.overall.toFixed(2)}</td>
          ${taskCells}
        </tr>
      `;
    })
    .join("");

  return `
    <section class="section-card card">
      <div class="section-head">
        <h3>Overall Leaderboard</h3>
      </div>
      <div class="table-scroll">
        <table>
          <colgroup>
            <col class="col-rank" />
            <col class="col-rankname" />
            <col class="col-model" />
            <col />
            <col />
            ${state.tasks.map(() => "<col />").join("")}
          </colgroup>
          <thead>
            <tr>${headerTop}</tr>
            <tr>${headerBottom}</tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;
}

function renderTask() {
  const task = state.taskMap[state.activeView];
  if (!task) {
    ui.taskView.innerHTML = "";
    return;
  }

  const options = [{ id: "Overall", label: "Overall" }, ...task.datasets];
  const initialDatasetId = options[0].id;

  ui.taskView.innerHTML = `
    <section class="section-card card">
      <div class="section-head">
        <div>
          <h3 class="task-title">Task : ${task.label}</h3>
        </div>
      </div>
      <div class="task-filters">
        <label class="field">
          <span>Dataset</span>
          <select id="taskDatasetSelect">
            ${options
              .map((dataset) => `<option value="${dataset.id}">${dataset.label}</option>`)
              .join("")}
          </select>
        </label>
        <label class="field">
          <span>Metric</span>
          <input type="text" value="${task.metricLabel}" readonly />
        </label>
      </div>
      <div id="taskTableMount"></div>
    </section>
  `;

  const datasetSelect = document.getElementById("taskDatasetSelect");
  const tableMount = document.getElementById("taskTableMount");

  function drawTaskTable(datasetId) {
    const rankedEntries = sortTask(state.entries, task, datasetId);
    const activeLabel =
      datasetId === "Overall"
        ? "Overall"
        : (task.datasets.find((dataset) => dataset.id === datasetId) || {}).label || datasetId;

    const rows = rankedEntries
      .map((entry) => {
        const numericValue =
          datasetId === "Overall"
            ? entry.taskOverall[task.id]
            : metricValue(entry, task.id, datasetId);
        const displayValue =
          datasetId === "Overall"
            ? (numericValue === null ? "-" : numericValue.toFixed(2))
            : metricDisplay(entry, task.id, datasetId);

        return `
          <tr>
            <td class="rank-col col-rank">${entry.rank}</td>
            <td class="col-rankname">${entry.rank_name}</td>
            <td class="col-model">${entry.model || entry.rank_name}</td>
            <td><span class="${metricClass(task.lowerBetter, numericValue)}">${displayValue}</span></td>
          </tr>
        `;
      })
      .join("");

    tableMount.innerHTML = `
      <div class="table-scroll">
        <table class="task-performance-table">
          <colgroup>
            <col class="col-rank" />
            <col class="col-rankname" />
            <col class="col-model" />
            <col />
          </colgroup>
          <thead>
            <tr>
              <th class="rank-col col-rank">Rank</th>
              <th class="col-rankname">RankName</th>
              <th class="col-model">Model</th>
              <th>${activeLabel}</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  }

  datasetSelect.addEventListener("change", (event) => {
    drawTaskTable(event.target.value);
  });

  drawTaskTable(initialDatasetId);
}

function renderLoading() {
  const content = `
    <section class="section-card card">
      <div class="section-head">
        <h3>Loading</h3>
      </div>
      <p class="section-meta">Reading aggregated summary data...</p>
    </section>
  `;
  ui.homeView.innerHTML = content;
  ui.taskView.innerHTML = "";
}

function renderError() {
  const content = `
    <section class="section-card card">
      <div class="section-head">
        <h3>Data Load Failed</h3>
      </div>
      <p class="section-meta">${state.error}</p>
    </section>
  `;
  ui.homeView.innerHTML = content;
  ui.taskView.innerHTML = "";
}

function renderHome() {
  const rankedEntries = sortOverall(state.entries);
  ui.homeView.innerHTML = `${renderRankStrip(rankedEntries)}${renderHomeTable(rankedEntries)}`;
}

function renderApp() {
  renderMenu();

  if (state.loading) {
    ui.homeView.classList.remove("hidden");
    ui.taskView.classList.add("hidden");
    renderLoading();
    return;
  }

  if (state.error) {
    ui.homeView.classList.remove("hidden");
    ui.taskView.classList.add("hidden");
    renderError();
    return;
  }

  if (state.activeView === HOME_VIEW) {
    ui.homeView.classList.remove("hidden");
    ui.taskView.classList.add("hidden");
    ui.taskView.innerHTML = "";
    renderHome();
    return;
  }

  ui.homeView.classList.add("hidden");
  ui.homeView.innerHTML = "";
  ui.taskView.classList.remove("hidden");
  renderTask();
}

async function loadLeaderboardData() {
  try {
    const response = await fetch("./data/leaderboard-data.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    state.tasks = orderedTasks(payload.tasks || []);
    state.taskMap = buildTaskMap(state.tasks);
    state.entries = enrichEntries(payload.entries || [], state.tasks);
    state.loading = false;
    state.error = "";
  } catch (error) {
    state.loading = false;
    state.error = error.message;
  }

  renderApp();
}

renderApp();
loadLeaderboardData();
