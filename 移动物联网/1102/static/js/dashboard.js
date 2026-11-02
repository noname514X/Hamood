const clockDate = document.querySelector("#clock-date");
const clockTime = document.querySelector("#clock-time");
const temperatureForm = document.querySelector("#temperatureForm");
const targetInput = document.querySelector("#target_temperature");
const targetBadge = document.querySelector("#targetBadge");
const targetFeedback = document.querySelector("#targetFeedback");
const temperatureCard = document.querySelector(".temperature-card");
const comfortMarker = document.querySelector("#comfort-marker");
const comfortStatus = document.querySelector("#comfort-status");

const dateFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "2-digit",
  day: "2-digit",
  weekday: "short",
});

const timeFormatter = new Intl.DateTimeFormat("zh-CN", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

function updateClock() {
  const now = new Date();
  clockDate.textContent = dateFormatter.format(now);
  clockTime.textContent = timeFormatter.format(now);
}

function getCurrentTemperature() {
  return Number(temperatureCard.dataset.temperature);
}

function updateComfortMarker() {
  const temperature = getCurrentTemperature();
  const percentage = Math.min(100, Math.max(0, ((temperature - 10) / 30) * 100));

  comfortMarker.style.left = `${percentage}%`;

  if (temperature < 18) {
    comfortStatus.textContent = "偏冷";
  } else if (temperature > 28) {
    comfortStatus.textContent = "偏热";
  } else {
    comfortStatus.textContent = "舒适";
  }
}

function normalizeTarget(value) {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) {
    return null;
  }

  return Math.min(50, Math.max(-10, Math.round(parsed * 10) / 10));
}

function updateTargetFeedback() {
  const target = normalizeTarget(targetInput.value);
  const current = getCurrentTemperature();

  if (target === null) {
    targetBadge.textContent = "未设置";
    targetFeedback.textContent = "输入温度后即可更新设定";
    return;
  }

  const difference = target - current;
  targetBadge.textContent = `${target.toFixed(1)} °C`;

  if (Math.abs(difference) < 0.05) {
    targetFeedback.textContent = "目标温度与当前环境一致";
  } else {
    const direction = difference > 0 ? "升高" : "降低";
    targetFeedback.textContent = `较当前环境${direction} ${Math.abs(difference).toFixed(1)} °C`;
  }
}

document.querySelectorAll(".step-button").forEach((button) => {
  button.addEventListener("click", () => {
    const delta = Number(button.dataset.delta);
    const currentValue = normalizeTarget(targetInput.value) ?? 24;
    const nextValue = Math.min(50, Math.max(-10, currentValue + delta));

    targetInput.value = nextValue.toFixed(1);
    updateTargetFeedback();
    targetInput.focus();
  });
});

targetInput.addEventListener("input", updateTargetFeedback);

temperatureForm.addEventListener("submit", (event) => {
  const target = normalizeTarget(targetInput.value);

  if (target === null) {
    event.preventDefault();
    targetInput.focus();
    targetFeedback.textContent = "请输入有效的目标温度";
    return;
  }

  targetInput.value = target.toFixed(1);
});

document.querySelectorAll(".rail-link").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".rail-link").forEach((item) => item.classList.remove("is-active"));
    link.classList.add("is-active");
  });
});

updateClock();
setInterval(updateClock, 1000);
updateComfortMarker();
updateTargetFeedback();
