function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("vramsherpa-theme", theme);
  } catch (err) {
    // Ignore storage errors in restricted browser contexts.
  }
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.textContent = theme === "dark" ? "Light mode" : "Dark mode";
    toggle.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
  }
}

function initThemeToggle() {
  var toggle = document.getElementById("theme-toggle");
  if (!toggle) {
    return;
  }

  var current = document.documentElement.dataset.theme || "";
  if (!current) {
    current =
      window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    applyTheme(current);
  } else {
    toggle.textContent = current === "dark" ? "Light mode" : "Dark mode";
    toggle.setAttribute("aria-pressed", current === "dark" ? "true" : "false");
  }

  toggle.addEventListener("click", function () {
    var active = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    applyTheme(active === "dark" ? "light" : "dark");
  });
}

function findGpuOptionByValue(value) {
  var options = document.querySelectorAll("#gpu-options option");
  for (var i = 0; i < options.length; i += 1) {
    if (options[i].value === value) {
      return options[i];
    }
  }
  return null;
}

function findGpuOptionById(gpuId) {
  var options = document.querySelectorAll("#gpu-options option");
  for (var i = 0; i < options.length; i += 1) {
    if (options[i].getAttribute("data-gpu-id") === gpuId) {
      return options[i];
    }
  }
  return null;
}

function initGpuInputs() {
  var gpuSearch = document.getElementById("gpu_search");
  var gpuIdInput = document.getElementById("gpu_id");
  var vramInput = document.getElementById("vram_gb");
  var clearButton = document.getElementById("clear-gpu");
  if (!gpuSearch || !gpuIdInput) {
    return;
  }

  function syncGpuSelection() {
    var match = findGpuOptionByValue(gpuSearch.value);
    if (match) {
      gpuIdInput.value = match.getAttribute("data-gpu-id") || "";
      if (vramInput) {
        vramInput.value = "";
      }
      return;
    }
    gpuIdInput.value = "";
  }

  gpuSearch.addEventListener("input", syncGpuSelection);
  gpuSearch.addEventListener("change", syncGpuSelection);
  gpuSearch.addEventListener("blur", syncGpuSelection);
  syncGpuSelection();

  if (vramInput) {
    vramInput.addEventListener("input", function () {
      if (vramInput.value) {
        gpuIdInput.value = "";
      }
    });
  }

  if (clearButton) {
    clearButton.addEventListener("click", function () {
      gpuSearch.value = "";
      gpuIdInput.value = "";
      gpuSearch.focus();
    });
  }
}

function initExampleChips() {
  var chips = document.querySelectorAll(".chip-example");
  if (!chips.length) {
    return;
  }
  var gpuSearch = document.getElementById("gpu_search");
  var gpuIdInput = document.getElementById("gpu_id");
  var vramInput = document.getElementById("vram_gb");

  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      var gpuId = chip.getAttribute("data-gpu-id");
      var vramGb = chip.getAttribute("data-vram-gb");

      if (gpuId && gpuIdInput && gpuSearch) {
        var option = findGpuOptionById(gpuId);
        gpuIdInput.value = gpuId;
        if (option) {
          gpuSearch.value = option.value;
        }
        if (vramInput) {
          vramInput.value = "";
        }
      }

      if (vramGb && vramInput) {
        vramInput.value = vramGb;
        if (gpuIdInput) {
          gpuIdInput.value = "";
        }
      }
    });
  });
}

function initModelDetailInputs() {
  var gpuSelect = document.getElementById("gpu_id");
  var vramInput = document.getElementById("vram_gb");
  var form = gpuSelect ? gpuSelect.closest("form") : null;
  if (!gpuSelect || !vramInput || !form || !form.action.match(/\\/models\\//)) {
    return;
  }

  gpuSelect.addEventListener("change", function () {
    if (gpuSelect.value) {
      vramInput.value = "";
    }
  });

  vramInput.addEventListener("input", function () {
    if (vramInput.value) {
      gpuSelect.value = "";
    }
  });
}

document.addEventListener("DOMContentLoaded", function () {
  initThemeToggle();
  initGpuInputs();
  initExampleChips();
  initModelDetailInputs();
});
