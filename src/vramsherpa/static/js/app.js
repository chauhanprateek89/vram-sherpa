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
  if (!gpuSelect || !vramInput || !form || !form.action.match(/\/models\//)) {
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

// Sliding thumb behind the context presets. Progressive: without this the checked label paints
// its own fill (styles.css), and with it the thumb is placed from the checked label's box.
function placeThumb(group) {
  var checked = group.querySelector("input:checked");
  var label = checked ? checked.closest("label") : null;
  if (!label) {
    return;
  }
  group.style.setProperty("--x", label.offsetLeft + "px");
  group.style.setProperty("--y", label.offsetTop + "px");
  group.style.setProperty("--w", label.offsetWidth + "px");
  group.style.setProperty("--h", label.offsetHeight + "px");
}

function placeAllThumbs() {
  document.querySelectorAll("[data-thumb]").forEach(placeThumb);
}

function initSegThumbs() {
  var groups = document.querySelectorAll("[data-thumb]");
  if (!groups.length) {
    return;
  }
  groups.forEach(function (group) {
    if (group.classList.contains("has-thumb")) {
      return;
    }
    var thumb = document.createElement("span");
    thumb.className = "thumb";
    thumb.setAttribute("aria-hidden", "true");
    group.prepend(thumb);
    group.classList.add("has-thumb");
    placeThumb(group);
  });
  // A web font changes label widths, a resize changes the layout, and a swap or a restored
  // history page can change which radio is checked.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(placeAllThumbs);
  }
  window.addEventListener("resize", placeAllThumbs);
  window.addEventListener("load", placeAllThumbs);
  window.addEventListener("pageshow", placeAllThumbs);
  document.addEventListener("change", function (event) {
    var group = event.target.closest ? event.target.closest("[data-thumb]") : null;
    if (group) {
      placeThumb(group);
    }
  });
  document.body.addEventListener("htmx:afterSettle", placeAllThumbs);
  document.body.addEventListener("htmx:historyRestore", placeAllThumbs);
}

// Filters (N): N counts checked family, quantisation and Recommended only boxes plus each
// non-empty, valid size bound, the same things the server turns into active filter chips.
// Nothing here opens or closes the disclosure (ADR-0006 point 4); the listener is on document so
// it survives any swap.
function isValidSizeBound(input) {
  if (input.value.trim() === "") {
    return false;
  }
  var value = Number(input.value);
  var min = input.min === "" ? -Infinity : Number(input.min);
  return Number.isFinite(value) && value >= min;
}

function countActiveFilters(scope) {
  var count = 0;
  scope.querySelectorAll("input[type=checkbox]:checked").forEach(function () {
    count += 1;
  });
  scope.querySelectorAll("input[type=number]").forEach(function (input) {
    if (isValidSizeBound(input)) {
      count += 1;
    }
  });
  return count;
}

function updateFilterCount() {
  document.querySelectorAll("details.filters").forEach(function (details) {
    var count = String(countActiveFilters(details));
    details.querySelectorAll("[data-fcount]").forEach(function (target) {
      target.textContent = count;
    });
  });
}

function initFilterCount() {
  document.addEventListener("input", updateFilterCount);
  document.addEventListener("change", updateFilterCount);
  window.addEventListener("pageshow", updateFilterCount);
  document.body.addEventListener("htmx:historyRestore", updateFilterCount);
  updateFilterCount();
}

document.addEventListener("DOMContentLoaded", function () {
  initThemeToggle();
  initSegThumbs();
  initFilterCount();
  initGpuInputs();
  initExampleChips();
  initModelDetailInputs();
});

// Quantisation selector on a model card. Every variant is already in the DOM, so
// switching is a visibility change rather than a round trip.
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-quant-btn]");
  if (!button) return;

  const card = button.closest("[data-model-card]");
  if (!card) return;

  const variantId = button.dataset.variant;
  card.querySelectorAll("[data-quant-btn]").forEach((other) => {
    other.setAttribute("aria-pressed", String(other === button));
  });
  card.querySelectorAll("[data-variant-pane]").forEach((pane) => {
    pane.hidden = pane.dataset.variantPane !== variantId;
  });
});
