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
}

// Styled GPU suggestion list (ADR-0005): a WAI-ARIA combobox over the server-rendered datalist.
// Progressive: the native list stays in the markup for no-JavaScript, and this removes the
// input's `list` attribute only once everything it needs is in place. Choosing writes the exact
// label and fires input and change, so the exact-match code in initGpuInputs sets gpu_id.
function normaliseGpuText(text) {
  return text.toLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

function initGpuCombobox() {
  var input = document.getElementById("gpu_search");
  var datalist = document.getElementById("gpu-options");
  var combo = input ? input.closest(".combo") : null;
  if (!input || !datalist || !combo || document.getElementById("gpu-listbox")) {
    return;
  }
  var gpus = [];
  datalist.querySelectorAll("option").forEach(function (option) {
    if (option.value) {
      gpus.push({ label: option.value, key: normaliseGpuText(option.value) });
    }
  });
  if (!gpus.length) {
    return;
  }

  var listbox = document.createElement("ul");
  listbox.id = "gpu-listbox";
  listbox.className = "listbox";
  listbox.setAttribute("role", "listbox");
  listbox.setAttribute("aria-label", "GPU suggestions");
  listbox.hidden = true;
  var nativeList = input.getAttribute("list");
  try {
    combo.appendChild(listbox);
    input.removeAttribute("list");
    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-expanded", "false");
    input.setAttribute("aria-controls", "gpu-listbox");
  } catch (err) {
    listbox.remove();
    if (nativeList) {
      input.setAttribute("list", nativeList);
    }
    return;
  }

  var shown = [];
  var active = -1;
  var choosing = false;

  function setActive(index) {
    var items = listbox.children;
    active = index;
    for (var i = 0; i < items.length; i += 1) {
      if (i === index) {
        items[i].setAttribute("aria-selected", "true");
      } else {
        items[i].removeAttribute("aria-selected");
      }
    }
    if (index >= 0 && items[index]) {
      input.setAttribute("aria-activedescendant", items[index].id);
      if (items[index].scrollIntoView) {
        items[index].scrollIntoView({ block: "nearest" });
      }
    } else {
      input.removeAttribute("aria-activedescendant");
    }
  }

  function close() {
    listbox.hidden = true;
    input.setAttribute("aria-expanded", "false");
    setActive(-1);
  }

  function open() {
    var query = normaliseGpuText(input.value);
    shown = gpus.filter(function (gpu) {
      return gpu.key.indexOf(query) !== -1;
    });
    listbox.textContent = "";
    shown.forEach(function (gpu, index) {
      var item = document.createElement("li");
      item.id = "gpu-opt-" + index;
      item.setAttribute("role", "option");
      item.textContent = gpu.label;
      listbox.appendChild(item);
    });
    active = -1;
    input.removeAttribute("aria-activedescendant");
    if (!shown.length) {
      close();
      return;
    }
    listbox.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }

  function choose(index) {
    if (index < 0 || index >= shown.length) {
      return;
    }
    choosing = true;
    input.value = shown[index].label;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    choosing = false;
    close();
    // The landing box (R-083) submits on a pick; the results page waits for the button (R-071).
    if (input.hasAttribute("data-submit-on-choose") && input.form) {
      if (typeof input.form.requestSubmit === "function") {
        input.form.requestSubmit();
      } else {
        input.form.submit();
      }
    }
  }

  function optionIndex(event) {
    var item = event.target.closest ? event.target.closest("li") : null;
    return item ? Array.prototype.indexOf.call(listbox.children, item) : -1;
  }

  input.addEventListener("input", function () {
    if (!choosing) {
      open();
    }
  });
  input.addEventListener("blur", close);
  input.addEventListener("keydown", function (event) {
    var isOpen = !listbox.hidden;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (!isOpen) {
        open();
        if (event.altKey) {
          return;
        }
      }
      if (!listbox.hidden && active < shown.length - 1) {
        setActive(active + 1);
      }
    } else if (event.key === "ArrowUp") {
      if (isOpen) {
        event.preventDefault();
        if (active > 0) {
          setActive(active - 1);
        }
      }
    } else if (event.key === "Enter") {
      if (isOpen && active >= 0) {
        event.preventDefault();
        choose(active);
      }
    } else if (event.key === "Escape") {
      if (isOpen) {
        event.preventDefault();
        close();
      }
    }
  });

  // pointerdown keeps focus in the field (no blur, so the list stays) and chooses at once for a
  // mouse or pen. A touch starts as a scroll, so it chooses on the click that follows a tap.
  listbox.addEventListener("pointerdown", function (event) {
    event.preventDefault();
    if (event.pointerType !== "touch") {
      choose(optionIndex(event));
    }
  });
  listbox.addEventListener("mousedown", function (event) {
    event.preventDefault();
  });
  listbox.addEventListener("click", function (event) {
    if (!listbox.hidden) {
      choose(optionIndex(event));
    }
  });
}

// A blank landing box does nothing: no navigation, no message, focus stays in the box (R-083).
function initLandingForm() {
  var form = document.getElementById("landing-form");
  var input = document.getElementById("gpu_search");
  if (!form || !input) {
    return;
  }
  form.addEventListener("submit", function (event) {
    if (input.value.trim() === "") {
      event.preventDefault();
      input.focus();
    }
  });
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

// Sliding thumb behind the context presets and the quantisation buttons. Progressive: without
// this the checked label or pressed button paints its own fill (styles.css), and with it the
// thumb is placed from that element's box.
function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

function thumbTarget(group) {
  var checked = group.querySelector("input:checked");
  if (checked) {
    return checked.closest("label");
  }
  return group.querySelector('[aria-pressed="true"]');
}

function placeThumb(group) {
  var target = thumbTarget(group);
  if (!target) {
    return;
  }
  group.style.setProperty("--x", target.offsetLeft + "px");
  group.style.setProperty("--y", target.offsetTop + "px");
  group.style.setProperty("--w", target.offsetWidth + "px");
  group.style.setProperty("--h", target.offsetHeight + "px");
}

function placeAllThumbs() {
  document.querySelectorAll("[data-thumb]").forEach(placeThumb);
}

// Idempotent: adds a thumb to each [data-thumb] group that has none, so it can run again after
// htmx swaps in new cards.
function addThumbs() {
  document.querySelectorAll("[data-thumb]").forEach(function (group) {
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
}

function initSegThumbs() {
  addThumbs();
  // A web font changes label widths, a resize changes the layout, and a swap or a restored
  // history page can change which control is checked or pressed.
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
  function refreshThumbs() {
    addThumbs();
    placeAllThumbs();
  }
  document.body.addEventListener("htmx:afterSettle", refreshThumbs);
  document.body.addEventListener("htmx:historyRestore", refreshThumbs);
}

// Pointer sheen on cards and Top Picks: a soft highlight that follows a fine pointer. One
// delegated listener, so it covers cards swapped in later.
function initSheen() {
  if (
    prefersReducedMotion() ||
    !window.matchMedia ||
    !window.matchMedia("(hover: hover) and (pointer: fine)").matches
  ) {
    return;
  }
  document.addEventListener("pointermove", function (event) {
    var el = event.target.closest ? event.target.closest("[data-sheen]") : null;
    if (!el) {
      return;
    }
    var rect = el.getBoundingClientRect();
    el.style.setProperty("--mx", event.clientX - rect.left + "px");
    el.style.setProperty("--my", event.clientY - rect.top + "px");
  });
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

// An in-place refresh swaps #results-shell but not the form, so the VRAM field's invalid state is
// re-read from the new summary (data-vram-invalid) after each swap. Without it a fixed value
// would stay marked and a newly rejected one would not be (R-071).
function syncVramInvalid() {
  var banner = document.getElementById("summary-banner");
  var field = document.getElementById("vram_gb");
  if (!banner || !field) {
    return;
  }
  if (banner.dataset.vramInvalid === "true") {
    field.setAttribute("aria-invalid", "true");
  } else {
    field.removeAttribute("aria-invalid");
  }
}

// A number typed in Search GPU is read by the server as VRAM (ADR-0011, R-003). The summary says
// so with data-box-vram; copy it into the form so it matches /results?vram_gb=<n>. Runs after
// syncVramInvalid so a box number of 0 keeps aria-invalid on the field.
function syncBoxVram() {
  var banner = document.getElementById("summary-banner");
  var field = document.getElementById("vram_gb");
  if (!banner || !field || banner.dataset.boxVram === undefined) {
    return;
  }
  field.value = banner.dataset.boxVram;
  var search = document.getElementById("gpu_search");
  var gpuId = document.getElementById("gpu_id");
  if (search) {
    search.value = "";
  }
  if (gpuId) {
    gpuId.value = "";
  }
}

function initVramInvalid() {
  document.body.addEventListener("htmx:afterSettle", syncVramInvalid);
  document.body.addEventListener("htmx:afterSettle", syncBoxVram);
}

document.addEventListener("DOMContentLoaded", function () {
  initThemeToggle();
  initVramInvalid();
  initSegThumbs();
  initSheen();
  initFilterCount();
  initGpuInputs();
  initGpuCombobox();
  initLandingForm();
  initExampleChips();
  initModelDetailInputs();
});

// Quantisation selector on a model card. Every variant is already in the DOM, so
// switching is a visibility change rather than a round trip. The pane ease-in and the gauge
// refill are motion only, and are skipped under prefers-reduced-motion.
function replayGauge(pane) {
  pane.classList.add("htmx-added");
  void pane.offsetWidth;
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      pane.classList.remove("htmx-added");
    });
  });
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-quant-btn]");
  if (!button) return;

  const card = button.closest("[data-model-card]");
  if (!card) return;

  const variantId = button.dataset.variant;
  const animate = !prefersReducedMotion();
  card.querySelectorAll("[data-quant-btn]").forEach((other) => {
    other.setAttribute("aria-pressed", String(other === button));
  });
  card.querySelectorAll("[data-variant-pane]").forEach((pane) => {
    const show = pane.dataset.variantPane === variantId;
    pane.hidden = !show;
    pane.classList.remove("is-entering");
    if (show && animate) {
      void pane.offsetWidth;
      pane.classList.add("is-entering");
      replayGauge(pane);
    }
  });
  const group = button.closest("[data-thumb]");
  if (group) placeThumb(group);
});
