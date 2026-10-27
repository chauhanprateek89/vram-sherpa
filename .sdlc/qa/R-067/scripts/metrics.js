() => {
  const vis = el => { const r = el.getBoundingClientRect(), cs = getComputedStyle(el); return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none'; };
  const r = { scrollWidth: document.documentElement.scrollWidth, docHeight: document.documentElement.scrollHeight, dtheme: document.documentElement.dataset.theme };
  const small = [];
  document.querySelectorAll('a, button, summary, select, input:not([type=hidden]):not([type=checkbox]):not([type=radio]), label.choice, .seg label').forEach(el => {
    if (!vis(el) || el.closest('.sr-only') || el.closest('template')) return;
    if (el.matches('.pick a')) return; if (el.matches('a') && el.closest('p, li') && !el.closest('.chip-link, .action-chip, .sources-list, nav') && !el.closest('.pick')) return;
    const b = el.getBoundingClientRect(); if (b.height < 43.5 || b.width < 43.5) small.push(el.tagName + '.' + (el.className || el.id) + ' ' + Math.round(b.width) + 'x' + Math.round(b.height));
  });
  r.smallTargets = small;
  let minFs = 99, minEl = '';
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (w.nextNode()) { const n = w.currentNode; if (!n.textContent.trim()) continue; const el = n.parentElement; if (!vis(el) || el.closest('.sr-only,template,svg,script,style')) continue; const f = parseFloat(getComputedStyle(el).fontSize); if (f < minFs) { minFs = f; minEl = el.tagName + '.' + el.className + ' ' + n.textContent.trim().slice(0, 20); } }
  r.minFont = minFs; r.minFontEl = minEl;
  r.cards = document.querySelectorAll('.model-card').length; const card = document.querySelector('.model-card'); if (card) { const pane = card.querySelector('.fit-pane:not([hidden])'); const t = card.querySelector('h3 a').getBoundingClientRect(); r.firstCardNameTop = Math.round(t.top + scrollY); if (pane) r.firstCardVerdictBottom = Math.round(pane.querySelector('.fit-top').getBoundingClientRect().bottom + scrollY); }
  const h = document.getElementById('ans-h'); if (h) r.headlineBottom = Math.round(h.getBoundingClientRect().bottom + scrollY);
  r.bodyText = document.body.innerText.replace(/[ \t]+/g, ' ').replace(/\n\s*\n+/g, '\n').trim();
  r.nativeControls = [...document.querySelectorAll('button, select, input[type=text], input[type=number], input[type=checkbox], input[type=radio]')].filter(vis).filter(el => !el.closest('.sr-only')).map(el => { const cs = getComputedStyle(el); return el.tagName + (el.type ? ':' + el.type : '') + '#' + (el.id || el.name || '') + ' appearance=' + cs.appearance; });
  return r;
}
