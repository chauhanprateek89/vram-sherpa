import json, os, sys, glob
from pathlib import Path
from urllib.parse import quote
from playwright.sync_api import sync_playwright
REPO = Path('/home/pc/src/vram-sherpa')
OUT = REPO / '.sdlc/qa/R-067'
SC = Path('/tmp/claude-1000/-home-pc-src-vram-sherpa/7cf73e3c-3e97-4db6-9f20-8a3c53563c0e/scratchpad')
(SC/'text').mkdir(exist_ok=True); (SC/'mock').mkdir(exist_ok=True)
MAIN, UNS, NOS = 'http://localhost:8000', 'http://localhost:8101', 'http://localhost:8102'
REF = '/results?vram_gb=24&context_tokens=4096'
STATES = {
 'home-default': (MAIN, '/', None),
 'results-gpu': (MAIN, '/results?gpu_id=gpu_nvidia_rtx_4090_laptop_16gb&context_tokens=4096', None),
 'results-manual-vram': (MAIN, REF, None),
 'results-gpu-ambiguous': (MAIN, '/results?gpu_search=RTX', None),
 'results-gpu-notfound': (MAIN, '/results?gpu_search=' + quote('RTX 3060 12GB'), None),
 'results-errors': (MAIN, '/results?vram_gb=abc', None),
 'results-chips': (MAIN, REF + '&family=qwen&quant_bucket=Q4', None),
 'results-filters-open': (MAIN, REF + '&family=qwen&quant_bucket=Q4', 'openFilters'),
 'results-picks-hidden': (MAIN, '/results?vram_gb=2&context_tokens=4096', None),
 'results-tight-gated': (MAIN, '/results?vram_gb=8&context_tokens=2048&family=llama&min_params_b=8&max_params_b=8', None),
 'gpu-suggest': (MAIN, '/', 'suggest'),
 'why-closed': (MAIN, REF, None),
 'why-loading': (MAIN, REF, 'whyLoading'),
 'why-loaded': (MAIN, REF, 'whyLoaded'),
 'empty-links': (MAIN, '/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8', None),
 'empty-inert': (MAIN, '/results?vram_gb=1&context_tokens=2048&quant_bucket=Q4', None),
 'empty-mixed': (MAIN, '/results?vram_gb=1&context_tokens=4096&quant_bucket=Q4', None),
 'unseeded': (UNS, '/', None),
 'detail-gated': (MAIN, '/models/model_llama_31_8b_instruct?vram_gb=24&context_tokens=2048', None),
 'detail-ungated': (MAIN, '/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096', None),
 'detail-no-sources': (NOS, '/models/model_qwen25_7b_instruct?vram_gb=24&context_tokens=4096', None),
 'detail-error': (MAIN, '/models/model_qwen25_7b_instruct?vram_gb=abc', None),
 'how-it-works': (MAIN, '/how-it-works', None),
}
METRICS = open(SC/'metrics.js').read()
flt = sys.argv[1] if len(sys.argv) > 1 else ''
kinds = ['mock', 'live']
exe = sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome')))[0]

def run(br, kind, name, w, theme, ncards=None):
    ctx = br.new_context(viewport={'width': w, 'height': 844 if w == 390 else 800}, device_scale_factor=1, color_scheme=theme, reduced_motion='reduce')
    page = ctx.new_page(); msgs = []
    page.on('console', lambda m: msgs.append(m.type + ': ' + m.text) if m.type in ('error', 'warning') else None)
    page.on('pageerror', lambda e: msgs.append('pageerror: ' + str(e)))
    setup = None
    if kind == 'mock':
        url = f'file://{REPO}/docs/design/mockups/final-{name}.html?theme={theme}'
    else:
        base, p, setup = STATES[name]; url = base + p
        ctx.add_init_script(f"try{{localStorage.setItem('vramsherpa-theme','{theme}')}}catch(e){{}}")
        if setup == 'whyLoading':
            page.route('**/results/why/**', lambda route: None)
    page.goto(url, wait_until='networkidle' if kind == 'live' else 'load')
    page.evaluate('document.fonts.ready'); page.wait_for_timeout(300)
    if kind == 'live':
        if setup == 'openFilters':
            page.locator('.filters > summary, details.filters summary').first.click()
        if setup == 'suggest':
            page.click('#gpu_search'); page.keyboard.type('4060', delay=40)
            try: page.wait_for_selector('#gpu-listbox li', state='visible', timeout=5000)
            except Exception as e: msgs.append('suggest: ' + str(e)[:80])
            if w == 1280 and theme == 'light': page.screenshot(path=str(OUT / 'gpu-suggest-typed-only-1280-light.png'))
            page.keyboard.press('ArrowDown'); page.wait_for_timeout(200)
        if setup in ('whyLoading', 'whyLoaded'):
            page.locator('.model-card').first.locator('details summary').first.click()
            if setup == 'whyLoaded':
                try: page.wait_for_function("/Total needed/.test(document.body.innerText)", timeout=8000)
                except Exception as e: msgs.append('why: ' + str(e)[:80])
            else: page.wait_for_timeout(400)
        page.wait_for_timeout(300)
    m = page.evaluate(METRICS)
    if kind == 'live' and ncards:
        page.evaluate('n => document.querySelectorAll(".model-card").forEach((c,i)=>{ if(i>=n) c.style.display="none" })', ncards)
        m['docHeightTrimmed'] = page.evaluate('document.documentElement.scrollHeight')
        m['docHeightFull'] = m['docHeight']; m['docHeight'] = m['docHeightTrimmed']
        m['bodyText'] = page.evaluate("document.body.innerText.replace(/[ \\t]+/g, ' ').replace(/\\n\\s*\\n+/g, '\\n').trim()")
    page.set_viewport_size({'width': w, 'height': m['docHeight']}); page.wait_for_timeout(250)
    out = (OUT / f'{name}-{w}-{theme}.png') if kind == 'live' else (SC/'mock'/f'{name}-{w}-{theme}.png')
    page.screenshot(path=str(out), full_page=True)
    (SC/'text'/f'{kind}-{name}-{w}-{theme}.txt').write_text(m.pop('bodyText'))
    ctx.close()
    return dict(kind=kind, name=name, w=w, theme=theme, console=msgs, **m)

res = []
mc = {}
with sync_playwright() as p:
    br = p.chromium.launch(executable_path=exe)
    for name in [n for n in STATES if flt in n]:
        for w in (390, 1280):
            for theme in ('light', 'dark'):
                for kind in kinds:
                    try:
                        r = run(br, kind, name, w, theme, mc.get((name,w,theme)))
                        print(kind, name, w, theme, 'sw', r['scrollWidth'], 'h', r['docHeight'], 'theme', r['dtheme'], 'console', len(r['console']), flush=True)
                    except Exception as e:
                        r = dict(kind=kind, name=name, w=w, theme=theme, error=str(e)[:200]); print('FAIL', kind, name, w, theme, str(e)[:150], flush=True)
                    res.append(r)
                    if kind == 'mock': mc[(name,w,theme)] = r.get('cards')
    br.close()
(SC/f'metrics{"-"+flt if flt else ""}.json').write_text(json.dumps(res, indent=1))
