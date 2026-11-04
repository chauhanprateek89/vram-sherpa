"""QA capture for R-067 G1-G13 (R-084 change). Live app vs docs/design/final/gist-*.png and mockup text."""
import glob, os, sys, json, difflib
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
REPO = Path('/home/pc/src/vram-sherpa'); OUT = REPO/'.sdlc/qa/R-067/gist'; OUT.mkdir(exist_ok=True)
SC = Path('/tmp/claude-1000/-home-pc-src-vram-sherpa/a6403559-a854-4d62-a35c-1b783f8576ef/scratchpad')
M, U = 'http://localhost:8000', 'http://localhost:8101'
B8 = '/results?vram_gb=8&context_tokens=2048'; REF = '/results?vram_gb=24&context_tokens=4096'
PAGES = {
 'reference': (M, REF), 'all-three': (M, B8),
 'list-fits': (M, REF+'&verdict=fits'), 'list-fits-middle': (M, REF+'&verdict=fits&page=6'),
 'list-fits-last': (M, REF+'&verdict=fits&page=15'), 'list-fits-size10': (M, REF+'&verdict=fits&page_size=10'),
 'list-tight': (M, B8+'&verdict=tight'), 'list-tight-short': (M, B8+'&family=llama&verdict=tight'),
 'list-tight-size10': (M, B8+'&verdict=tight&page_size=10'), 'list-wont-fit': (M, REF+'&verdict=wont-fit'),
 'list-wont-fit-page10': (M, '/results?vram_gb=1&verdict=wont-fit&page=10'),
 'verdict-zero': (M, REF+'&verdict=tight'),
 'message-ambiguous': (M, '/results?gpu_search=RTX'), 'message-notfound': (M, '/results?gpu_search=RTX%203060%2012GB'),
 'message-invalid': (M, '/results?vram_gb=abc'), 'empty-wont-link': (M, '/results?vram_gb=1'),
 'empty-all-inert': (M, '/results?vram_gb=24&min_params_b=5&max_params_b=1'), 'unseeded': (U, REF),
}
FITS6 = REF+'&verdict=fits&page=6'; FITS5 = REF+'&verdict=fits&page=5'
INTER = ['loading-from-gist','loading-next-page','failed-from-gist','failed-next-page','focus-verdict','focus-size','focus-page','focus-heading']
MODE = {'loading-from-gist':'tiles','failed-from-gist':'tiles','loading-next-page':'pager','failed-next-page':'pager','focus-page':'pager','focus-size':'top','focus-heading':'top'}
exe = sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome')))[0]
flt = sys.argv[1] if len(sys.argv) > 1 else ''
TXT = "document.body.innerText.replace(/[ \\t]+/g,' ').replace(/\\n\\s*\\n+/g,'\\n').trim()"
MET = "() => ({sw: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight, cards: document.querySelectorAll('[data-model-card],.model-card').length, c1: (()=>{const c=document.querySelector('.model-card .fit-pane:not([hidden]) .fit-top'); return c?Math.round(c.getBoundingClientRect().bottom+scrollY):null})(), head: (()=>{const e=document.getElementById('ans-h'); return e?Math.round(e.getBoundingClientRect().bottom+scrollY):null})(), picks:(()=>{const p=[...document.querySelectorAll('.pick')]; return p.length?Math.round(Math.max(...p.map(e=>e.getBoundingClientRect().bottom+scrollY))):null})()})"

def scroll_for(pg, mode, w, vh):
    if mode == 'tiles': y = (pg.evaluate("document.querySelector('.verdicts').getBoundingClientRect().top+scrollY") - 420) if w < 900 else 0
    elif mode == 'pager': y = pg.evaluate("document.querySelector('.pager').getBoundingClientRect().bottom+scrollY") - vh + 40
    else: y = pg.evaluate("document.querySelector('.list-pane').getBoundingClientRect().top+scrollY") - (110 if w >= 900 else 24)
    pg.evaluate(f"window.scrollTo(0,{max(0,y)})"); pg.wait_for_timeout(250)

def run(br, name, w, theme):
    vh = 844 if w == 390 else 800
    ctx = br.new_context(viewport={'width': w, 'height': vh}, device_scale_factor=1, color_scheme=theme, reduced_motion='reduce')
    ctx.add_init_script(f"try{{localStorage.setItem('vramsherpa-theme','{theme}')}}catch(e){{}}")
    pg = ctx.new_page(); msgs = []
    pg.on('console', lambda m: msgs.append(m.type+': '+m.text[:120]) if m.type in ('error','warning') else None)
    pg.on('pageerror', lambda e: msgs.append('pageerror: '+str(e)[:120]))
    mode = MODE.get(name, 'page'); info = {}
    if name in PAGES:
        base, p = PAGES[name]; pg.goto(base+p, wait_until='networkidle')
    else:
        start = {'loading-from-gist':REF,'failed-from-gist':REF,'loading-next-page':FITS6,'failed-next-page':FITS6,'focus-verdict':REF,'focus-size':FITS6,'focus-page':FITS6,'focus-heading':FITS5}[name]
        pg.goto(M+start, wait_until='networkidle')
    pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(300)
    if name in INTER:
        if name.startswith(('loading','failed')):
            hold = name.startswith('loading')
            def h(route):
                if hold: return  # never answered
                route.abort()
            pg.route('**/results/partials/content*', lambda r: h(r) if r.request.headers.get('hx-request') else r.continue_())
            if 'gist' in name: pg.locator('a.vt[href*="verdict=fits"]').first.click()
            else:
                pg.locator('nav[aria-label="Pages"] a', has_text='Next').first.click()
            pg.wait_for_timeout(700)
            info['status'] = pg.evaluate("[...document.querySelectorAll('#results-feedback p')].map(p=>[p.getAttribute('role'),p.hidden,p.innerText.trim()])")
            info['pending'] = pg.evaluate("document.querySelectorAll('.is-pending,[aria-busy=true]').length")
        elif name == 'focus-verdict':
            for _ in range(60):
                pg.keyboard.press('Tab')
                if pg.evaluate("(document.activeElement.getAttribute('href')||'').includes('verdict=fits') && document.activeElement.classList.contains('vt')"): break
        elif name == 'focus-size':
            for _ in range(80):
                pg.keyboard.press('Tab')
                if pg.evaluate("document.activeElement.closest('[aria-labelledby]') && document.activeElement.textContent.trim()=='10'"): break
        elif name == 'focus-page':
            for _ in range(120):
                pg.keyboard.press('Tab')
                if pg.evaluate("document.activeElement.getAttribute('aria-label')==='Page 7'"): break
        elif name == 'focus-heading':
            for _ in range(120):
                pg.keyboard.press('Tab')
                if pg.evaluate("document.activeElement.textContent.trim()==='Next' && !!document.activeElement.closest('nav')"): break
            pg.keyboard.press('Enter'); pg.wait_for_timeout(1500)
        info['active'] = pg.evaluate("(()=>{const e=document.activeElement;return e.tagName+'#'+e.id+' '+(e.getAttribute('aria-label')||e.textContent.trim().slice(0,20))+' outline='+getComputedStyle(e).outlineStyle+' '+getComputedStyle(e).outlineWidth})()")
        info['url'] = pg.url
    m = pg.evaluate(MET); text = pg.evaluate(TXT)
    if mode == 'page':
        pg.set_viewport_size({'width': w, 'height': m['h']}); pg.wait_for_timeout(200)
        pg.screenshot(path=str(OUT/f'{name}-{w}-{theme}.png'), full_page=True)
    else:
        if name not in ('focus-heading',) : scroll_for(pg, mode, w, vh)
        pg.screenshot(path=str(OUT/f'gist-{name}-{w}-{theme}.png') if False else str(OUT/f'{name}-{w}-{theme}.png'))
    # mock text
    mp = pg.context.new_page(); mp.goto(f'file://{REPO}/docs/design/mockups/final-gist-{name}.html?theme={theme}'); mp.wait_for_timeout(200)
    mtext = mp.evaluate(TXT)
    ctx.close()
    ref = Image.open(REPO/f'docs/design/final/gist-{name}-{w}-{theme}.png').convert('RGB'); live = Image.open(OUT/f'{name}-{w}-{theme}.png').convert('RGB')
    hh = min(ref.height, live.height)
    d = np.abs(np.asarray(ref.crop((0,0,ref.width,hh))).astype(int)-np.asarray(live.crop((0,0,live.width,hh))).astype(int)).max(axis=2)
    dl = [l for l in difflib.unified_diff(mtext.splitlines(), text.splitlines(), lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++','---')]
    return dict(name=name, w=w, theme=theme, ref=f'{ref.width}x{ref.height}', live=f'{live.width}x{live.height}', dH=live.height-ref.height, diff40=round(float((d>40).mean())*100, 2), textdiff=dl[:14], ndiff=len(dl), console=msgs, info=info, **m)

res = []
with sync_playwright() as p:
    br = p.chromium.launch(executable_path=exe)
    for name in [n for n in list(PAGES)+INTER if flt in n]:
        for w in (390, 1280):
            for theme in ('light', 'dark'):
                try: r = run(br, name, w, theme)
                except Exception as e: r = dict(name=name, w=w, theme=theme, error=str(e)[:300])
                res.append(r)
                print(r['name'], w, theme, r.get('error') or f"dH={r['dH']} diff40={r['diff40']}% textdiff={r['ndiff']} sw={r['sw']} cards={r['cards']} c1={r['c1']} console={len(r['console'])}", flush=True)
    br.close()
(SC/f'gist-res{"-"+flt if flt else ""}.json').write_text(json.dumps(res, indent=1))
