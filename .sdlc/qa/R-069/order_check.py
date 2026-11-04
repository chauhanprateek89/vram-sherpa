import glob,os
from playwright.sync_api import sync_playwright
exe=sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome')))[0]
J="""()=>{const t=s=>{const e=document.querySelector(s);return e?Math.round(e.getBoundingClientRect().top+scrollY):null};
const c=document.querySelector('[data-model-card]');let m=null;
if(c){const fp=[...c.querySelectorAll('.fit-pane')].find(e=>e.offsetParent);m=[...fp.querySelectorAll('*')].filter(e=>e.offsetParent&&!e.children.length&&e.textContent.trim()).map(e=>[e.className,e.textContent.trim().slice(0,30),Math.round(e.getBoundingClientRect().bottom+scrollY)]).slice(0,6)}
return {head:t('#ans-h'),verdicts:t('.verdicts'),picks:t('.picks'),listHead:t('#list-heading'),card:t('[data-model-card]'),m}}"""
with sync_playwright() as p:
    b=p.chromium.launch(executable_path=exe);pg=b.new_page(viewport={'width':1280,'height':800})
    for u in['/results?vram_gb=24&context_tokens=4096','/results?vram_gb=24&context_tokens=4096&verdict=fits','/results?vram_gb=abc&verdict=fits']:
        pg.goto('http://localhost:8000'+u);pg.wait_for_timeout(400);print(u,pg.evaluate(J))
