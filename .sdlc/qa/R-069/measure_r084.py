"""R-069/R-084 layout limits measured on the live app (Chromium, 1280 and 390, light and dark)."""
import glob, os, json
from playwright.sync_api import sync_playwright
M='http://localhost:8000'; OUT='/home/pc/src/vram-sherpa/.sdlc/qa/R-069/gist'
REF='/results?vram_gb=24&context_tokens=4096'
MSG={'ambiguous':'/results?gpu_search=RTX','notfound':'/results?gpu_search=RTX%203060%2012GB','invalid':'/results?vram_gb=abc'}
S={'reference':REF,'all-three':'/results?vram_gb=8&context_tokens=2048','empty-wont':'/results?vram_gb=1','empty-inert':'/results?vram_gb=24&min_params_b=5&max_params_b=1',**MSG}
S.update({k+'+fits':v+('&' if '?' in v else '?')+'verdict=fits' for k,v in list(S.items()) if k in ['reference','ambiguous','notfound','invalid']})
J="""()=>{const b=e=>e?Math.round(e.getBoundingClientRect().bottom+scrollY):null;
const h=document.getElementById('ans-h');const v=document.querySelector('.verdicts');
const pk=[...document.querySelectorAll('.pick')];const c=document.querySelector('[data-model-card]');
let ft=null,name=null,cb=null,mar=null;
if(c){const fp=[...c.querySelectorAll('.fit-pane')].find(e=>!e.hidden&&e.offsetParent)||c.querySelector('.fit-pane');
 const f=fp.querySelector('.fit-top');ft=b(f);const kids=[...fp.querySelectorAll('*')].filter(e=>e.offsetParent&&e.children.length==0&&e.textContent.trim());
 mar=Math.max(...kids.filter(e=>e.getBoundingClientRect().top<f.getBoundingClientRect().bottom+30).map(b));
 name=b(c.querySelector('h3,h2,.model-id'));cb=b(c)}
return {head:b(h),verdicts:b(v),picks:pk.length?Math.max(...pk.map(b)):null,npicks:pk.length,card:!!c,name,fitTop:ft,cardBottom:cb,sw:document.documentElement.scrollWidth,vw:innerWidth,
 picksOpen:(document.querySelector('.picks')||{}).open}}"""
exe=sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome')))[0]
res={}
with sync_playwright() as p:
    br=p.chromium.launch(executable_path=exe)
    for w,vh in((1280,800),(390,844)):
        for th in('light','dark'):
            ctx=br.new_context(viewport={'width':w,'height':vh},color_scheme=th,reduced_motion='reduce')
            ctx.add_init_script(f"try{{localStorage.setItem('vramsherpa-theme','{th}')}}catch(e){{}}")
            pg=ctx.new_page(); errs=[]
            pg.on('console',lambda m:errs.append(m.text) if m.type=='error' else None)
            for n,u in S.items():
                pg.goto(M+u); pg.wait_for_timeout(500)
                r=pg.evaluate(J); r['errs']=errs[:]; errs.clear(); res[f'{n}|{w}|{th}']=r
                if th=='light' and n in('reference','reference+fits','ambiguous+fits','invalid','invalid+fits'):
                    pg.screenshot(path=f'{OUT}/{n}-{w}.png',full_page=False)
            ctx.close()
    br.close()
json.dump(res,open('/home/pc/src/vram-sherpa/.sdlc/qa/R-069/measure_r084.json','w'),indent=1)
for k,r in res.items(): print(k,{x:r[x] for x in('head','verdicts','picks','npicks','name','fitTop','cardBottom','sw','picksOpen')},r['errs'])
