import glob, os, json
from pathlib import Path
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
REPO=Path('/home/pc/src/vram-sherpa'); OUT=REPO/'.sdlc/qa/R-067'
SC=Path('/tmp/claude-1000/-home-pc-src-vram-sherpa/7cf73e3c-3e97-4db6-9f20-8a3c53563c0e/scratchpad'); (SC/'focus').mkdir(exist_ok=True)
exe = sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome')))[0]
M='http://localhost:8000'
PAGES=[('/results?vram_gb=8&context_tokens=2048&family=llama&min_params_b=8&max_params_b=8','filters'),
       ('/results?vram_gb=24&context_tokens=4096&family=qwen&quant_bucket=Q4',None),
       ('/results?vram_gb=1&context_tokens=4096&quant_bucket=Q8',None),
       ('/models/model_llama_31_8b_instruct?vram_gb=24&context_tokens=2048',None),
       ('/how-it-works',None)]
RING="""() => { const e=document.activeElement; if(!e||e===document.body) return null;
 const ring = x => { const c=getComputedStyle(x); return {w:parseFloat(c.outlineWidth), s:c.outlineStyle, o:c.outlineOffset, c:c.outlineColor, bs:c.boxShadow!=='none'}; };
 let own=ring(e), via=null; 
 if(!(own.w>=3&&own.s!=='none')){ let p=e.parentElement; for(let i=0;i<4&&p;i++,p=p.parentElement){ const r=ring(p); if(r.w>=3&&r.s!=='none'){via=p.tagName+'.'+p.className; own=r; break;} } }
 const b=e.getBoundingClientRect();
 return {tag:e.tagName.toLowerCase()+(e.type?':'+e.type:''), id:e.id, cls:e.className&&e.className.baseVal===undefined?e.className:'', text:(e.innerText||e.value||e.getAttribute('aria-label')||'').trim().slice(0,30), ring:own, via, vis:b.width>0&&b.height>0, box:[b.x,b.y,b.width,b.height]} }"""
summary={}
with sync_playwright() as p:
    br=p.chromium.launch(executable_path=exe)
    for w in (390,1280):
      for theme in ('light','dark'):
        ctx=br.new_context(viewport={'width':w,'height':844 if w==390 else 800},color_scheme=theme,reduced_motion='reduce')
        ctx.add_init_script(f"try{{localStorage.setItem('vramsherpa-theme','{theme}')}}catch(e){{}}")
        pg=ctx.new_page(); stops=[]; kinds={}; clips=[]
        for url,setup in PAGES:
            pg.goto(M+url,wait_until='networkidle'); pg.wait_for_timeout(300)
            if setup=='filters':
                pg.locator('.filters > summary, details.filters summary').first.click(); pg.wait_for_timeout(200)
            pg.evaluate('window.scrollTo(0,0); document.activeElement && document.activeElement.blur()')
            seen=set()
            for i in range(400):
                pg.keyboard.press('Tab'); pg.wait_for_timeout(15)
                r=pg.evaluate(RING)
                if not r: break
                key=(r['tag'],r['id'],r['text'],round(r['box'][1]))
                if key in seen: break
                seen.add(key)
                r['page']=url.split('?')[0]; stops.append(r)
                k=r['tag']+'|'+(r['cls'] or '').split(' ')[0]+'|'+(r['via'] or '')
                if k not in kinds and r['vis']:
                    kinds[k]=1
                    pg.evaluate("e=>document.activeElement.scrollIntoView({block:'center'})",None); pg.wait_for_timeout(120)
                    b=pg.evaluate("()=>{const r=document.activeElement.getBoundingClientRect();return [r.x,r.y,r.width,r.height]}")
                    pad=14; x0=max(b[0]-pad,0); y0=max(b[1]-pad,0)
                    cl={'x':x0,'y':y0,'width':min(b[2]+2*pad,w-x0),'height':min(b[3]+2*pad,300)}
                    f=SC/'focus'/f'{w}-{theme}-{len(clips):02d}.png'; pg.screenshot(path=str(f),clip=cl); clips.append((f,k,r['text']))
        bad=[s for s in stops if not (s['ring']['w']>=3 and s['ring']['s']!='none')]
        invis=[s for s in stops if not s['vis']]
        summary[f'{w}-{theme}']={'stops':len(stops),'no_ring':[(s['page'],s['tag'],s['id'],s['text']) for s in bad],'invisible_stops':[(s['page'],s['tag'],s['id'],s['text']) for s in invis],
           'rings':sorted({(s['tag'],s['ring']['w'],s['ring']['s'],s['ring']['o'],s['ring']['c'],bool(s['via'])) for s in stops})}
        # contact sheet
        ims=[(Image.open(f).convert('RGB'),k,t) for f,k,t in clips]
        cols=3; cw=(w-20)//cols if w==390 else 410
        cols=1 if w==390 else 3
        cw=w-20 if w==390 else 410
        rows_h=[]; 
        sheet_items=[]
        for im,k,t in ims:
            if im.width>cw: im=im.resize((cw,int(im.height*cw/im.width)))
            sheet_items.append((im,k,t))
        col_h=[10]*cols; W=w if w==390 else 1280
        bg=(20,20,30) if theme=='dark' else (240,240,248)
        H=10+sum(i.height+30 for i,_,_ in sheet_items)//cols+400
        sheet=Image.new('RGB',(W,max(H,300)),bg); d=ImageDraw.Draw(sheet)
        col_h=[10]*cols
        for im,k,t in sheet_items:
            c=col_h.index(min(col_h)); x=10+c*(cw+10)
            d.text((x,col_h[c]),f'{k} "{t}"',fill=(120,120,140) if theme=='dark' else (60,60,80)); sheet.paste(im,(x,col_h[c]+14)); col_h[c]+=im.height+30
        sheet=sheet.crop((0,0,W,max(col_h)+10)); sheet.save(OUT/f'focus-states-{w}-{theme}.png')
        ctx.close()
    br.close()
(SC/'focus.json').write_text(json.dumps(summary,indent=1))
for k,v in summary.items(): print(k,'stops',v['stops'],'no_ring',len(v['no_ring']),v['no_ring'][:5],'invis',len(v['invisible_stops']))
print(json.dumps(summary['1280-light']['rings'],indent=0))
