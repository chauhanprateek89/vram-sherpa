import json, sys, difflib
from pathlib import Path
import numpy as np
from PIL import Image
REPO=Path('/home/pc/src/vram-sherpa'); OUT=REPO/'.sdlc/qa/R-067'
SC=Path('/tmp/claude-1000/-home-pc-src-vram-sherpa/7cf73e3c-3e97-4db6-9f20-8a3c53563c0e/scratchpad')
flt=sys.argv[1] if len(sys.argv)>1 else ''
met=[]
for f in SC.glob('metrics*.json'): met+=json.loads(f.read_text())
names=sorted({m['name'] for m in met if flt in m['name']})
rows=[]
for n in names:
  for w in (390,1280):
    for t in ('light','dark'):
      ref=REPO/f'docs/design/final/{n}-{w}-{t}.png'; live=OUT/f'{n}-{w}-{t}.png'
      if not live.exists(): rows.append((n,w,t,'MISSING')); continue
      a=Image.open(ref).convert('RGB'); b=Image.open(live).convert('RGB')
      h=min(a.height,b.height)
      d=np.abs(np.asarray(a.crop((0,0,a.width,h))).astype(int)-np.asarray(b.crop((0,0,b.width,h))).astype(int)).max(axis=2)
      frac=(d>40).mean()
      mt=(SC/'text'/f'mock-{n}-{w}-{t}.txt').read_text().splitlines(); lt=(SC/'text'/f'live-{n}-{w}-{t}.txt').read_text().splitlines()
      dl=[l for l in difflib.unified_diff(mt,lt,lineterm='',n=0) if l[:1] in '+-' and l[:3] not in ('+++','---')]
      lm=[m for m in met if m['name']==n and m['w']==w and m['theme']==t and m['kind']=='live'][0]
      rows.append((n,w,t,f'ref {a.width}x{a.height} live {b.width}x{b.height} dH={b.height-a.height} diff>40={frac*100:.2f}% textdiff={len(dl)} sw={lm.get("scrollWidth")} console={len(lm.get("console",[]))}',dl))
for r in rows:
  print(*r[:4])
  if len(r)>4 and r[4] and '-v' in sys.argv:
    for l in r[4][:12]: print('     ',l[:140])
