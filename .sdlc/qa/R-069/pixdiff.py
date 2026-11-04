import glob
import os

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

exe = sorted(
    glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome"))
)[0]
R = "/results?vram_gb=24&context_tokens=4096"
C = {
    "reference": R,
    "list-fits": R + "&verdict=fits",
    "message-ambiguous": "/results?gpu_search=RTX",
    "message-notfound": "/results?gpu_search=RTX%203060%2012GB",
    "message-invalid": "/results?vram_gb=abc",
}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=exe)
    for w in (1280, 390):
        for th in ("light", "dark"):
            ctx = b.new_context(
                viewport={"width": w, "height": 900}, color_scheme=th, reduced_motion="reduce"
            )
            ctx.add_init_script(f"localStorage.setItem('vramsherpa-theme','{th}')")
            pg = ctx.new_page()
            for n, u in C.items():
                pg.goto("http://localhost:8000" + u)
                pg.wait_for_timeout(400)
                pg.add_style_tag(content="*{caret-color:transparent!important}")
                pg.screenshot(
                    path=f"/tmp/claude-1000/-home-pc-src-vram-sherpa/a6403559-a854-4d62-a35c-1b783f8576ef/scratchpad/{n}-{w}-{th}.png",
                    full_page=True,
                )
                a = Image.open(
                    f"/tmp/claude-1000/-home-pc-src-vram-sherpa/a6403559-a854-4d62-a35c-1b783f8576ef/scratchpad/{n}-{w}-{th}.png"
                ).convert("RGB")
                m = Image.open(f"docs/design/final/gist-{n}-{w}-{th}.png").convert("RGB")
                if a.size != m.size:
                    print(n, w, th, "size", a.size, m.size)
                h = min(a.size[1], m.size[1], 1100)
                x = np.abs(
                    np.asarray(a.crop((0, 0, w, h)), int) - np.asarray(m.crop((0, 0, w, h)), int)
                ).max(2)
                print(n, w, th, "diff>40: %.3f%%" % (100 * (x > 40).mean()))
