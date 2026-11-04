import numpy as np
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

cases = {
    "all-three": "vram_gb=8&context_tokens=2048",
    "list-tight": "vram_gb=8&context_tokens=2048&verdict=tight",
    "list-fits": "vram_gb=8&context_tokens=2048&verdict=fits",
}
with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path="/home/pc/.cache/ms-playwright/chromium_headless_shell-1247/chrome-headless-shell-linux64/chrome-headless-shell"
    )
    for w in (1280, 390):
        for cs in ("light", "dark"):
            ctx = b.new_context(viewport={"width": w, "height": 900}, color_scheme=cs)
            pg = ctx.new_page()
            for n, q in cases.items():
                pg.goto("http://localhost:8000/results?" + q)
                pg.wait_for_load_state("networkidle")
                out = f".sdlc/qa/R-023/{n}-{w}-{cs}.png"
                pg.screenshot(path=out, full_page=True)
                ref = f"docs/design/final/gist-{n}-{w}-{cs}.png"
                try:
                    a = Image.open(out).convert("RGB")
                    r = Image.open(ref).convert("RGB")
                    print(n, w, cs, "app", a.size, "mock", r.size, end=" ")
                    h = min(a.height, r.height, 900)
                    d = (
                        np.asarray(
                            ImageChops.difference(a.crop((0, 0, w, h)), r.crop((0, 0, w, h)))
                        ).sum(2)
                        > 60
                    )
                    print("top900 diff%.2f%%" % (100 * d.mean()))
                except Exception as e:
                    print(n, w, cs, "no ref", e)
            ctx.close()
