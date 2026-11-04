import re
import urllib.parse
import urllib.request

B = "http://localhost:8000/results"


def get(q):
    return urllib.request.urlopen(B + "?" + q).read().decode()


def counts(h):
    d = {}
    for k, v in re.findall(r'data-verdict-control="([a-z-]+)".*?<b class="num">(\d+)</b>', h, re.S):
        d.setdefault(k, int(v))
    m = re.search(r'class="headline"[^>]*>(\d+) model', h)
    return d, (int(m.group(1)) if m else None)


def cards(h):
    return len(re.findall(r'<article class="model-card', h))


queries = [
    "vram_gb=8&context_tokens=2048",
    "vram_gb=24&context_tokens=4096",
    "vram_gb=1",
    "vram_gb=8&context_tokens=2048&family=llama",
    "vram_gb=12",
    "gpu_search=RTX%204090",
    "vram_gb=24&min_params_b=5&max_params_b=1",
    "vram_gb=6&context_tokens=8192&min_params_b=3",
]
bad = 0
for q in queries:
    h = get(q)
    base, N = counts(h)
    print(q, base, "headline", N, "cards on gist", cards(h))
    if base and N is not None and base.get("fits", 0) + base.get("tight", 0) != N:
        print("  FAIL sum")
        bad += 1
    for v in ("fits", "tight", "wont-fit"):
        for ps in (None, 5, 10, 25):
            tot = 0
            p = 1
            while True:
                qq = f"{q}&verdict={v}" + (f"&page_size={ps}" if ps else "") + f"&page={p}"
                hh = get(qq)
                c, _ = counts(hh)
                if c != base:
                    print("  FAIL counts changed", qq, c)
                    bad += 1
                n = cards(hh)
                tot += n
                m = re.findall(r"page=(\d+)", hh)
                mx = max([int(x) for x in m] + [1])
                if p >= mx or n == 0:
                    break
                p += 1
            if tot != base.get(v, 0):
                print("  FAIL total", v, ps, tot, "vs", base.get(v))
                bad += 1
    print("  ok-so-far, bad=", bad)
print("BAD", bad)
