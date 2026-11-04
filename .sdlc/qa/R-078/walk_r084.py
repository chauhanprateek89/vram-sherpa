import json
import sys
import urllib.request

sys.path.insert(0, "/home/pc/src/vram-sherpa/tests")
from test_reference_queries import REFERENCE_QUERIES, extract_cards

fx = json.load(open("/home/pc/src/vram-sherpa/tests/fixtures/r078_reference.json"))
print("fixture", fx["catalog_version"], fx["captured_from"])


def get(u):
    return urllib.request.urlopen("http://localhost:8000" + u).read().decode()


ok = True
for q in REFERENCE_QUERIES:
    cards = []
    pages = 0
    for v in ("fits", "tight"):
        prev = None
        p = 1
        while True:
            c = extract_cards(get(f"{q}&verdict={v}&page_size=10&page={p}"))
            pages += 1
            if c == prev:
                break
            cards += c
            if len(c) < 10:
                break
            prev = c
            p += 1
    exp = fx["queries"][q]
    same = cards == exp
    ok &= same
    print(
        q,
        len(cards),
        len(exp),
        "pages",
        pages,
        "IDENTICAL" if same else "DIFF",
        sum(len(c["panes"]) for c in cards),
    )
    # gist has no cards
    g = get(q)
    print(" gist cards:", g.count("data-model-card"))
print("ALL OK" if ok else "FAIL")
