# QA regression R-067, R-069, R-071, R-078 after T-033: PASS
Pixel diff vs docs/design/final (threshold 40), 4 screens x 1280/390 x light/dark = 16 renders: all 0.00%, same heights, no console errors (regress2/diff.txt, renders in regress2/).
Behaviour (regress2/beh.txt): filters, combobox (4 labels, roles, 48 px options, Esc, pick), error clears on refresh, Clear a#clear-gpu href="/" (52 px tall, 1px border, radius 16), no-JS path all as before. Labels Search GPU / Available VRAM (GB) unchanged. pytest all pass.
