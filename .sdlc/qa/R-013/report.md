# R-013 QA (R-084 and R-089 changes), tasks T-039, T-046, T-048
- PASS /results with params renders gist (200, summary-banner, 0 model cards, "71 models fit or are tight"); verdict=fits / fits&page=6 / wont-fit render card lists, same headline. Screens: gist-1440/1280/390.png, list-1440.png.
- PASS no-JS: form is GET /results, Check compatibility has no name; noscript Apply filters carries name=apply_view value=fits:10. apply_view=fits:10 -> 302 Location /results?...&verdict=fits&page_size=10; fits:5 -> verdict only; invalid -> no view; no hardware -> /; context_tokens=abc -> 422.
- PASS JS: banner data-verdict/page-size read; changing context keeps verdict=tight&page_size=10 in URL; Check compatibility submit drops them; hx-sync="#results-shell:replace". No console errors.
- Mockup gist-reference-1280-light matches after animation settles. pytest r084/r089/r083 tests pass.
