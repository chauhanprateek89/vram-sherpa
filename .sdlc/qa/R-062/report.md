# R-062 QA
Evidence: .sdlc/qa/R-062/output.txt (check logic copied from release.yml into a script, run against http://localhost:8000).
- /results?vram_gb=24&context_tokens=4096: 200 with summary-banner and Manual VRAM (24 GB): PASS
- /results with no query: 302 to / -> check fails as intended: PASS
- Landing page (200, markers absent) -> check fails: PASS
- Workflow step has no -f/-L, exact 200, YAML parses; other checks remain (host guard 400 verified locally): PASS
