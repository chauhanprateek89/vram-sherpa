# ADR-0011: One server-side rule decides whether `gpu_search` is a GPU, a VRAM number or search text

Date: 2026-10-28 · Status: proposed · Requirements: R-083 ("box number", "how the box decides"), R-003, R-001, R-002, R-004, R-005, R-008, R-071, R-078 · Decisions: Q-14, Q-15

## Context

R-083 puts one box on `/`: a visitor types a GPU or, for a GPU that isn't listed, a VRAM number such as `12 GB`. The box submits its text as `gpu_search` (R-013). Today `_resolve_available_vram` in `main.py` treats every `gpu_search` the same way: an exact match on a GPU's full label, `<vendor> <name>` or `<name>`; then a contains-match; then a GB hint (`_extract_vram_hint`). With catalog `2026-10-23` that sends `8GB` to the RTX 4060 Ti (8GB) and gives `12` and `24` the "matched multiple entries" message.

At Gate 1 (Q-14) the user decided the rule applies on both pages, including `Search GPU` on `/results`. R-003 now says: exact GPU first, then a *box number* is VRAM, then R-003's search and messages, and a page whose hardware came from a box number shows what `/results?vram_gb=<n>` shows, with the number in `Available VRAM (GB)` and `Search GPU` empty. A box number of 0 shows `/results?vram_gb=0` (Q-15). It must work without JavaScript, so the server owns it.

## Decision

1. **One module, pure functions.** A new module `src/vramsherpa/gpu_input.py` holds the rule and the helpers it shares with the results code. It imports nothing from `main.py` and needs no session, so it is unit-testable with plain objects that have `vendor`, `name` and `vram_gb` (ORM `GPU` rows qualify).
   - `gpu_option_label(gpu)` and `normalize_for_match(text)` move here from `main.py` (`_gpu_option_label`, `_normalize_for_match`); `main.py` imports them, so there is one definition of a label and of "case-insensitive, whitespace collapsed".
   - `exact_gpu_match(text, gpus) -> GPU | None`: step 1. Returns the GPU when exactly one GPU's full label, `<vendor> <name>` or `<name>` equals the normalised text, as `_resolve_available_vram` does today. `_resolve_available_vram` calls it in place of its inline loop.
   - `parse_box_number(text) -> str | None`: strips leading and trailing whitespace and returns the numeric part as typed (`12.5` for `12.5gb`, `0.0` for `0.0`) when the whole text matches `re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:\s*gb)?", text, re.IGNORECASE)`, otherwise `None`. The class is `[0-9]`, not `\d`, because Python's `\d` also matches digits from other scripts that `float()` would accept. This accepts exactly R-083's box numbers (`12`, `12 GB`, `12gb`, `12.5 GB`, `24GB`, `0`) and rejects its counter-examples (`12 G`, `12 GiB`, `12,5`, `.5`, `-4`, `12 GB VRAM`, `RTX 3060 12GB`).
   - `decide_box(text, gpus) -> BoxDecision`, a small frozen dataclass with `kind` (`"blank"`, `"gpu"`, `"vram"` or `"search"`), `gpu` and `vram_text`. In order: blank text is `blank`; an exact GPU is `gpu`; a box number is `vram`; anything else is `search`.
2. **Applied once per request, in `_results_page_context`,** which every results path goes through: the full page, the htmx refresh (`/results/partials/content`), `summary` and `list`. Before `vram_gb` is parsed: when `gpu_id` and `vram_gb` are each absent or blank and `gpu_search` is not blank (R-003's case), it loads the GPUs and calls `decide_box`. For `vram`, it rewrites the request to `vram_gb = decision.vram_text` and `gpu_search = None`, and the rest of the function runs unchanged. The page is then exactly the `/results?vram_gb=<n>` page: hardware `Manual VRAM (<n> GB)`, `<n>` in `Available VRAM (GB)`, `Search GPU` empty, no search message and no default note; for 0, R-008's inline error, 8 GB and the default note; and the filter chips, empty-state links and hardware query carry `vram_gb=<n>`. For `gpu`, `blank` and `search`, the text goes on to `_resolve_available_vram` as today.
3. **No contains-match on a box number anywhere.** If a box number reaches `_resolve_available_vram`'s search branch some other way (`gpu_id` unknown, or `vram_gb` present but invalid, which R-003 does not cover), that branch also calls `decide_box`. A box number above 0 is used as available VRAM with no message; a box number of 0 falls back to 8 GB with no message. Contains-match and the GB hint run only for `search` text (R-003: "only to text that is not a box number").
4. **The results page with JavaScript.** An in-place refresh does not re-render the form (`hx-target="#results-shell"`), so after typing `12` in `Search GPU` the fields would still show `12` in the search box and nothing in `Available VRAM (GB)`. The summary already carries `data-vram-invalid`, which `syncVramInvalid` in `app.js` reads after each swap. In the same way, when the hardware came from a box number the summary carries `data-box-vram="<n>"`, and after each swap `app.js` writes `<n>` into `#vram_gb` and empties `#gpu_search` and `#gpu_id`. The server decides, and the script copies the result into the form. The address htmx pushes (ADR-0008 fix 1) keeps the raw `gpu_search=12`, which reloads to the same page because the server applies the same rule.
5. **No-JavaScript path.** The landing form and the results form are plain GETs to `/results` without JavaScript, and the server applies the rule on every request, so the outcome is the same with and without JavaScript.

Guards (pytest, rule 19):

- `parse_box_number`: R-083's two lists, plus `"  12  "` (strips), `"12  GB"`, `"12Gb"`, `"0.0"`, `"１２"` (full-width digits, rejected), `""` and `"GB"` (rejected).
- `decide_box` order, with stub GPUs: an exact label wins over everything; a box number is `vram`; `RTX` is `search`; blank is `blank`.
- Routes, with catalog `2026-10-23`: `gpu_search=12%20GB` shows `Manual VRAM (12 GB)` and `value="12"` on `#vram_gb` and an empty `#gpu_search`; `12.5gb` shows `Manual VRAM (12.5 GB)`; `8GB` shows `Manual VRAM (8 GB)`, not the RTX 4060 Ti; `24` shows `Manual VRAM (24 GB)` and no "matched multiple entries"; `0`, `0 GB` and `0.0` render as `/results?vram_gb=0` does; `nvidia geforce rtx 4090 (24.0 gb)` and `GeForce RTX 4090` show `NVIDIA GeForce RTX 4090 (24.0 GB)`; `RTX`, `RTX 3060 12GB` and `Quadro XYZ` keep R-003's messages word for word.
- The same queries through `/results/partials/content` with `HX-Request: true` give the same summary, and the summary carries `data-box-vram` only for box numbers.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Decide in JavaScript on the landing page and submit `vram_gb` | Breaks the no-JavaScript path (R-083, R-013), and the results page's `Search GPU` would need its own copy of the rule. |
| Put the box-number check inside `_resolve_available_vram` only | It returns a VRAM figure, not a rewritten request, so the hardware label, the VRAM field, the filter chips and the empty-state links would each need a special case to look like `?vram_gb=<n>`. Rewriting the request once gets all of them for free. |
| Keep the helpers in `main.py` | Works, but `main.py` is 1500 lines and the rule would be tested through the app and the database. A pure module of about 60 lines makes the rule's one home obvious. |
| Widen the box number (`12 GiB`, `12,5`, `12 G`) | R-083 defines the set exactly; anything outside it goes through R-003's search, whose GB hint still catches `RTX 3060 12GB`. |
| A new `vram_text` query parameter for the box | Adds a parameter (R-078 keeps them), and the results page's `Search GPU` would still need the rule. |

## Consequences

- `8GB`, `12` and `24` typed into `Search GPU` on `/results` now mean VRAM, as the user chose (Q-14). Nothing else in R-003's messages changes.
- Existing precedence is kept: on `/results`, a non-blank `Available VRAM (GB)` still wins over `Search GPU` text, so typing `12` into `Search GPU` while `Available VRAM (GB)` holds `24` still computes for 24 GB, as typing any free text does today.
- `app.js` gains the `data-box-vram` copy-back (a few lines beside `syncVramInvalid`), which bumps `ASSET_VERSION` (rule 21).
- `ESTIMATION_POLICY_VERSION` does not change: the formula is untouched (rule 16).
- `UI: yes` only through R-003's visible outcomes on `/results`; no new screen.
