# ADR-0007: The how-it-works worked example is computed by `estimation.py` at request time from a curated catalog model

Date: 2026-10-25 · Revised: 2026-10-25 (Gate 2 round 1) · Status: proposed · Requirements: R-075, R-037, R-009 to R-012, R-026, R-078

> **Revision, 2026-10-25 (Gate 2 round 1).** The first version computed the example from fixed test inputs (kv_gb_per_1k_ctx 0.18, giving 6.72 GB) and rejected a catalog lookup. At Gate 2 round 1 the user decided the example uses a real curated catalog 8B model at Q4 and 2048 tokens, so that the page agrees with that model's card (about 7.33 GB with catalog 2026-10-23). The decision below replaces the original one. The principle stays the same: no figure is typed into the template.

## Context

At Gate 1 (Q-6) the user approved adding plain-language terms and a worked example to `/how-it-works`, with the formulas unchanged and `ESTIMATION_POLICY_VERSION` kept at v0.1. Rule 16 reads: `estimation.py`, the page and the policy version change together when the estimate itself changes. A worked example typed into the template is a second copy of the formula's output, which would go wrong without anyone noticing the first time a constant changed.

The first version used R-009's test inputs (kv 0.18). No catalog model has that KV value: every 8B model in catalog 2026-10-23 has 0.48 GB per 1k tokens, so a visitor comparing the page with an 8B card saw 6.72 GB on one and 7.33 GB on the other. At Gate 2 round 1 the user chose to have the page agree with the card. R-075 names the model. This ADR does not, so it stays valid if product changes the choice.

Facts the design relies on:

- The catalog is rebuilt from `data/seed_*.json` on every start (rules 12 to 14). Curated models come from `tools/catalog_sources.yaml`, and the weekly refresh only adds discovered models (R-054, R-059), so a curated model leaves the catalog only when someone edits that file by hand.
- Variant ids are derived (`variant_<model_id>_<bucket>`, `tools/ingest_catalog.py`), and the ingest writes one variant per quant bucket.
- The seeder upserts and never deletes. A model dropped from the seed files stays in an existing production volume until the volume is recreated. In that case its card stays visible too, so the page and the card still agree.
- `how_it_works` already queries the database through `_base_context` (catalog version, last updated), so reading one more row adds no new failure mode.
- `variant_breakdown` and `model_detail` already compute from `Model` and `Variant` rows with `estimate_breakdown` and `estimate_variant`. The example should use the same inputs they use, so all three show the same figures.

## Decision

1. **Which row.** `main.py` declares the example once, next to `ESTIMATION_POLICY_VERSION`:
   - `WORKED_EXAMPLE_MODEL_ID`: the catalog model id R-075 names;
   - `WORKED_EXAMPLE_QUANT_BUCKET = "Q4"` and `WORKED_EXAMPLE_CONTEXT_TOKENS = 2048`;
   - `WORKED_EXAMPLE_FALLBACK`: the inputs to use when the row is absent (params_b, bits_effective and kv_gb_per_1k_ctx, today 8.0, 4.5 and 0.48, as R-075's provisional Q-12 answer states).
2. **Lookup.** `how_it_works` loads the model with `session.get(Model, WORKED_EXAMPLE_MODEL_ID)` (a primary-key read) and picks the variant in `model.variants` whose `quant_bucket` is `"Q4"`. If there is more than one, it takes the lowest id so the choice is deterministic. It looks the row up by model id and bucket, not by a hard-coded variant id, so the constant is the same id that `/models/{model_id}` and R-075 use.
3. **Computation.** It calls `estimate_breakdown(params_b=model.params_b, bits_effective=variant.bits_effective, kv_gb_per_1k_ctx=model.kv_gb_per_1k_ctx, context_tokens=2048, available_vram_gb=…)`, which is exactly the call `variant_breakdown` makes for that card. `estimate_breakdown` requires an available VRAM. R-075 lists only the four steps (weights, KV cache, overhead, required), so the route passes a declared constant that the template does not render. If the approved mockup adds a reserve or verdict step, that constant becomes visible, and R-075 must state its value.
4. **Template contract.** The route passes one `worked_example` object: `model_name` and `model_id` (both `None` in the fallback case), the four inputs, `context_tokens` and the `breakdown`. The template prints the inputs and formats each term to 2 decimals, as the breakdown does (R-020). When `model_id` is set, the template may link to `/models/{model_id}?context_tokens=2048` so the visitor can check the card. Whether to show that link is a choice for the Gate 2 mockup.
5. **Absent row.** If the model is missing, it has no Q4 variant, or the catalog is unseeded (R-026), the route builds the same object from `WORKED_EXAMPLE_FALLBACK`, still through `estimate_breakdown`, with `model_name` set to `None`. The template then labels the example "an 8B model at Q4", with no model name and no link. The page always returns 200, with every other section intact. It never returns a 500 and never shows the rejected 6.72 GB inputs. If the user answers Q-12 with "drop the section" instead, the only change is that the template skips the block when `model_name` is `None`.
6. **Guards (pytest, no network, rule 19).**
   - With the real seed files (`seeded_session`), the page shows the model's name, and its four figures equal `estimate_breakdown` for that model's Q4 variant at 2048 tokens. Its required figure also equals `estimate_variant(...).required_vram_gb`, the figure the card and model detail show.
   - The named model is in `data/seed_models.json` without `discovered: true`, and has a Q4 row in `data/seed_variants.json`. A hand edit that drops it from the curated list fails CI, which forces a deliberate choice of a new example model (new constant, new R-075 text).
   - With an unseeded database (`db_session`), the page returns 200 and shows the fallback figures 4.95, 0.98, 1.40 and 7.33 with no model name. These literals stay in the test on purpose: if a formula constant changes, this test fails, which is a reminder of rule 16.
7. **What does not change:** `estimation.py`, its constants and `ESTIMATION_POLICY_VERSION` (v0.1), and every route path and parameter (R-078).

## Alternatives considered

| Option | Why not |
| --- | --- |
| Hard-code the four figures in `how_it_works.html` | A second copy of the formula's output that goes wrong the first time a constant changes, and nothing would catch it. |
| Keep the fixed R-009 inputs (kv 0.18, 6.72 GB), the first version of this ADR | Rejected by the user at Gate 2 round 1: the page disagreed with every 8B card. |
| Hard-code a variant id (`variant_…_q4`) | That id is derived from the model id by the ingest, so it is a second identifier to keep in step. The model id is what R-075 names, and what the detail link uses. |
| Pick "any 8B model at Q4" with a query (for example the first `params_b == 8` row) | The model shown would depend on catalog ordering and could change on a refresh that adds a discovered 8B model. The user asked for a curated model, named in R-075. |
| Drop the worked example when the row is absent | Leaves a gap in a page whose sections R-075 fixes, and makes an unseeded or test database render a different page structure. The fallback keeps the same structure, without claiming to describe a named model. This is still open as Q-12. |
| Return 500 or 404 when the row is absent | An explanatory page should not break because one catalog row is missing. |
| Read the seed JSON files at request time instead of the database | Bypasses the catalog the rest of the app shows, which on a long-lived volume can differ (rows are upserted, never deleted), so the page and the card could disagree. |
| Compute in a Jinja filter or macro | The template layer would then import estimation code. The route is where the other pages already call it. |

## Consequences

- The example always equals the card for the same model, quantisation and context, because both come from the same row through the same function. If the catalog changes that model's params_b, bits_effective or kv_gb_per_1k_ctx, the example follows (R-075).
- If a formula constant changes, the example follows automatically, the fallback test fails loudly, and the policy version still has to be bumped by hand (rule 16).
- The route gains one primary-key read on a page that already queries the catalog, plus four constants. There is no new route, dependency or failure mode.
- Rule 12 holds: the row the example needs is recreated from the seed files on every start, and its absence has a defined result.
- If R-075 later names a different model, only `WORKED_EXAMPLE_MODEL_ID` and the expected name in the test change.
- UI: the worked example block on `/how-it-works` (R-075, `UI: yes`) and the optional link to the model detail page are drawn in the Gate 2 mockups.
