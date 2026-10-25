# ADR-0007: The how-it-works worked example is computed by `estimation.py` at request time

Date: 2026-10-25 · Status: proposed · Requirements: R-075, R-037, R-009 to R-012, R-078

## Context

At Gate 1 (Q-6) the user approved adding plain-language terms and a worked example to `/how-it-works`, with the formulas unchanged and `ESTIMATION_POLICY_VERSION` kept at v0.1. Rule 16 now reads: `estimation.py`, the page and the policy version change together when the estimate itself changes. R-075 fixes the example: an 8B model at Q4 with a 2048-token context, params_b 8, bits_effective 4.5 and kv_gb_per_1k_ctx 0.18 (the case `tests/test_estimation.py` already tests), giving weights 4.95 + KV cache 0.37 + overhead 1.40 = 6.72 GB. A worked example typed into the template is a second copy of the formula's output. If the constants ever change, the page would silently show wrong numbers, which rule 16 exists to prevent.

The example's KV value matches no catalog model. Every 8B model in catalog 2026-10-23 has 0.48 GB per 1k tokens, and `Llama 3.1 8B Instruct` Q4 at 2048 tokens shows 7.33 GB on its card. R-075 deals with this by stating the input values rather than naming a model.

## Decision

The `how_it_works` route calls `estimate_breakdown` with the R-075 inputs, declared once as a constant in `main.py` (params_b 8, bits_effective 4.5, kv_gb_per_1k_ctx 0.18, context_tokens 2048, plus an available VRAM only if the approved mockup shows a reserve or verdict step). It passes the result to the template, which formats each term to 2 decimals as the breakdown does (R-020) and prints the input values next to the steps. The example does not read the database (rule 12). A pytest checks that the page shows 4.95, 0.37, 1.40 and 6.72 and that they equal `estimate_breakdown` for those inputs. The formulas, constants and `ESTIMATION_POLICY_VERSION` do not change.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Hard-code the four figures in `how_it_works.html` | A second copy of the formula's output that drifts the first time a constant changes, and nothing would catch it. |
| Look up a real catalog variant (for example Llama 3.1 8B Instruct Q4) at request time | Ties the page to one catalog row, which a refresh can rename or drop, adds a query and an error path to a static page, and gives 7.33 GB rather than the 6.72 GB the user approved. |
| Compute in a Jinja filter or macro instead of the route | The template layer would then import estimation code. The route is where the other pages already call it. |

## Consequences

- The worked example can never disagree with the formula. If the constants change, the example's figures follow automatically, the pytest's expected values fail loudly, and the policy version still has to be bumped by hand (rule 16).
- The route gains a constant and a few lines, and the template a block of context values.
- Because the example's KV value (0.18) differs from the catalog's 8B models (0.48), the page must show its inputs, so a visitor comparing it with an 8B card is not misled. The Gate 2 mockup should make this visible.
