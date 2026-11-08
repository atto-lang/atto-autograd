# Oreslang Autograd

Pure-Oreslang automatic differentiation, initially scalar reverse-mode primitives and then tensor reverse-mode with a verified topological tape. This is not a PyTorch wrapper.

## P0 acceptance
- Independent analytic / finite-difference / differential gradient fixtures.
- Deterministic gradient accumulation and explicit saved-value lifetime contracts.
- No mutation of saved-forward intermediates without ownership/version proof.

See `docs/CONTRACT.md` after bootstrap PR.

## Engineering policy: no Python

Python is prohibited in this repository and throughout the `ores-truffle-oreslang` organization for scripts, tests, CI helpers, generators, analyzers, automation, and host tooling. Existing Python functionality must be migrated to **Rust** without weakening behavior, fixtures, or test coverage. The autograd implementation remains Oreslang; this rule specifically replaces Python tooling/reference code with Rust.

## Repository ownership and migration (2026-11-08)

**AttoLang** is the public language, formerly Oreslang. The repository remains `ores-truffle-oreslang/oreslang-autograd` in the legacy org until approved cutover. Preserve supported `.ores` sources, old import paths and package/CLI names; don't infer backend completeness from a README.

- **Belongs here:** automatic differentiation primitives and gradient correctness tests.
- **Excluded:** managed training cluster operations or unverifiable numerical parity.
- **Organization split:** [atto-lang](https://github.com/atto-lang) owns portable core/std, compilers, language libraries, test harnesses and demos; [atto-co](https://github.com/atto-co) independently owns cloud hosting, deployment, identity, billing and managed services. Existing `atto-*` repos are not automatically migrated copies or equivalent codebases.
- **Proposed first transfer (not executed):** retain original name, repo ID and Git history in `atto-lang/oreslang-autograd` via approved GitHub ownership transfer. No direct source-org → `atto-co` moves are authorized currently; future product carve-outs need separate signoffs.

**Maintainer safety procedure:** coordinate exact source head, owners and rollback in [issue #26](https://github.com/ores-truffle-oreslang/.github/issues/26), then follow private [runbook draft #25](https://github.com/ores-truffle-oreslang/.github/pull/25). Obtain source/target admin and independent security approval; inventory exact SHA/tree, numeric ID, target branch protections/rulesets, access, open PRs, releases/tags, packages, consumers, Actions/runners, secrets, GitHub Apps, webhooks and Pages. Don't clone/mirror private code as transfer workaround. After transfer verify ID/tree/history/controls, update links and rerun exact-head CI; **`steps=[]`/`null` is `NOT_EXECUTED`**, and host-only/source-free checks are not native guest/compiler evidence.
