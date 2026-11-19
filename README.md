> **AI agent instructions (organization-wide):** Before working on code, issues, or pull requests, read and follow the [organization `AGENTS.md`](https://github.com/ores-truffle-oreslang/.github/blob/main/AGENTS.md), including its **merge-first stealth-development policy**. Honor any repository-local `AGENTS.md`, explicit holds, and GitHub-enforced rules.

# Oreslang Autograd

## Oreslang → AttoLang project migration (2026-11-18)

The public language name is **AttoLang** (formerly **Oreslang**). This is the legacy `ores-truffle-oreslang/oreslang-autograd` repository. New corresponding AttoLang work belongs in **[atto-lang/atto-autograd](https://github.com/atto-lang/atto-autograd)**. Do not infer identical Git histories, issues, PRs, execution evidence or completed ownership transfer from matching project names. Retain old checkout paths and commands below as historical reference.

The **canonical language specification** is [atto-lang/atto-spec](https://github.com/atto-lang/atto-spec), not an older Oreslang README or historical specification copy. [atto-platform-conformance](https://github.com/atto-lang/atto-platform-conformance) owns portable executed guest tests, [atto-compilers-parity-testing](https://github.com/atto-lang/atto-compilers-parity-testing) owns differential checks, and [atto-common-source](https://github.com/atto-lang/atto-common-source) owns portable `.atto` libraries. Current design targets include `snake_case` names, `borrow!`/`take!`/`copy!`/`copy!!`/`share!`, [implicit borrowing](https://github.com/atto-lang/atto-spec/blob/main/language/implicit-argument-borrow-v1.md), [file/module constants including one-time non-throwing `os.read_env`](https://github.com/atto-lang/atto-spec/blob/main/language/file-module-const-v1.md), and [`<<<`/`>>>`/`ashr` shifts distinct from stream `<<`/`>>`](https://github.com/atto-lang/atto-spec/blob/main/language/bitwise-and-stream-operators-v1.md). These are normative/proposed targets; unsupported syntax must fail closed.

Preserve legacy `.ores` fixtures, historical CLI/package names and ABI/wire contracts until replacements are verified. Do not use Python for new AttoLang scripts/tests/CI; prefer Rust for support tooling. Report exact SHA/tree, toolchain, positive/negative compiled guest runs and explicit failed/unsupported/not-executed states; host-only tests and zero executed Actions steps do not prove guest parity.

Pure-Oreslang automatic differentiation, initially scalar reverse-mode primitives and then tensor reverse-mode with a verified topological tape. This is not a PyTorch wrapper.

## P0 acceptance
- Independent analytic / finite-difference / differential gradient fixtures.
- Deterministic gradient accumulation and explicit saved-value lifetime contracts.
- No mutation of saved-forward intermediates without ownership/version proof.

See `docs/CONTRACT.md` after bootstrap PR.

## Engineering policy: no Python

Python is prohibited in this repository and throughout the `ores-truffle-oreslang` organization for scripts, tests, CI helpers, generators, analyzers, automation, and host tooling. Existing Python functionality must be migrated to **Rust** without weakening behavior, fixtures, or test coverage. The autograd implementation remains Oreslang; this rule specifically replaces Python tooling/reference code with Rust.
