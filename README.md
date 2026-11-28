> **AI agent instructions (organization-wide):** Before working on code, issues, or pull requests, read and follow the [organization `AGENTS.md`](https://github.com/atto-lang/.github/blob/main/AGENTS.md), including its **merge-first stealth-development policy**. Honor any repository-local `AGENTS.md`, explicit holds, and GitHub-enforced rules.

# atto-autograd

Automatic differentiation and computational graph infrastructure for AttoLang.

## Naming migration and language authority

This repository is maintained at [atto-lang/atto-autograd](https://github.com/atto-lang/atto-autograd); use `https://github.com/atto-lang/atto-autograd.git` for Git remotes and canonical `.atto` source examples. The normative language contract is [atto-spec](https://github.com/atto-lang/atto-spec), with executable guest evidence tracked by [atto-platform-conformance](https://github.com/atto-lang/atto-platform-conformance). Follow current ownership, borrowing, effects, actor, and syntax rules there; do not infer implemented features from a README or unexecuted CI.

## Scope

Differentiable computation graphs, gradient propagation, and numerical correctness tests.

## API and portability principles

- Specify numerical/semantic behavior before assuming support on every backend.
- Document input types, error behavior, shape and memory ownership requirements as applicable.
- Distinguish CPU execution, SIMD, GPU offload, and other target-specific capabilities.

## CPU tensor gradients

Check out `atto-tensor` next to this repository. `TensorVjp`, `RowGatherVjp`,
and `TensorTrainingVjp` expose checked gradients for broadcast add and
multiply, row gather, ReLU, mean squared error, and gradient accumulation.
They borrow inputs and use caller-owned scratch and output buffers. The
workspace guest suite runs the three tensor gradient fixtures with the
compiler selected by `ORESLANG_BIN`.

`TapeOps` provides a scalar forward and reverse pass. Its `stop_gradient`
operation (opcode `4`) copies the operand in the forward pass and stops
gradient propagation to that operand in the reverse pass.

## Verification

Use checked-in tests, documented tolerances, and exact-commit runtime evidence. Host-only or compilation tests are not equivalent to executed AttoLang guest tests. Record backend gaps explicitly.

## Development status

This describes the repository's intended purpose, not a verified list of implemented features. Refer to checked-in code and CI for current functionality and validated commands.

## Related

- [AttoLang](https://github.com/atto-lang)
- [ML conformance](https://github.com/atto-lang/atto-ml-conformance)
- [Language specification](https://github.com/atto-lang/atto-spec)

## Repository ownership and contribution routing

**Belongs here:** Differentiation graph representation, gradient propagation, tape lifetime and numerical gradient checks.

**Belongs elsewhere:** Base tensor shapes/layout kernels in atto-tensor; model-level orchestration in atto-ml; parity fixtures in atto-ml-conformance.

**Security and portability boundary:** Do not retain borrowed buffers past their owner; bound graph growth and verify gradients with finite-difference, non-differentiable and cancellation tests.

AttoLang is the canonical project name; new source, examples and documentation use `.atto` and the `atto-lang` repository organization. This README defines scope, not feature readiness.

**Acceptance:** name exact source SHA/tree, runtime/toolchain, positive and negative checks and execution receipt before claiming support. Host-only tests and empty CI steps do not prove guest execution.

## Organization boundaries and legacy compatibility

Autograd owns differentiation, not managed training-cluster operations. Independently validate analytic and finite-difference gradients, deterministic accumulation and saved-value lifetimes. New portable language libraries, compilers, tooling and conformance belong in AttoLang; managed cloud hosting, deployment, tenant identity and billing belong in separately governed services. Preserve supported legacy .ores source and historical package/CLI aliases until compiler-backed .atto replacements and consumers are validated. Similar repository names do not establish an identity-preserving transfer or source parity.

Any future repository transfer requires an approved owner and rollback plan, independently authorized source and destination, recorded numeric repository IDs and commit/tree, verified visibility, rulesets, access, releases, packages, consumers, webhooks, Apps, secrets and runner continuity. Never transfer private source without separate explicit authorization. Verify identity, history and controls after a transfer and rerun exact-head CI with executed steps; zero-step and host-only tests do not certify native guest execution.
