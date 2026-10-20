# Autograd P0: scalar reverse-mode proof slice

## Implemented in `src/tape.ores`
- Bounded static topological tape with leaf/add/mul nodes.
- Full graph validation rejects unknown opcodes, non-earlier dependencies, sentinel violations, ragged metadata and invalid roots.
- Deterministic forward and reverse traversal, including repeated operands and fan-out gradient accumulation.
- Descriptors, roots and saved activations are checked before direct backward writes. Legacy direct APIs may partially write on arithmetic overflow; use scratch-backed checked wrappers for output publication.
- Central-difference gradient oracle exercised in `tests/smoke.ores`.

## Limitations / safety gates
- This is a **scalar reference differentiator**, not yet tensor autograd or an automatic tape recorder. It is intentionally implemented in pure `.ores`.
- `List<T>` backing references are not yet enforced as immutable/unique across all compiler backends. The caller must not modify the tape or saved activations between validation and computation, and must not alias mutable output with input.
- Tape admission rejects nonfinite leaf values; backward admission rejects nonfinite seed/saved activations. `src/forward_checked.ores` and `src/backward_checked.ores` add scratch-backed publication and direct alias guards, but compiler-enforced transitive exclusive leases and concurrent mutation safety remain pending.
- No claims about exact floating-point exceptional-value semantics, higher-order gradients, JVP, broadcasting, dynamic control flow, in-place version detection, or cross-backend execution.
- `backward_f64_into` now checks saved activations against the tape operation sequence. This is not cryptographic generation identity: a future graph-generation token and pinned activation lease must enforce lifetime and mutation isolation.

## Next milestones
1. Integrate `oreslang-tensor` logical shapes and VJP rules, especially sum-to-shape broadcasting.
2. Make immutable forward activations and tape generations compiler-enforced.
3. Build an eager recording API and gradient checkpointing only after ownership-safe saved tensors.
4. Add finite differences, analytic fixtures, property fuzzing and genuine backend execution evidence.

## Differentiable dataset row selection (stacked P1, October 18, 2026)

`src/row_gather_vjp.ores` adds `row_gather_vjp_with_saved_f64_checked_into(original, indices, saved_forward, upstream, forward_scratch, gradient_scratch, gradient_output)` as an Oreslang-native adapter to [oreslang-tensor row gather PR #5](https://github.com/ores-truffle-oreslang/oreslang-tensor/pull/5).

The adapter:
- checks saved and upstream tensor ranks/extents and separate mutable scratch/output storage;
- recomputes gather into a caller-supplied scratch buffer;
- rejects any saved-forward element that does not match recomputation;
- calls the tensor-side `gather_rows_vjp_f64_checked_into` for deterministic scatter-add of repeated sampled rows;
- preserves the externally supplied gradient output on all rejected inputs (scratch may be changed).

This is a **value-consistency** check only. It cannot prove that an identical tensor buffer is the same generation/lease, that two mutable `List<float>` references are transitively nonaliased, or that a completed GPU event still pins storage. Those are mandatory compiler/runtime requirements before the stage can participate in real versioned-parameter actor training.

Sampling indices and dataset I/O are **not differentiable**; only the values selected from a differentiable source tensor carry the gather VJP. This is a custom tensor-level VJP adapter, not an eager recording engine or tensor-opcode addition to the existing scalar `TapeF64` runtime.

`tests/row_gather_vjp_smoke.ores` is the real guest acceptance target. `scripts/test-workspace.sh` now requires sibling `oreslang-tensor/src/row_batches.ores`, an explicitly configured genuine `ORESLANG_BIN`, and runs the new fixture. `tests/row_gather_vjp_contract.test.mjs` is a separate host-only numerical oracle; isolated V8 evaluation passed **5/5 cases (20 assertions)**. Neither the source nor its VJP has been verified by a pinned full Java/Graal or LLVM compiler in this draft.

Release only after compiled .ores guest output, unique/frozen tensor storage, dynamic gradient tape integration, finite-difference oracles, actor-state RW/lease fences, bounded reverse-graph fan-in, LLVM sanitizer checks, and checkpoint/restart fault injection.
