# Autograd P0: scalar reverse-mode proof slice

## Implemented in `src/tape.ores`
- Bounded static topological tape with leaf/add/mul nodes.
- Full graph validation rejects unknown opcodes, non-earlier dependencies, sentinel violations, ragged metadata and invalid roots.
- Deterministic forward and reverse traversal, including repeated operands and fan-out gradient accumulation.
- Outputs are checked *before* any write. Caller supplies forward and gradient buffers.
- Central-difference gradient oracle exercised in `tests/smoke.ores`.

## Limitations / safety gates
- This is a **scalar reference differentiator**, not yet tensor autograd or an automatic tape recorder. It is intentionally implemented in pure `.ores`.
- `List<T>` backing references are not yet enforced as immutable/unique across all compiler backends. The caller must not modify the tape or saved activations between validation and computation, and must not alias mutable output with input.
- Tape admission rejects nonfinite leaf values; backward admission rejects nonfinite seed/saved activations. Intermediate arithmetic overflow and atomic result publication still need a transactional scratch/lease design.
- No claims about exact floating-point exceptional-value semantics, higher-order gradients, JVP, broadcasting, dynamic control flow, in-place version detection, or cross-backend execution.
- `backward_f64_into` trusts that `forward_values` were produced by the same tape generation. A future graph-generation identity and pinned activation lease must enforce this.

## Next milestones
1. Integrate `oreslang-tensor` logical shapes and VJP rules, especially sum-to-shape broadcasting.
2. Make immutable forward activations and tape generations compiler-enforced.
3. Build an eager recording API and gradient checkpointing only after ownership-safe saved tensors.
4. Add finite differences, analytic fixtures, property fuzzing and genuine backend execution evidence.
