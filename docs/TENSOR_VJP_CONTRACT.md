# Tensor VJP P0.3: add and multiply with broadcasting

Status: **pure-Oreslang source committed; real compiler execution NOT verified**.

`src/tensor_vjp.ores` extends the scalar tape with explicit tensor-valued
vector-Jacobian products, consuming the shared `oreslang-tensor` CPU reference
kernels as a sibling workspace dependency.

For `z = left + right` and upstream gradient `g`:
- `dleft = sum_to_shape(g, left.shape)`
- `dright = sum_to_shape(g, right.shape)`

For `z = left * right`:
- `dleft = sum_to_shape(g * right, left.shape)`
- `dright = sum_to_shape(g * left, right.shape)`

Use right-aligned broadcasting including rank-zero scalars, singleton dimensions
and zero extents. Require the upstream shape to match the forward broadcast shape.
Operations produce one operand gradient at a time. They reject malformed shapes,
nonfinite values and **direct** buffer identity overlaps before publishing outputs.
The multiply VJP requires caller-provided weighted-product and reduction scratch
buffers; scratch content may be altered on failure, but caller output is not
published until the checked reduction succeeds.

## What is NOT proved yet

- `List<float>` may have aliases beyond the identity comparisons.
- Concurrent mutation, stale views, and saved-value generations require compiler
  borrow checking/leased immutable `Array<T>` buffers. A source-only alias check
  is NOT a safety proof.
- A fused multi-operand gradient commit is not atomic: left and right VJPs are
  separate operations, each with its own output publication.
- No eager runtime graph recording, higher-order differentiation, compiler
  parity, accelerator kernel, or LLVM AOT claim.
- `float` is the current numeric source type, not proven IEEE f64 on every
  backend; exceptional-value/tolerance policy must be specified per backend.

## Execution admission

The source-only `scripts/test.sh` remains independent of tensor packages.
The separate `scripts/test-workspace.sh` executes
`tests/tensor_vjp_smoke.ores`, requiring `oreslang-tensor` cloned alongside
`oreslang-autograd` and a real compiler provided in `ORESLANG_BIN`.
The conformance workspace pins each producer commit + Git tree and runs the new
suite as an additional distinct guest case. Do not promote source-free numerical
oracles to compiled code evidence.
