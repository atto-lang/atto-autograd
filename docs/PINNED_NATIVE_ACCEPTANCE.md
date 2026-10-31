# Pinned native Tensor + Autograd acceptance (NOT YET EXECUTED)

This procedure is for an **authorized local machine or source-organization
private runner**, not a cross-organization Git-source mirror. See the open
[source-exposure incident](https://github.com/ores-truffle-oreslang/oreslang-platform-conformance/issues/22).
The script makes exact Git-head/digest and fixture-output checks; it **cannot**
prove that an independently reviewed compiler produced its executable.

## Preconditions

1. Obtain independently reviewed full 40-character Git commit IDs for BOTH
   integration branches and for the compiler source. Do not accept a pin
   supplied only by the checkout being tested.
2. Clone the private repositories into sibling directories named
   `oreslang-tensor` and `oreslang-autograd`; check out those exact commits.
   Both repositories must be clean, independent Git checkouts.
3. Build or obtain an authorized, verified compiler/runner and independently
   record its executable SHA-256 digest. This entrypoint expects an executable
   accepting one fixture-file argument; wrappers are NOT independently trusted.
4. Run on a machine with `git`, a SHA-256 utility and GNU `timeout`.
   For Rust reference tests, also install a reviewed Rust toolchain.
   Never move private source to an unapproved runner destination.

From the Autograd checkout:

```sh
export ORESLANG_TENSOR_EXPECTED_SHA='REVIEWED_40_HEX_TENSOR_COMMIT'
export ORESLANG_AUTOGRAD_EXPECTED_SHA='REVIEWED_40_HEX_AUTOGRAD_COMMIT'
export ORESLANG_COMPILER_SOURCE_SHA='REVIEWED_40_HEX_COMPILER_COMMIT'
export ORESLANG_BIN='/absolute/path/to/oreslang'
export ORESLANG_RUNNER_SHA256='REVIEWED_64_HEX_BINARY_DIGEST'
export ORESLANG_BACKEND='java' # rerun separately with 'llvm'
export ORESLANG_GUEST_TIMEOUT_SEC='120'
sh scripts/verify-pinned-native-workspace.sh
```

Replace placeholders with reviewed lowercase hexadecimal values. The script
refuses invalid/mismatched pins, dirty/untracked source, missing tracked
fixtures, malformed backend labels, or missing timeout utilities. It invokes
Tensor smoke and both matmul VJPs, then Autograd scalar tape, tensor/row-gather
VJPs and training ReLU/MSE/fan-in fixtures. Underlying smoke harnesses require
**exact stdout**, empty stderr, and successful exit; runs have watchdogs.

The final line is
`PINNED_WORKSPACE_HOST_CHECKS_PASSED (requires independent compiler provenance review)`.
This is NOT an authenticated CI receipt or proof of cross-backend parity.
Record the compiler source tree SHA, toolchain version, command, fixture blob
digests, real runner/job/step IDs where applicable, outputs, negative
diagnostics, and reviewer approval. The backend variable is only a declared
label, not proof of which backend actually ran.

## Independent Rust reference checks

From Tensor, run `cargo test --all-targets` and compile/run
`tests/test_descriptor_reference.rs` with `rustc --edition=2021 -D warnings --test`.

From Autograd, run `cargo test --manifest-path rust-reference/Cargo.toml`.

Host-only `npm test` runs metadata and synthetic admission tests; fixtures may
intentionally use `/bin/true` or local synthetic repositories. **They must
never be reported as native Oreslang guest execution.**

## Strict promotion gate

Do not merge Tensor integration PR #17 or Autograd integration PR #12 until
independent real Java/GraalVM and LLVM exact-head receipts, executable Rust
regressions, reviewed compiler provenance, ownership/alias rejections, and
runner-admission recovery are established.
See [issue #963](https://github.com/ores-truffle-oreslang/oreslang-source.java/issues/963).
.NET, WASM and Forge have separate parity gates.
