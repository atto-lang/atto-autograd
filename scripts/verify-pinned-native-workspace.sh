#!/bin/sh
# Source-local acceptance entrypoint. Executes actual guest scripts but cannot
# authenticate that a reviewed compiler source produced the supplied binary.
set -eu
unconfigured() { printf 'UNCONFIGURED: %s\n' "$1" >&2; exit 78; }
valid_sha() {
  case "$1" in *[!0-9a-f]*|'') return 1 ;; esac
  [ "${#1}" -eq "$2" ]
}
: "${ORESLANG_BIN:=}"
: "${ORESLANG_RUNNER_SHA256:=}"
: "${ORESLANG_TENSOR_EXPECTED_SHA:=}"
: "${ORESLANG_AUTOGRAD_EXPECTED_SHA:=}"
: "${ORESLANG_COMPILER_SOURCE_SHA:=}"
: "${ORESLANG_BACKEND:=}"
case "$ORESLANG_BACKEND" in java|llvm) ;; *) unconfigured 'ORESLANG_BACKEND must be java or llvm' ;; esac
valid_sha "$ORESLANG_TENSOR_EXPECTED_SHA" 40 || unconfigured 'invalid Tensor commit pin'
valid_sha "$ORESLANG_AUTOGRAD_EXPECTED_SHA" 40 || unconfigured 'invalid Autograd commit pin'
valid_sha "$ORESLANG_COMPILER_SOURCE_SHA" 40 || unconfigured 'invalid declared compiler source pin'
valid_sha "$ORESLANG_RUNNER_SHA256" 64 || unconfigured 'invalid compiler executable SHA-256 pin'
case "$ORESLANG_BIN" in /*) ;; *) unconfigured 'ORESLANG_BIN must be an absolute path' ;; esac
[ -f "$ORESLANG_BIN" ] && [ -x "$ORESLANG_BIN" ] || unconfigured 'compiler executable unavailable'
command -v git >/dev/null 2>&1 || unconfigured 'git unavailable'
command -v timeout >/dev/null 2>&1 || unconfigured 'bounded timeout unavailable'
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P) || unconfigured 'script location unavailable'
autograd_root=$(CDPATH='' cd -- "$script_dir/.." && pwd -P) || unconfigured 'Autograd checkout unavailable'
tensor_root=$(CDPATH='' cd -- "$autograd_root/../oreslang-tensor" && pwd -P) || unconfigured 'sibling Tensor checkout unavailable'
check_repo() {
  root="$1"; expected="$2"; label="$3"; shift 3
  [ -d "$root/.git" ] || unconfigured "$label is not an independent Git checkout"
  actual=$(git -C "$root" rev-parse HEAD 2>/dev/null) || unconfigured "$label Git head unavailable"
  [ "$actual" = "$expected" ] || unconfigured "$label HEAD does not match expected SHA"
  status=$(git -C "$root" status --porcelain --untracked-files=all 2>/dev/null) || unconfigured "$label Git status unavailable"
  [ -z "$status" ] || unconfigured "$label checkout contains modified or untracked files"
  for path do
    [ -f "$root/$path" ] || unconfigured "$label missing tracked input: $path"
    [ ! -L "$root/$path" ] || unconfigured "$label symlinked input forbidden: $path"
    git -C "$root" ls-files --error-unmatch -- "$path" >/dev/null 2>&1 || unconfigured "$label untracked input: $path"
    # `git status` can be suppressed by assume-unchanged / skip-worktree.
    # Compare working bytes to the exact commit tree, not only the index.
    expected_blob=$(git -C "$root" rev-parse "$expected:$path" 2>/dev/null) || unconfigured "$label input absent from pinned commit: $path"
    actual_blob=$(git -C "$root" hash-object -- "$root/$path" 2>/dev/null) || unconfigured "$label cannot hash input: $path"
    [ "$actual_blob" = "$expected_blob" ] || unconfigured "$label source blob mismatch: $path"
  done
}
check_repo "$tensor_root" "$ORESLANG_TENSOR_EXPECTED_SHA" Tensor \
  src/tensor.ores src/broadcast.ores src/gradients.ores \
  src/matmul_vjp.ores src/matmul_both_vjp.ores src/row_batches.ores \
  tests/smoke.ores tests/row_batches_smoke.ores \
  tests/matmul_vjp_smoke.ores tests/matmul_two_sided_vjp_smoke.ores \
  scripts/test.sh
check_repo "$autograd_root" "$ORESLANG_AUTOGRAD_EXPECTED_SHA" Autograd \
  src/tape.ores src/tensor_vjp.ores src/tensor_training_vjp.ores \
  src/row_gather_vjp.ores tests/smoke.ores tests/tensor_vjp_smoke.ores \
  tests/row_gather_vjp_smoke.ores examples/tensor_training_vjp_smoke.ores \
  scripts/test.sh scripts/test-workspace.sh scripts/verify-guest-case.sh
if command -v sha256sum >/dev/null 2>&1; then
  actual_bin_hash=$(sha256sum "$ORESLANG_BIN") || unconfigured 'cannot hash compiler executable'
  actual_bin_hash=${actual_bin_hash%% *}
elif command -v shasum >/dev/null 2>&1; then
  actual_bin_hash=$(shasum -a 256 "$ORESLANG_BIN") || unconfigured 'cannot hash compiler executable'
  actual_bin_hash=${actual_bin_hash%% *}
else
  unconfigured 'SHA-256 tool unavailable'
fi
[ "$actual_bin_hash" = "$ORESLANG_RUNNER_SHA256" ] || unconfigured 'compiler executable hash mismatches pinned digest'
case "${ORESLANG_GUEST_TIMEOUT_SEC:-120}" in
  1|5|10|30|60|120) ;; *) unconfigured 'invalid bounded guest timeout' ;;
esac
printf 'PINNED_WORKSPACE_START backend=%s tensor=%s autograd=%s declared_compiler_source=%s runner_sha256=%s\n' \
  "$ORESLANG_BACKEND" "$ORESLANG_TENSOR_EXPECTED_SHA" "$ORESLANG_AUTOGRAD_EXPECTED_SHA" \
  "$ORESLANG_COMPILER_SOURCE_SHA" "$actual_bin_hash"
(cd "$tensor_root" && timeout -k 2s 120s sh scripts/test.sh) || { echo 'TENSOR_GUEST_SUITE_FAILED' >&2; exit 1; }
(cd "$autograd_root" && timeout -k 2s 120s sh scripts/test.sh && timeout -k 2s 120s sh scripts/test-workspace.sh) || { echo 'AUTOGRAD_GUEST_SUITE_FAILED' >&2; exit 1; }
echo 'PINNED_WORKSPACE_HOST_CHECKS_PASSED (requires independent compiler provenance review)'
