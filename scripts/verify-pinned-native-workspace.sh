#!/bin/sh
# Source-local acceptance entrypoint. Executes actual guest scripts but cannot
# authenticate that a reviewed compiler source produced the supplied binary.
set -eu
unconfigured() { printf 'UNCONFIGURED: %s\n' "$1" >&2; exit 78; }
valid_sha() {
  case "$1" in *[!0-9a-f]*|'') return 1 ;; esac
  [ "${#1}" -eq "$2" ] || return 1
  # Zero hashes are missing provenance, not valid evidence.
  [ -n "$(printf '%s' "$1" | tr -d 0)" ]
}
: "${ORESLANG_BIN:=}"
: "${ORESLANG_RUNNER_SHA256:=}"
: "${ORESLANG_TENSOR_EXPECTED_SHA:=}"
: "${ORESLANG_AUTOGRAD_EXPECTED_SHA:=}"
: "${ORESLANG_COMPILER_SOURCE_SHA:=}"
: "${ORESLANG_BACKEND:=}"
: "${ORESLANG_TENSOR_TREE_SHA:=}"
: "${ORESLANG_AUTOGRAD_TREE_SHA:=}"
: "${ORESLANG_TENSOR_SOURCE_BLOB:=}"
: "${ORESLANG_TRAINING_VJP_BLOB:=}"
: "${ORESLANG_TRAINING_SMOKE_BLOB:=}"
case "$ORESLANG_BACKEND" in java|llvm) ;; *) unconfigured 'ORESLANG_BACKEND must be java or llvm' ;; esac
valid_sha "$ORESLANG_TENSOR_EXPECTED_SHA" 40 || unconfigured 'invalid Tensor commit pin'
valid_sha "$ORESLANG_AUTOGRAD_EXPECTED_SHA" 40 || unconfigured 'invalid Autograd commit pin'
valid_sha "$ORESLANG_COMPILER_SOURCE_SHA" 40 || unconfigured 'invalid declared compiler source pin'
valid_sha "$ORESLANG_RUNNER_SHA256" 64 || unconfigured 'invalid compiler executable SHA-256 pin'
valid_sha "$ORESLANG_TENSOR_TREE_SHA" 40 || unconfigured 'invalid Tensor tree pin'
valid_sha "$ORESLANG_AUTOGRAD_TREE_SHA" 40 || unconfigured 'invalid Autograd tree pin'
valid_sha "$ORESLANG_TENSOR_SOURCE_BLOB" 40 || unconfigured 'invalid Tensor source blob pin'
valid_sha "$ORESLANG_TRAINING_VJP_BLOB" 40 || unconfigured 'invalid training VJP blob pin'
valid_sha "$ORESLANG_TRAINING_SMOKE_BLOB" 40 || unconfigured 'invalid training smoke blob pin'
case "$ORESLANG_BIN" in /*) ;; *) unconfigured 'ORESLANG_BIN must be an absolute path' ;; esac
[ -f "$ORESLANG_BIN" ] && [ -x "$ORESLANG_BIN" ] || unconfigured 'compiler executable unavailable'
command -v git >/dev/null 2>&1 || unconfigured 'git unavailable'
command -v timeout >/dev/null 2>&1 || unconfigured 'bounded timeout unavailable'
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P) || unconfigured 'script location unavailable'
autograd_root=$(CDPATH='' cd -- "$script_dir/.." && pwd -P) || unconfigured 'Autograd checkout unavailable'
tensor_root=$(CDPATH='' cd -- "$autograd_root/../oreslang-tensor" && pwd -P) || unconfigured 'sibling Tensor checkout unavailable'
# Re-run all checks after guest execution to reject persistent source/binary
# mutation, including changes made by an invoked runner.
check_repo() {
  dir="$1"; expected="$2"; expected_tree="$3"; label="$4"; shift 4
  [ -d "$dir/.git" ] || unconfigured "$label is not an independent Git checkout"
  top=$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null) || unconfigured "$label checkout root missing"
  top=$(CDPATH='' cd -- "$top" && pwd -P) || unconfigured "$label checkout root invalid"
  [ "$top" = "$dir" ] || unconfigured "$label has foreign Git checkout root"
  actual=$(git -C "$dir" rev-parse --verify HEAD 2>/dev/null) || unconfigured "$label Git head unavailable"
  [ "$actual" = "$expected" ] || unconfigured "$label HEAD does not match expected SHA"
  actual_tree=$(git -C "$dir" rev-parse --verify 'HEAD^{tree}' 2>/dev/null) || unconfigured "$label Git tree unavailable"
  [ "$actual_tree" = "$expected_tree" ] || unconfigured "$label Git tree does not match expected SHA"
  status=$(git -C "$dir" status --porcelain --untracked-files=all 2>/dev/null) || unconfigured "$label Git status unavailable"
  [ -z "$status" ] || unconfigured "$label checkout contains modified or untracked files"
  for path do
    [ -f "$dir/$path" ] && [ ! -L "$dir/$path" ] || unconfigured "$label missing/linked tracked input: $path"
    git -C "$dir" ls-files --error-unmatch -- "$path" >/dev/null 2>&1 || unconfigured "$label untracked input: $path"
  done
}
check_blob() {
  file="$1"; expected="$2"
  [ -f "$file" ] && [ ! -L "$file" ] || unconfigured 'missing or linked source blob'
  actual=$(git hash-object "$file" 2>/dev/null) || unconfigured 'cannot hash source blob'
  [ "$actual" = "$expected" ] || unconfigured 'source blob identity mismatch'
}
check_compiler() {
  [ -f "$ORESLANG_BIN" ] && [ -x "$ORESLANG_BIN" ] && [ ! -L "$ORESLANG_BIN" ] ||
    unconfigured 'compiler missing, linked, or nonexecutable'
  if command -v sha256sum >/dev/null 2>&1; then
    actual_bin_hash=$(sha256sum "$ORESLANG_BIN") || unconfigured 'cannot hash compiler executable'
  elif command -v shasum >/dev/null 2>&1; then
    actual_bin_hash=$(shasum -a 256 "$ORESLANG_BIN") || unconfigured 'cannot hash compiler executable'
  else
    unconfigured 'SHA-256 tool unavailable'
  fi
  actual_bin_hash=${actual_bin_hash%% *}
  [ "$actual_bin_hash" = "$ORESLANG_RUNNER_SHA256" ] || unconfigured 'compiler executable hash mismatches pinned digest'
}
check_all() {
  check_repo "$tensor_root" "$ORESLANG_TENSOR_EXPECTED_SHA" "$ORESLANG_TENSOR_TREE_SHA" Tensor \
    src/tensor.ores src/broadcast.ores src/gradients.ores \
    src/matmul_vjp.ores src/matmul_both_vjp.ores src/row_batches.ores \
    tests/smoke.ores tests/row_batches_smoke.ores \
    tests/matmul_vjp_smoke.ores tests/matmul_two_sided_vjp_smoke.ores \
    scripts/test.sh
  check_repo "$autograd_root" "$ORESLANG_AUTOGRAD_EXPECTED_SHA" "$ORESLANG_AUTOGRAD_TREE_SHA" Autograd \
    src/tape.ores src/tensor_vjp.ores src/tensor_training_vjp.ores \
    src/row_gather_vjp.ores tests/smoke.ores tests/tensor_vjp_smoke.ores \
    tests/row_gather_vjp_smoke.ores examples/tensor_training_vjp_smoke.ores \
    scripts/test.sh scripts/test-workspace.sh scripts/verify-guest-case.sh
  check_blob "$tensor_root/src/tensor.ores" "$ORESLANG_TENSOR_SOURCE_BLOB"
  check_blob "$autograd_root/src/tensor_training_vjp.ores" "$ORESLANG_TRAINING_VJP_BLOB"
  check_blob "$autograd_root/examples/tensor_training_vjp_smoke.ores" "$ORESLANG_TRAINING_SMOKE_BLOB"
  check_compiler
}
check_all
case "${ORESLANG_GUEST_TIMEOUT_SEC:-120}" in
  1|5|10|30|60|120) timeout_sec="${ORESLANG_GUEST_TIMEOUT_SEC:-120}" ;;
  *) unconfigured 'invalid bounded guest timeout' ;;
esac
printf 'PINNED_WORKSPACE_START backend=%s tensor=%s autograd=%s declared_compiler_source=%s runner_sha256=%s\n' \
  "$ORESLANG_BACKEND" "$ORESLANG_TENSOR_EXPECTED_SHA" "$ORESLANG_AUTOGRAD_EXPECTED_SHA" \
  "$ORESLANG_COMPILER_SOURCE_SHA" "$ORESLANG_RUNNER_SHA256"
(cd "$tensor_root" && timeout -k 2s "${timeout_sec}s" sh scripts/test.sh) || { echo 'TENSOR_GUEST_SUITE_FAILED' >&2; exit 1; }
(cd "$autograd_root" && timeout -k 2s "${timeout_sec}s" sh scripts/test.sh && timeout -k 2s "${timeout_sec}s" sh scripts/test-workspace.sh) || { echo 'AUTOGRAD_GUEST_SUITE_FAILED' >&2; exit 1; }
check_all
echo 'PINNED_WORKSPACE_HOST_CHECKS_PASSED (unauthenticated guest output, independent compiler provenance required)'
