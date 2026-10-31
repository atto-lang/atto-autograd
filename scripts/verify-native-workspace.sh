#!/bin/sh
# Source-local exact-head gate; NOT an attestation that the compiler is trusted.
# Never transfer private source across organizations to satisfy this check.
set -e

unconfigured() {
  printf 'NOT_EXECUTED: %s\n' "$1" >&2
  exit 78
}
valid_hex() {
  value="$1"
  width="$2"
  [ "${#value}" -eq "$width" ] || return 1
  case "$value" in ''|*[!0123456789abcdef]*) return 1;; esac
  [ -n "$(printf '%s' "$value" | tr -d 0)" ]
}
for program in git sha256sum cut tr printenv; do
  command -v "$program" >/dev/null 2>&1 || unconfigured "missing command: $program"
done
[ -n "$ORESLANG_BIN" ] || unconfigured 'ORESLANG_BIN must name a reviewed compiler'
case "$ORESLANG_BIN" in /*) ;; *) unconfigured 'compiler path must be absolute' ;; esac
[ -f "$ORESLANG_BIN" ] && [ -x "$ORESLANG_BIN" ] || unconfigured 'compiler not executable'

for key in EXPECTED_AUTOGRAD_HEAD_SHA EXPECTED_AUTOGRAD_TREE_SHA \
  EXPECTED_TENSOR_HEAD_SHA EXPECTED_TENSOR_TREE_SHA \
  EXPECTED_TRAINING_VJP_BLOB EXPECTED_TRAINING_SMOKE_BLOB \
  EXPECTED_TENSOR_SOURCE_BLOB EXPECTED_COMPILER_SHA256; do
  expected="$(printenv "$key" 2>/dev/null)" || unconfigured "missing required identity: $key"
  width=40
  [ "$key" = EXPECTED_COMPILER_SHA256 ] && width=64
  valid_hex "$expected" "$width" || unconfigured "invalid required identity: $key"
done

root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)" || unconfigured 'autograd root missing'
tensor="$(CDPATH='' cd -- "$root/../oreslang-tensor" && pwd -P)" || unconfigured 'pinned tensor sibling missing'

check_repo() {
  dir="$1"; head="$2"; tree="$3"
  top="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)" || unconfigured 'not a Git checkout'
  top="$(CDPATH='' cd -- "$top" && pwd -P)" || unconfigured 'invalid Git checkout root'
  [ "$top" = "$dir" ] || unconfigured 'nested or foreign checkout root'
  [ "$(git -C "$dir" rev-parse --verify HEAD 2>/dev/null)" = "$head" ] || unconfigured 'commit mismatch'
  [ "$(git -C "$dir" rev-parse --verify 'HEAD^{tree}' 2>/dev/null)" = "$tree" ] || unconfigured 'tree mismatch'
  [ -z "$(git -C "$dir" status --porcelain --untracked-files=all 2>/dev/null)" ] || unconfigured 'dirty checkout'
}
check_blob() {
  [ -f "$1" ] || unconfigured 'required guest source file missing'
  [ "$(git hash-object "$1" 2>/dev/null)" = "$2" ] || unconfigured 'guest source blob mismatch'
}
check_all() {
  check_repo "$root" "$EXPECTED_AUTOGRAD_HEAD_SHA" "$EXPECTED_AUTOGRAD_TREE_SHA"
  check_repo "$tensor" "$EXPECTED_TENSOR_HEAD_SHA" "$EXPECTED_TENSOR_TREE_SHA"
  check_blob "$root/src/tensor_training_vjp.ores" "$EXPECTED_TRAINING_VJP_BLOB"
  check_blob "$root/examples/tensor_training_vjp_smoke.ores" "$EXPECTED_TRAINING_SMOKE_BLOB"
  check_blob "$tensor/src/tensor.ores" "$EXPECTED_TENSOR_SOURCE_BLOB"
  binary_sha="$(sha256sum "$ORESLANG_BIN" | cut -d ' ' -f 1)" || unconfigured 'cannot hash compiler'
  [ "$binary_sha" = "$EXPECTED_COMPILER_SHA256" ] || unconfigured 'compiler SHA256 mismatch'
}
check_all
(cd "$root" && sh scripts/test.sh && sh scripts/test-workspace.sh) || {
  echo 'GUEST_FAILED: one or more native fixtures failed' >&2
  exit 1
}
# The runner may have mutated the compiler or source after passing preflight.
check_all
printf 'UNAUTHENTICATED_EXACT_WORKSPACE_GUEST_MATCH autograd=%s tensor=%s compiler_sha256=%s\n' \
  "$EXPECTED_AUTOGRAD_HEAD_SHA" "$EXPECTED_TENSOR_HEAD_SHA" "$EXPECTED_COMPILER_SHA256"
