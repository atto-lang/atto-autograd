#!/usr/bin/env sh
# Source-only helper: this verifies guest exit/output, not compiler provenance.
set -eu
if [ "$#" -ne 2 ] || [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: supply executable ORESLANG_BIN, fixture, exact marker" >&2
  exit 78
fi
fixture="$1"
marker="$2"
tmpdir="$(mktemp -d)" || exit 78
trap 'rm -rf "$tmpdir"' EXIT
if ! "$ORESLANG_BIN" "$fixture" >"$tmpdir/stdout" 2>"$tmpdir/stderr"; then
  echo "GUEST_FAILED: $fixture" >&2
  cat "$tmpdir/stderr" >&2
  exit 1
fi
printf '%s\n' "$marker" >"$tmpdir/expected"
if ! diff -u "$tmpdir/expected" "$tmpdir/stdout"; then
  echo "GUEST_OUTPUT_MISMATCH: $fixture" >&2
  cat "$tmpdir/stderr" >&2
  exit 1
fi
if [ -s "$tmpdir/stderr" ]; then
  echo "GUEST_UNEXPECTED_STDERR: $fixture" >&2
  cat "$tmpdir/stderr" >&2
  exit 1
fi
printf 'GUEST_CASE_PASS: %s\n' "$fixture"
