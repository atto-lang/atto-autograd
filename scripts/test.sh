#!/usr/bin/env sh
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: ORESLANG_BIN must point to the real Oreslang runner" >&2
  exit 78
fi
"$ORESLANG_BIN" tests/smoke.ores
