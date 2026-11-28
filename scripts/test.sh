#!/usr/bin/env sh
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: ORESLANG_BIN must be a real executable Oreslang runner" >&2
  exit 78
fi
sh scripts/verify-guest-case.sh tests/smoke.ores ORES_ML_AUTOGRAD_SMOKE_PASS
