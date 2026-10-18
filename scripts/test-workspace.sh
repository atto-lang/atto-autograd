#!/usr/bin/env sh
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: set ORESLANG_BIN to a real Oreslang compiler/runner" >&2
  exit 78
fi
if [ ! -f "../oreslang-tensor/src/tensor.ores" ]; then
  echo "UNCONFIGURED: oreslang-tensor must be a sibling repository" >&2
  exit 78
fi
"$ORESLANG_BIN" tests/tensor_vjp_smoke.ores
