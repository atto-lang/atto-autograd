#!/usr/bin/env sh
set -eu
if [ -z "${ORESLANG_BIN:-}" ] || [ ! -x "$ORESLANG_BIN" ]; then
  echo "UNCONFIGURED: set ORESLANG_BIN to a real Oreslang compiler/runner" >&2
  exit 78
fi
if [ ! -f "../atto-tensor/src/tensor.ores" ] || [ ! -f "../atto-tensor/src/broadcast.ores" ]; then
  echo "UNCONFIGURED: atto-tensor source must be a pinned sibling checkout" >&2
  exit 78
fi
sh scripts/verify-guest-case.sh tests/tensor_vjp_smoke.ores ORES_ML_TENSOR_VJP_SMOKE_PASS
if [ ! -f "../atto-tensor/src/row_batches.ores" ]; then
  echo "UNCONFIGURED: atto-tensor row-batch feature must be a sibling" >&2
  exit 78
fi
sh scripts/verify-guest-case.sh tests/row_gather_vjp_smoke.ores ORES_AUTOGRAD_ROW_GATHER_VJP_PASS
sh scripts/verify-guest-case.sh examples/tensor_training_vjp_smoke.ores ORES_AUTOGRAD_TENSOR_TRAINING_VJP_PASS
