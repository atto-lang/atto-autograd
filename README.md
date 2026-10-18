# Oreslang Autograd

Pure-Oreslang automatic differentiation, initially scalar reverse-mode primitives and then tensor reverse-mode with a verified topological tape. This is not a PyTorch wrapper.

## P0 acceptance
- Independent analytic / finite-difference / differential gradient fixtures.
- Deterministic gradient accumulation and explicit saved-value lifetime contracts.
- No mutation of saved-forward intermediates without ownership/version proof.

See `docs/CONTRACT.md` after bootstrap PR.
