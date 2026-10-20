# P0 implementation / v1 contract reconciliation

Review branch is based on P0 implementation source commit `a1d9c4385f016c6256e193eafce4d566484321dc` (PR #1) and adds proposed v1 interface metadata from `a7e074ad3dea86fc78453b959ec6011fffd44f50` (PR #3). All file transfers remain **inside the same Oreslang repository**.

**Actual implementation scope:** The P0 source has a bounded scalar tape with forward/backward functions. The proposed tensor interface dependency is NOT proven to be consumed. Saved-forward ownership/version proofs remain pending.

The P0 implementation's `docs/CONTRACT.md` remains canonical for **what code currently does**; `docs/PROPOSED_INTEGRATION_V1.md` and `integration/contract.v1.json` state **future provider/consumer acceptance targets**. The contract `requires` set is an intended interface, **not** a validated import/link edge or a passed Oreslang compiler test.

To validate: run the Node contract test (metadata only) and then run `scripts/test.sh` with a pinned, executable, actual Oreslang compiler binary. Capture exact source and compiler Git revisions, fixture IDs, executed guest result, stdout, stderr, environment, and negative-case results. Never report a source-only or independent host oracle as native compiler conformance.

Private Git source transfer to an alternate GitHub organization remains blocked pending reviewed incident remediation at [conformance #22](https://github.com/ores-truffle-oreslang/oreslang-platform-conformance/issues/22) and protected source-org approval. No releases or merges are authorized by the metadata test alone.
