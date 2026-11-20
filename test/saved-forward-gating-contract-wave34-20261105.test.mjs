// Contract admission only; does not execute the AttoLang autodiff implementation.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const c = JSON.parse(readFileSync(new URL('../integration/contract.v1.json', import.meta.url), 'utf8'));

test('scalar autograd is dependent on tensor, not an independent ownership authority',()=>{
  assert.deepEqual(c.provides, ['oreslang.autograd.scalar.v1']);
  assert.deepEqual(c.requires, ['oreslang.tensor.cpu.v1']);
  assert.deepEqual(c.upstreams, [
    'oreslang-common-source','oreslang-target-profiles','oreslang-tensor'
  ]);
  assert.ok(!c.upstreams.includes(c.repository));
  assert.deepEqual(c.fixtureIds, ['autograd/scalar-tape/v1']);
});

test('saved-forward leases require versioning and deterministic reverse accumulation',()=>{
  assert.match(c.nextGate, /ownership\/versioned saved-forward tape/i);
  assert.match(c.nextGate, /deterministic reverse topological accumulation/i);
  assert.match(c.nextGate, /guest gradients to finite differences/i);
});

test('host finite differences cannot promote themselves to certified Oreslang gradients',()=>{
  assert.equal(c.kind, 'PROPOSED_CONTRACT_NOT_IMPLEMENTATION');
  assert.equal(c.hostOracleRepo, 'oreslang-ml-conformance');
  assert.equal(c.hostOracleVerified, false);
  assert.equal(c.guestExecution, 'NOT_EXECUTED');
  assert.deepEqual(c.evidence, []);
  assert.match(c.mainSnapshotSha, /^[0-9a-f]{40}$/);
});
