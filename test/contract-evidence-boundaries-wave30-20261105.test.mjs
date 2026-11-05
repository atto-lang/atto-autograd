import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const c = JSON.parse(readFileSync(new URL('../integration/contract.v1.json', import.meta.url), 'utf8'));

test('public API IDs are versioned namespaced interfaces, not filesystem paths', () => {
  for (const id of [...c.provides, ...c.requires]) {
    assert.match(id, /^oreslang\.[a-z][a-z0-9_.-]*\.v[1-9][0-9]*$/);
    assert.ok(!id.includes('..'));
    assert.ok(!id.includes('/'));
  }
});

test('fixture IDs are canonical relative, non-traversing case names', () => {
  for (const id of c.fixtureIds) {
    assert.match(id, /^[a-z0-9][a-z0-9._/-]*$/);
    assert.ok(!id.startsWith('/'));
    assert.ok(!id.endsWith('/'));
    assert.ok(!id.split('/').some(x => x === '' || x === '.' || x === '..'));
  }
});

test('host oracle does not grant guest or production certification authority', () => {
  assert.equal(c.hostOracleRepo, 'oreslang-ml-conformance');
  assert.notEqual(c.hostOracleRepo, c.repository);
  assert.equal(c.hostOracleVerified, false);
  assert.equal(c.guestExecution, 'NOT_EXECUTED');
  assert.deepEqual(c.evidence, []);
  assert.match(c.nextGate, /independently compare guest gradients/i);
});
