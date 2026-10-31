import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
function probe(env) {
  return spawnSync('sh', ['scripts/verify-native-workspace.sh'], {
    cwd: root, encoding: 'utf8', timeout: 10000,
    env: { PATH: process.env.PATH, ...env }
  });
}
const pinned = Object.fromEntries([
  'EXPECTED_AUTOGRAD_HEAD_SHA',
  'EXPECTED_AUTOGRAD_TREE_SHA',
  'EXPECTED_TENSOR_HEAD_SHA',
  'EXPECTED_TENSOR_TREE_SHA',
  'EXPECTED_TRAINING_VJP_BLOB',
  'EXPECTED_TRAINING_SMOKE_BLOB',
  'EXPECTED_TENSOR_SOURCE_BLOB'
].map(key => [key, 'a'.repeat(40)]));
pinned.EXPECTED_COMPILER_SHA256 = 'b'.repeat(64);

test('native workspace gate refuses missing identities before running guest code', () => {
  const r = probe({ ORESLANG_BIN: '/bin/true' });
  assert.equal(r.status, 78);
  assert.match(r.stderr, /NOT_EXECUTED: missing required identity/);
  assert.equal(r.stdout, '');
});
test('native workspace gate rejects relative and nonexecutable compilers', () => {
  const relative = probe({ ...pinned, ORESLANG_BIN: './false-compiler' });
  assert.equal(relative.status, 78);
  assert.match(relative.stderr, /compiler path must be absolute/);
  const missing = probe({ ...pinned, ORESLANG_BIN: '/this/path/does/not/exist' });
  assert.equal(missing.status, 78);
  assert.match(missing.stderr, /compiler not executable/);
});
test('native workspace gate rejects forged and zero identity pins', () => {
  const bad = probe({ ...pinned, ORESLANG_BIN: '/bin/true',
    EXPECTED_AUTOGRAD_HEAD_SHA: 'f'.repeat(39) + 'z' });
  assert.equal(bad.status, 78);
  assert.match(bad.stderr, /invalid required identity: EXPECTED_AUTOGRAD_HEAD_SHA/);
  const zero = probe({ ...pinned, ORESLANG_BIN: '/bin/true',
    EXPECTED_COMPILER_SHA256: '0'.repeat(64) });
  assert.equal(zero.status, 78);
  assert.match(zero.stderr, /invalid required identity: EXPECTED_COMPILER_SHA256/);
});
