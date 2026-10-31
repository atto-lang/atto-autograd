import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

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

test('synthetic clean two-repo workspace requires exact source and compiler pins', {skip: process.platform !== 'linux'}, (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'ores-exact-workspace-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const ar = join(dir, 'oreslang-autograd'), tr = join(dir, 'oreslang-tensor');
  for (const folder of [join(ar, 'scripts'), join(ar, 'src'), join(ar, 'examples'), join(tr, 'src')])
    mkdirSync(folder, { recursive: true });
  copyFileSync(join(root, 'scripts/verify-native-workspace.sh'), join(ar, 'scripts/verify-native-workspace.sh'));
  for (const name of ['test.sh', 'test-workspace.sh'])
    writeFileSync(join(ar, 'scripts', name), "#!/bin/sh\nprintf 'SYNTHETIC_GUEST_ONLY\\n'\n");
  writeFileSync(join(ar, 'src/tensor_training_vjp.ores'), '// synthetic autograd source only\n');
  writeFileSync(join(ar, 'examples/tensor_training_vjp_smoke.ores'), '// synthetic guest smoke only\n');
  const originalTensor = '// synthetic tensor source only\n';
  writeFileSync(join(tr, 'src/tensor.ores'), originalTensor);
  const compiler = join(dir, 'synthetic-runner.sh');
  writeFileSync(compiler, '#!/bin/sh\nexit 0\n');
  chmodSync(compiler, 0o700);
  function git(cwd, ...args) {
    const r = spawnSync('git', args, {cwd, encoding: 'utf8'});
    assert.equal(r.status, 0, r.stderr);
    return r.stdout.trim();
  }
  for (const cwd of [ar, tr]) {
    git(cwd, 'init', '-q');
    git(cwd, 'add', '.');
    git(cwd, '-c', 'user.email=local@example.invalid', '-c', 'user.name=Local Test',
      'commit', '-qm', 'synthetic fixture');
  }
  const sha256 = (p) => createHash('sha256').update(readFileSync(p)).digest('hex');
  const pins = {
    ORESLANG_BIN: compiler,
    EXPECTED_AUTOGRAD_HEAD_SHA: git(ar, 'rev-parse', 'HEAD'),
    EXPECTED_AUTOGRAD_TREE_SHA: git(ar, 'rev-parse', 'HEAD^{tree}'),
    EXPECTED_TENSOR_HEAD_SHA: git(tr, 'rev-parse', 'HEAD'),
    EXPECTED_TENSOR_TREE_SHA: git(tr, 'rev-parse', 'HEAD^{tree}'),
    EXPECTED_TRAINING_VJP_BLOB: git(ar, 'hash-object', 'src/tensor_training_vjp.ores'),
    EXPECTED_TRAINING_SMOKE_BLOB: git(ar, 'hash-object', 'examples/tensor_training_vjp_smoke.ores'),
    EXPECTED_TENSOR_SOURCE_BLOB: git(tr, 'hash-object', 'src/tensor.ores'),
    EXPECTED_COMPILER_SHA256: sha256(compiler),
  };
  function run() {
    return spawnSync('sh', ['scripts/verify-native-workspace.sh'], {
      cwd: ar, encoding: 'utf8', timeout: 10000,
      env: {PATH: process.env.PATH, ...pins}
    });
  }
  const clean = run();
  assert.equal(clean.status, 0, clean.stderr);
  assert.match(clean.stdout, /UNAUTHENTICATED_EXACT_WORKSPACE_GUEST_MATCH/);
  // The synthetic marker is explicitly not compiler provenance.
  writeFileSync(join(tr, 'src/tensor.ores'), '// mutated tensor source\n');
  const mutated = run();
  assert.equal(mutated.status, 78);
  assert.match(mutated.stderr, /dirty checkout|guest source blob mismatch/);
  writeFileSync(join(tr, 'src/tensor.ores'), originalTensor);
  writeFileSync(compiler, '#!/bin/sh\necho changed\n');
  const replaced = run();
  assert.equal(replaced.status, 78);
  assert.match(replaced.stderr, /compiler SHA256 mismatch/);
});
