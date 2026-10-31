// Host-only source/provenance gate self-tests. Synthetic stubs are NOT guest results.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync, appendFileSync, symlinkSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const hostScript = fileURLToPath(new URL('../scripts/verify-pinned-native-workspace.sh', import.meta.url));
const baseEnv = { ...process.env };
function run(cmd, args, options = {}) {
  return spawnSync(cmd, args, { encoding: 'utf8', timeout: 10000, ...options });
}
function git(cwd, ...args) {
  const result = run('git', args, { cwd });
  assert.equal(result.status, 0, `git ${args.join(' ')}: ${result.stderr}`);
  return result.stdout.trim();
}
function makeWorkspace(t) {
  const root = mkdtempSync(join(tmpdir(), 'ores-pinned-harness-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const tensor = join(root, 'oreslang-tensor');
  const autograd = join(root, 'oreslang-autograd');
  const tree = {
    [tensor]: [
      'src/tensor.ores', 'src/broadcast.ores', 'src/gradients.ores',
      'src/matmul_vjp.ores', 'src/matmul_both_vjp.ores', 'src/row_batches.ores',
      'tests/smoke.ores', 'tests/row_batches_smoke.ores',
      'tests/matmul_vjp_smoke.ores', 'tests/matmul_two_sided_vjp_smoke.ores',
      'scripts/test.sh',
    ],
    [autograd]: [
      'src/tape.ores', 'src/tensor_vjp.ores', 'src/tensor_training_vjp.ores',
      'src/row_gather_vjp.ores', 'tests/smoke.ores', 'tests/tensor_vjp_smoke.ores',
      'tests/row_gather_vjp_smoke.ores', 'examples/tensor_training_vjp_smoke.ores',
      'scripts/test.sh', 'scripts/test-workspace.sh', 'scripts/verify-guest-case.sh',
      'scripts/verify-pinned-native-workspace.sh',
    ],
  };
  for (const [directory, paths] of Object.entries(tree)) {
    mkdirSync(directory, { recursive: true });
    git(directory, 'init', '-q');
    git(directory, 'config', 'user.email', 'synthetic@example.invalid');
    git(directory, 'config', 'user.name', 'Synthetic Harness');
    for (const path of paths) {
      const dest = join(directory, path);
      mkdirSync(join(dest, '..'), { recursive: true });
      writeFileSync(dest, '// synthetic source placeholder - not an Oreslang guest\n');
    }
    for (const path of ['scripts/test.sh', 'scripts/test-workspace.sh']) {
      if (!paths.includes(path)) continue;
      writeFileSync(join(directory, path), '#!/bin/sh\necho SYNTHETIC_FIXTURE_ONLY\n');
    }
    if (directory === autograd) {
      copyFileSync(hostScript, join(directory, 'scripts/verify-pinned-native-workspace.sh'));
    }
    git(directory, 'add', '.');
    git(directory, 'commit', '-qm', 'synthetic fixture for admission tests');
  }
  const env = {
    ...baseEnv,
    ORESLANG_BACKEND: 'java',
    ORESLANG_BIN: '/bin/true',
    ORESLANG_RUNNER_SHA256: run('sha256sum', ['/bin/true']).stdout.split(' ')[0],
    ORESLANG_TENSOR_EXPECTED_SHA: git(tensor, 'rev-parse', 'HEAD'),
    ORESLANG_AUTOGRAD_EXPECTED_SHA: git(autograd, 'rev-parse', 'HEAD'),
    ORESLANG_COMPILER_SOURCE_SHA: 'a'.repeat(40),
  };
  const script = join(autograd, 'scripts/verify-pinned-native-workspace.sh');
  const verify = (updates = {}) => run('sh', [script], { cwd: autograd, env: { ...env, ...updates } });
  return { root, tensor, autograd, verify, env };
}

test('pinned workspace fails closed before any guest on invalid configuration', (t) => {
  const { verify } = makeWorkspace(t);
  const good = verify();
  assert.equal(good.status, 0, good.stderr);
  assert.match(good.stdout, /PINNED_WORKSPACE_HOST_CHECKS_PASSED/);
  // A synthetic positive proves only this script's admission behavior.
  for (const updates of [
    { ORESLANG_BACKEND: 'fake' },
    { ORESLANG_BIN: 'relative/runner' },
    { ORESLANG_TENSOR_EXPECTED_SHA: '0'.repeat(40) },
    { ORESLANG_AUTOGRAD_EXPECTED_SHA: '0'.repeat(40) },
    { ORESLANG_COMPILER_SOURCE_SHA: 'not-a-commit' },
    { ORESLANG_RUNNER_SHA256: 'f'.repeat(64) },
    { ORESLANG_GUEST_TIMEOUT_SEC: '999' },
  ]) {
    const result = verify(updates);
    assert.equal(result.status, 78, `${JSON.stringify(updates)}: ${result.stderr}`);
    assert.match(result.stderr, /UNCONFIGURED/);
    assert.doesNotMatch(result.stdout, /PINNED_WORKSPACE_HOST_CHECKS_PASSED/);
  }
});

test('pinned workspace rejects modified or untracked source checkouts', (t) => {
  const { tensor, autograd, verify } = makeWorkspace(t);
  appendFileSync(join(tensor, 'src/tensor.ores'), '// edited\n');
  let result = verify();
  assert.equal(result.status, 78);
  assert.match(result.stderr, /Tensor checkout contains modified or untracked/);
  git(tensor, 'checkout', '--', 'src/tensor.ores');
  writeFileSync(join(autograd, 'src/untracked.ores'), '// fake\n');
  result = verify();
  assert.equal(result.status, 78);
  assert.match(result.stderr, /Autograd checkout contains modified or untracked/);
});

// Git may omit modified files from status when marked assume-unchanged.
test('pinned workspace detects hidden modified bytes despite Git index hints', (t) => {
  const { tensor, verify } = makeWorkspace(t);
  git(tensor, 'update-index', '--assume-unchanged', 'src/tensor.ores');
  appendFileSync(join(tensor, 'src/tensor.ores'), '// hidden edit\n');
  // The status heuristic alone is insufficient.
  assert.equal(git(tensor, 'status', '--porcelain'), '');
  const result = verify();
  assert.equal(result.status, 78, result.stderr);
  assert.match(result.stderr, /Tensor source blob mismatch: src\/tensor\.ores/);
});

test('pinned workspace rejects an exact-head tracked symlink input', (t) => {
  const { tensor, verify, env } = makeWorkspace(t);
  const target = join(tensor, 'src/tensor.ores');
  unlinkSync(target);
  symlinkSync('broadcast.ores', target);
  git(tensor, 'add', 'src/tensor.ores');
  git(tensor, 'commit', '-qm', 'symbolic link admission trap');
  const result = verify({ ORESLANG_TENSOR_EXPECTED_SHA: git(tensor, 'rev-parse', 'HEAD') });
  assert.equal(result.status, 78, result.stderr);
  assert.match(result.stderr, /Tensor symlinked input forbidden: src\/tensor\.ores/);
});
