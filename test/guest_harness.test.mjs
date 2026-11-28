// Synthetic harness self-tests. Do not treat these as native Oreslang execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { chmodSync, copyFileSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const execute = (cwd, script, executable) => spawnSync('sh', [script], {
  cwd, env: { ...process.env, ORESLANG_BIN: executable },
  encoding: 'utf8', timeout: 10000,
});
function fakeRunner(dir, contents) {
  const executable = join(dir, 'synthetic-runner.sh');
  writeFileSync(executable, contents);
  chmodSync(executable, 0o700);
  return executable;
}

test('silent success never passes the scalar autograd smoke gate', { skip: !existsSync('/bin/true') }, () => {
  const result = execute(root, 'scripts/test.sh', '/bin/true');
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /GUEST_OUTPUT_MISMATCH/);
});

test('scalar gate requires the exact stdout marker and empty stderr', (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'ores-autograd-scalar-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const valid = fakeRunner(dir, '#!/bin/sh\necho ORES_ML_AUTOGRAD_SMOKE_PASS\n');
  assert.equal(execute(root, 'scripts/test.sh', valid).status, 0);
  const noisy = fakeRunner(dir, '#!/bin/sh\necho ORES_ML_AUTOGRAD_SMOKE_PASS\necho warning >&2\n');
  const result = execute(root, 'scripts/test.sh', noisy);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /GUEST_UNEXPECTED_STDERR/);
});

test('tensor workspace smoke rejects silent success but accepts distinct synthetic markers', (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'ores-autograd-workspace-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const cwd = join(dir, 'oreslang-autograd');
  const scripts = join(cwd, 'scripts');
  const tensorSrc = join(dir, 'atto-tensor', 'src');
  mkdirSync(scripts, { recursive: true });
  mkdirSync(tensorSrc, { recursive: true });
  for (const name of ['test-workspace.sh', 'verify-guest-case.sh']) {
    copyFileSync(join(root, 'scripts', name), join(scripts, name));
  }
  for (const name of ['tensor', 'broadcast', 'row_batches']) {
    writeFileSync(join(tensorSrc, name + '.ores'), '// synthetic fixture presence only\n');
  }
  if (existsSync('/bin/true')) {
    const silent = execute(cwd, 'scripts/test-workspace.sh', '/bin/true');
    assert.notEqual(silent.status, 0);
    assert.match(silent.stderr, /GUEST_OUTPUT_MISMATCH/);
  }
  const runner = fakeRunner(dir, `#!/bin/sh
case "$1" in
  tests/tensor_vjp_smoke.ores) echo ORES_ML_TENSOR_VJP_SMOKE_PASS;;
  tests/row_gather_vjp_smoke.ores) echo ORES_AUTOGRAD_ROW_GATHER_VJP_PASS;;
  examples/tensor_training_vjp_smoke.ores) echo ORES_AUTOGRAD_TENSOR_TRAINING_VJP_PASS;;
  *) exit 23;;
esac
`);
  assert.equal(execute(cwd, 'scripts/test-workspace.sh', runner).status, 0);
});

test('hanging synthetic guest is terminated by the bounded watchdog', (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'ores-autograd-timeout-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const hanging = fakeRunner(dir, '#!/bin/sh\nsleep 15\n');
  const runCase = (limit) => spawnSync('sh', [
    'scripts/verify-guest-case.sh', 'tests/smoke.ores', 'ORES_ML_AUTOGRAD_SMOKE_PASS',
  ], {
    cwd: root,
    env: { ...process.env, ORESLANG_BIN: hanging, ORESLANG_GUEST_TIMEOUT_SEC: limit },
    encoding: 'utf8',
    timeout: 5000,
  });
  const expired = runCase('1');
  assert.notEqual(expired.status, 0);
  assert.match(expired.stderr, /GUEST_FAILED/);
  assert.equal(expired.error, undefined, 'watchdog must end before host test timeout');
  const invalid = runCase('999');
  assert.equal(invalid.status, 78);
  assert.match(invalid.stderr, /invalid bounded guest timeout/);
});
