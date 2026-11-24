// Host-only numerical oracle and source guard check. NOT Oreslang guest execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const tape = readFileSync(new URL('../src/tape.ores', import.meta.url), 'utf8');
const smoke = readFileSync(new URL('../tests/smoke.ores', import.meta.url), 'utf8');

test('finite-forward overflow-backward fixture is a discriminating IEEE-754 case', () => {
  const large = 1.0e308;
  const small = 1.0e-308;
  const seed = 2.0;
  assert.equal(Number.isFinite(large), true);
  assert.equal(Number.isFinite(small), true);
  assert.equal(Number.isFinite(large * small), true);
  assert.equal(Number.isFinite(seed * large), false);
  assert.match(smoke, /backward_overflow/);
  assert.match(smoke, /arr\[1\.0e308, 1\.0e-308, 0\.0\]/);
  assert.match(smoke, /overflow_output\[2\], 89\.0/);
});

test('duplicated add edge must reject the second overflowing contribution', () => {
  const seed = 1.0e308;
  assert.equal(Number.isFinite(1.0 + 1.0), true);
  assert.equal(Number.isFinite(seed), true);
  assert.equal(Number.isFinite(seed + seed), false);
  assert.match(smoke, /duplicate_add_overflow/);
  assert.ok(smoke.includes('duplicate_published[0], 63.0'));
  assert.ok(smoke.includes('duplicate_published[1], 64.0'));
});

test('every reverse accumulator checks finiteness before storing', () => {
  const start = tape.indexOf('  for int i = n - 1; i >= 0; i--');
  assert.ok(start >= 0, 'reverse tape traversal missing');
  const loop = tape.slice(start);
  const branches = ['if op == 1; do', 'if op == 2; do', 'if op == 3; do'];
  const checks = [
    [branches[0], 'next_left', 'gradients[a]'],
    [branches[0], 'next_right', 'gradients[b]'],
    [branches[1], 'next_left', 'gradients[a]'],
    [branches[1], 'next_right', 'gradients[b]'],
    [branches[2], 'next_gradient', 'gradients[a]'],
  ];
  for (const [branch, variable, target] of checks) {
    const from = loop.indexOf(branch);
    assert.ok(from >= 0, branch);
    const next = branches.filter(b => b !== branch)
      .map(b => loop.indexOf(b, from + branch.length))
      .filter(n => n > from);
    const part = loop.slice(from, next.length ? Math.min(...next) : undefined);
    const computed = part.indexOf(`val float ${variable} =`);
    const guarded = part.indexOf(`if !finite_f64(${variable}); do`, computed);
    const stored = part.indexOf(`${target} = ${variable};`, guarded);
    assert.ok(computed >= 0 && guarded > computed && stored > guarded,
      `${branch} must reject nonfinite ${variable} before writing ${target}`);
  }
});

test('checked forward atomicity on overflow is a separate guest acceptance case', () => {
  // Host-only arithmetic oracle plus native fixture/source presence checks.
  assert.equal(Number.isFinite(1.0e308), true);
  assert.equal(Number.isFinite(2.0), true);
  assert.equal(Number.isFinite(1.0e308 * 2.0), false);
  assert.match(smoke, /val TapeF64 forward_overflow = new TapeF64/);
  assert.match(smoke, /TapeOps\.forward_f64_into\(\s*rt borrow forward_overflow, rt mut borrow raw_forward_scratch/);
  assert.match(smoke, /ForwardChecked\.forward_checked_into\(\s*rt borrow forward_overflow, rt mut borrow safe_forward_scratch, rt mut borrow forward_published/);
  for (const [index, sentinel] of [[0, 77], [1, 78], [2, 79]]) {
    assert.ok(smoke.includes(`forward_published[${index}],${sentinel}.0`));
  }
  const checked = readFileSync(new URL('../src/forward_checked.ores', import.meta.url), 'utf8');
  const call = checked.indexOf('if TapeOps.forward_f64_into(rt borrow tape, rt mut borrow scratch).is_err(); do');
  const publish = checked.indexOf('output[i] = scratch[i];');
  assert.ok(call >= 0 && publish > call, 'failure return must precede publication');
  assert.match(checked.slice(call,publish), /return Err\("forward failed"\);/);
});
