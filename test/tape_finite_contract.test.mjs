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
