import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, existsSync} from 'node:fs';
const root=new URL('../',import.meta.url);
const manifest=JSON.parse(readFileSync(new URL('atto-capabilities.json',root),'utf8'));
test('inventory reflects host reference implementation without certifying Atto guest',()=>{
  assert.deepEqual(Object.keys(manifest).sort(),['schema','repository','role','source_state','feature_status','certification','evidence'].sort());
  assert.equal(manifest.schema,'atto.repository-capabilities.v1');
  assert.equal(manifest.repository,'atto-lang/atto-autograd');
  assert.equal(manifest.role,'library');
  assert.equal(manifest.source_state,'implementation');
  assert.ok(existsSync(new URL('rust-reference/src/lib.rs',root)));
  assert.deepEqual(manifest.feature_status,{public_api:'not_executed',guest_execution:'not_executed'});
  assert.equal(manifest.certification,'NOT_CERTIFIED');
  assert.deepEqual(manifest.evidence,[]);
});
