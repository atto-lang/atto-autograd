// HOST-ONLY saved-forward row-gather VJP oracle.
// Actual Oreslang compilation, storage leases and actor state are NOT executed.
import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source=readFileSync(new URL("../src/row_gather_vjp.ores",import.meta.url),"utf8");
const guest=readFileSync(new URL("./row_gather_vjp_smoke.ores",import.meta.url),"utf8");
function tensor(rows,cols,values){
  if(!Number.isSafeInteger(rows)||!Number.isSafeInteger(cols)||rows<0||cols<0||
     values.length!==rows*cols||!values.every(Number.isFinite))throw Error("INVALID_TENSOR");
  return {rows,cols,values:[...values]};
}
function referenceGather(original,indices){
  const v=[];
  for(const index of indices){
    if(!Number.isSafeInteger(index)||index<0||index>=original.rows)throw Error("INVALID_GATHER");
    for(let c=0;c<original.cols;c++)v.push(original.values[index*original.cols+c]);
  }
  return tensor(indices.length,original.cols,v);
}
function referenceVjp(original,indices,saved,upstream,output) {
  if(saved.rows!==indices.length||upstream.rows!==indices.length||
     saved.cols!==original.cols||upstream.cols!==original.cols)throw Error("WRONG_SHAPE");
  const expected=referenceGather(original,indices);
  for(let i=0;i<expected.values.length;i++)
    if(saved.values[i]!==expected.values[i])throw Error("STALE_FORWARD");
  const scratch=new Array(original.rows*original.cols).fill(0);
  for(let i=0;i<indices.length;i++)for(let c=0;c<original.cols;c++) {
    const k=indices[i]*original.cols+c;
    const value=scratch[k]+upstream.values[i*original.cols+c];
    if(!Number.isFinite(value))throw Error("NONFINITE");
    scratch[k]=value;
  }
  for(let i=0;i<output.length;i++)output[i]=scratch[i];
}
test("adapter calls tensor VJP only after saved-forward value recheck",()=>{
  assert.match(source,/pub fnc row_gather_vjp_with_saved_f64_checked_into/);
  assert.match(source,/gather_rows_2d_f64_into\(original, indices, forward_scratch\)/);
  assert.match(source,/stale saved row-gather activation/);
  assert.match(source,/return gather_rows_vjp_f64_checked_into\(/);
  assert(source.indexOf("stale saved row-gather activation") <
         source.indexOf("return gather_rows_vjp_f64_checked_into("));
  assert.match(guest,/ORES_AUTOGRAD_ROW_GATHER_VJP_PASS/);
});
test("duplicate-row VJP and zero-contribution rows match expected adjoint",()=>{
  const x=tensor(3,2,[1,2,3,4,5,6]);
  const indices=[2,0,2];
  const saved=referenceGather(x,indices);
  const upstream=tensor(3,2,[1,2,3,4,5,6]);
  const output=Array(6).fill(99);
  referenceVjp(x,indices,saved,upstream,output);
  assert.deepEqual(output,[3,4,0,0,6,8]);
});
test("stale activation, out-of-range index and bad shape preserve gradient output",()=>{
  const x=tensor(3,2,[1,2,3,4,5,6]), upstream=tensor(3,2,[1,2,3,4,5,6]);
  for(const [ids,saved] of [
    [[2,0,2],tensor(3,2,[8,6,1,2,5,6])],
    [[2,9,2],tensor(3,2,[5,6,1,2,5,6])],
    [[2,0,2],tensor(3,1,[5,1,5])]
  ]){
    const out=Array(6).fill(88);
    assert.throws(()=>referenceVjp(x,ids,saved,upstream,out));
    assert.deepEqual(out,Array(6).fill(88));
  }
});
test("zero-row/zero-feature tensors preserve legal empty gradients",()=>{
  const x=tensor(0,3,[]),saved=tensor(0,3,[]),upstream=tensor(0,3,[]);
  const out=[];
  referenceVjp(x,[],saved,upstream,out);
  assert.deepEqual(out,[]);
  const x0=tensor(2,0,[]),selected=tensor(3,0,[]);
  referenceVjp(x0,[1,0,1],selected,selected,[]);
});
test("saved-forward VJP matches central finite differences",()=>{
  const x=tensor(3,2,[1,2,-3,4,5,-6]),idx=[2,0,2];
  const saved=referenceGather(x,idx),upstream=tensor(3,2,[2,-1,4,3,-2,5]);
  const analytic=Array(6).fill(0);
  referenceVjp(x,idx,saved,upstream,analytic);
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const h=1e-5;
  for(let k=0;k<x.values.length;k++){
    const plus=x.values.slice(),minus=x.values.slice();
    plus[k]+=h;minus[k]-=h;
    const yPlus=referenceGather(tensor(3,2,plus),idx);
    const yMinus=referenceGather(tensor(3,2,minus),idx);
    const finite=(dot(yPlus.values,upstream.values)-dot(yMinus.values,upstream.values))/(2*h);
    assert(Math.abs(finite-analytic[k])<2e-9);
  }
});
