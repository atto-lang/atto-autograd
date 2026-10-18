"""Independent scalar/vector VJP oracle, not guest compiler execution."""
import unittest

def finite_difference(fn, value, index, eps=1e-6):
    left=value.copy();right=value.copy()
    left[index]-=eps;right[index]+=eps
    return (fn(right)-fn(left))/(2*eps)

def scalar_loss(xs, weights):
    if len(xs)!=len(weights): raise ValueError("shape mismatch")
    # Repeated x contributes through two paths: x*w and x^2.
    return sum((x*w+x*x)**2 for x,w in zip(xs,weights))

def gradient(xs, weights):
    if len(xs)!=len(weights): raise ValueError("shape mismatch")
    return [2*(x*w+x*x)*(w+2*x) for x,w in zip(xs,weights)]

class SavedForward:
    def __init__(self, values):
        self.saved=tuple(values)
        self.generation=1
    def backward(self, generation, weights):
        if generation!=self.generation:
            raise RuntimeError("stale saved-forward generation")
        return gradient(list(self.saved),weights)
    def replace(self, values):
        self.saved=tuple(values)
        self.generation+=1

class GradientReferenceTests(unittest.TestCase):
    def test_analytic_vjp_matches_independent_finite_difference(self):
        x=[-1.25,0.0,2.5,0.7];w=[2.0,-3.0,0.25,1.0]
        actual=gradient(x,w)
        for i,dy in enumerate(actual):
            with self.subTest(i=i):
                numeric=finite_difference(lambda z:scalar_loss(z,w),x,i)
                self.assertAlmostEqual(dy,numeric,delta=1e-5)
    def test_duplicate_edges_accumulate_not_overwrite(self):
        self.assertEqual(gradient([2.0],[3.0]),[2*(6+4)*(3+4)])
    def test_empty_batch(self):
        self.assertEqual(scalar_loss([],[]),0)
        self.assertEqual(gradient([],[]),[])
    def test_shape_rejection_and_generation_fence(self):
        with self.assertRaises(ValueError):gradient([1],[])
        state=SavedForward([1.0]);initial=state.generation
        self.assertEqual(state.backward(initial,[2.0]),gradient([1.0],[2.0]))
        state.replace([3.0])
        with self.assertRaisesRegex(RuntimeError,"stale"):
            state.backward(initial,[2.0])
    def test_snapshot_cannot_be_changed_by_caller_mutation(self):
        x=[4.0];saved=SavedForward(x);x[0]=999.0
        self.assertEqual(saved.backward(1,[1.0]),gradient([4.0],[1.0]))

if __name__=="__main__":
    unittest.main()
