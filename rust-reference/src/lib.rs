fn finite_difference<F>(mut value: Vec<f64>, index: usize, function: F, eps: f64) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut left = value.clone();
    left[index] -= eps;
    value[index] += eps;
    (function(&value) - function(&left)) / (2.0 * eps)
}

fn scalar_loss(xs: &[f64], weights: &[f64]) -> Result<f64, &'static str> {
    if xs.len() != weights.len() {
        return Err("shape mismatch");
    }
    xs.iter().zip(weights).try_fold(0.0_f64, |total, (&x, &w)| {
        if !x.is_finite() || !w.is_finite() {
            return Err("nonfinite input");
        }
        let product = x * w;
        let squared_x = x * x;
        if !product.is_finite() || !squared_x.is_finite() {
            return Err("nonfinite forward intermediate");
        }
        let value = product + squared_x;
        let square = value * value;
        let next = total + square;
        if !value.is_finite() || !square.is_finite() || !next.is_finite() {
            return Err("nonfinite loss");
        }
        Ok(next)
    })
}

fn gradient(xs: &[f64], weights: &[f64]) -> Result<Vec<f64>, &'static str> {
    if xs.len() != weights.len() {
        return Err("shape mismatch");
    }
    xs.iter()
        .zip(weights)
        .map(|(&x, &w)| {
            if !x.is_finite() || !w.is_finite() {
                return Err("nonfinite input");
            }
            let product = x * w;
            let squared_x = x * x;
            let local = w + 2.0 * x;
            if !product.is_finite() || !squared_x.is_finite() || !local.is_finite() {
                return Err("nonfinite gradient intermediate");
            }
            let value = product + squared_x;
            let twice = 2.0 * value;
            let gradient = twice * local;
            if !value.is_finite() || !twice.is_finite() || !gradient.is_finite() {
                return Err("nonfinite gradient");
            }
            Ok(gradient)
        })
        .collect()
}

#[derive(Debug)]
struct SavedForward {
    saved: Vec<f64>,
    generation: u64,
}

impl SavedForward {
    fn new(values: &[f64]) -> Self {
        Self {
            saved: values.to_vec(),
            generation: 1,
        }
    }

    fn backward(&self, generation: u64, weights: &[f64]) -> Result<Vec<f64>, &'static str> {
        if generation != self.generation {
            return Err("stale saved-forward generation");
        }
        gradient(&self.saved, weights)
    }

    fn replace(&mut self, values: &[f64]) -> Result<u64, &'static str> {
        // Never wrap an incarnation lease: generation reuse could authorize a
        // stale backward request against a newer saved activation.
        // Preflight before touching the existing snapshot.
        let next = self
            .generation
            .checked_add(1)
            .ok_or("saved-forward generation exhausted")?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite saved-forward activation");
        }
        // Fully construct the replacement before publishing it. If the
        // allocation fails or unwinds, the previous snapshot remains intact.
        let replacement = values.to_vec();
        self.saved = replacement;
        self.generation = next;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::{finite_difference, gradient, scalar_loss, SavedForward};

    #[test]
    fn analytic_vjp_matches_independent_finite_difference() {
        let x = vec![-1.25, 0.0, 2.5, 0.7];
        let w = vec![2.0, -3.0, 0.25, 1.0];
        let actual = gradient(&x, &w).unwrap();
        for (index, &dy) in actual.iter().enumerate() {
            let numeric = finite_difference(
                x.clone(),
                index,
                |z| scalar_loss(z, &w).unwrap(),
                1e-6,
            );
            assert!(
                (dy - numeric).abs() <= 1e-5,
                "index {index}: analytic={dy} numeric={numeric}"
            );
        }
    }

    #[test]
    fn duplicate_edges_accumulate_not_overwrite() {
        assert_eq!(
            gradient(&[2.0], &[3.0]).unwrap(),
            vec![2.0 * (6.0 + 4.0) * (3.0 + 4.0)]
        );
    }

    #[test]
    fn empty_batch() {
        assert_eq!(scalar_loss(&[], &[]), Ok(0.0));
        assert_eq!(gradient(&[], &[]), Ok(Vec::new()));
    }

    #[test]
    fn shape_rejection_and_generation_fence() {
        assert_eq!(gradient(&[1.0], &[]), Err("shape mismatch"));

        let mut state = SavedForward::new(&[1.0]);
        let initial = state.generation;
        assert_eq!(
            state.backward(initial, &[2.0]).unwrap(),
            gradient(&[1.0], &[2.0]).unwrap()
        );
        assert_eq!(state.replace(&[3.0]), Ok(initial + 1));
        assert_eq!(
            state.backward(initial, &[2.0]),
            Err("stale saved-forward generation")
        );
    }

    #[test]
    fn exhausted_generation_refuses_reuse_without_modifying_state() {
        let mut state = SavedForward::new(&[4.0]);
        state.generation = u64::MAX;
        let previous = state.saved.clone();
        assert_eq!(
            state.replace(&[999.0]),
            Err("saved-forward generation exhausted")
        );
        assert_eq!(state.saved, previous);
        assert_eq!(state.generation, u64::MAX);
        assert_eq!(
            state.backward(u64::MAX, &[1.0]).unwrap(),
            gradient(&[4.0], &[1.0]).unwrap()
        );
        assert_eq!(
            state.backward(1, &[1.0]),
            Err("stale saved-forward generation")
        );
    }

    #[test]
    fn invalid_numeric_inputs_and_overflow_fail_closed() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(scalar_loss(&[invalid], &[1.0]), Err("nonfinite input"));
            assert_eq!(gradient(&[1.0], &[invalid]), Err("nonfinite input"));
        }
        assert!(scalar_loss(&[1.0e200], &[1.0]).is_err());
        assert!(gradient(&[1.0e308], &[1.0e308]).is_err());
        assert!(scalar_loss(&[1.0e308, 1.0e308], &[0.0, 0.0]).is_err());
    }

    #[test]
    fn invalid_replacement_preserves_saved_snapshot_and_lease() {
        let mut saved = SavedForward::new(&[2.0]);
        let old_generation = saved.generation;
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                saved.replace(&[invalid]),
                Err("nonfinite saved-forward activation")
            );
            assert_eq!(saved.generation, old_generation);
            assert_eq!(saved.saved, vec![2.0]);
        }
        assert_eq!(
            saved.backward(old_generation, &[3.0]).unwrap(),
            gradient(&[2.0], &[3.0]).unwrap()
        );
        assert!(saved.backward(old_generation, &[f64::NAN]).is_err());
        assert_eq!(saved.saved, vec![2.0]);
    }

    #[test]
    fn snapshot_cannot_be_changed_by_caller_mutation() {
        let mut x = vec![4.0];
        let saved = SavedForward::new(&x);
        x[0] = 999.0;
        assert_eq!(
            saved.backward(1, &[1.0]).unwrap(),
            gradient(&[4.0], &[1.0]).unwrap()
        );
    }
}
