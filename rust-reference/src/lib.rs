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
    Ok(xs
        .iter()
        .zip(weights)
        .map(|(&x, &w)| {
            let value = x * w + x * x;
            value * value
        })
        .sum())
}

fn gradient(xs: &[f64], weights: &[f64]) -> Result<Vec<f64>, &'static str> {
    if xs.len() != weights.len() {
        return Err("shape mismatch");
    }
    Ok(xs
        .iter()
        .zip(weights)
        .map(|(&x, &w)| 2.0 * (x * w + x * x) * (w + 2.0 * x))
        .collect())
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

    fn replace(&mut self, values: &[f64]) -> Result<(), &'static str> {
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or("saved-forward generation exhausted")?;
        self.saved.clear();
        self.saved.extend_from_slice(values);
        self.generation = next_generation;
        Ok(())
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
        state.replace(&[3.0]).unwrap();
        assert_eq!(
            state.backward(initial, &[2.0]),
            Err("stale saved-forward generation")
        );
    }

    #[test]
    fn generation_exhaustion_fails_closed_without_replacing_snapshot() {
        let mut state = SavedForward::new(&[7.0]);
        state.generation = u64::MAX;
        assert_eq!(
            state.replace(&[9.0]),
            Err("saved-forward generation exhausted")
        );
        assert_eq!(state.saved, vec![7.0]);
        assert_eq!(state.generation, u64::MAX);
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
