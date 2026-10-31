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

use std::sync::atomic::{AtomicU64, Ordering};

// Process-local unique incarnation IDs for the host reference only.
// This does not establish distributed/actor-isolate capability safety.
static NEXT_SAVED_FORWARD_INSTANCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SavedForwardLease {
    instance_id: u64,
    generation: u64,
}

#[derive(Debug)]
struct SavedForward {
    saved: Vec<f64>,
    instance_id: u64,
    generation: u64,
}

impl SavedForward {
    fn new(values: &[f64]) -> Result<Self, &'static str> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite saved-forward activation");
        }
        // Construct the snapshot before issuing a process-local lease identity.
        let saved = values.to_vec();
        let instance_id = NEXT_SAVED_FORWARD_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| "saved-forward instance identity exhausted")?;
        Ok(Self {
            saved,
            instance_id,
            generation: 1,
        })
    }

    fn lease(&self) -> SavedForwardLease {
        SavedForwardLease {
            instance_id: self.instance_id,
            generation: self.generation,
        }
    }

    fn backward(&self, lease: SavedForwardLease, weights: &[f64]) -> Result<Vec<f64>, &'static str> {
        if lease.instance_id != self.instance_id || lease.generation != self.generation {
            return Err("stale saved-forward generation");
        }
        gradient(&self.saved, weights)
    }

    fn replace(&mut self, values: &[f64]) -> Result<SavedForwardLease, &'static str> {
        // Generation wrap must never authorize a stale lease.
        let next = self
            .generation
            .checked_add(1)
            .ok_or("saved-forward generation exhausted")?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite saved-forward activation");
        }
        // Allocate a complete replacement before publishing either field.
        let replacement = values.to_vec();
        self.saved = replacement;
        self.generation = next;
        Ok(self.lease())
    }
}

#[cfg(test)]
mod tests {
    use super::{finite_difference, gradient, scalar_loss, SavedForward, SavedForwardLease};

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

        let mut state = SavedForward::new(&[1.0]).unwrap();
        let initial = state.lease();
        assert_eq!(
            state.backward(initial, &[2.0]).unwrap(),
            gradient(&[1.0], &[2.0]).unwrap()
        );
        let replacement = state.replace(&[3.0]).unwrap();
        assert_eq!(replacement.instance_id, initial.instance_id);
        assert_eq!(replacement.generation, initial.generation + 1);
        assert_eq!(
            state.backward(initial, &[2.0]),
            Err("stale saved-forward generation")
        );
    }

    #[test]
    fn exhausted_generation_refuses_reuse_without_modifying_state() {
        let mut state = SavedForward::new(&[4.0]).unwrap();
        state.generation = u64::MAX;
        let current = state.lease();
        let previous = state.saved.clone();
        assert_eq!(
            state.replace(&[999.0]),
            Err("saved-forward generation exhausted")
        );
        assert_eq!(state.saved, previous);
        assert_eq!(state.lease(), current);
        assert_eq!(
            state.backward(current, &[1.0]).unwrap(),
            gradient(&[4.0], &[1.0]).unwrap()
        );
        assert_eq!(
            state.backward(SavedForwardLease { generation: 1, ..current }, &[1.0]),
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
        let mut saved = SavedForward::new(&[2.0]).unwrap();
        let old_lease = saved.lease();
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                saved.replace(&[invalid]),
                Err("nonfinite saved-forward activation")
            );
            assert_eq!(saved.lease(), old_lease);
            assert_eq!(saved.saved, vec![2.0]);
        }
        assert_eq!(
            saved.backward(old_lease, &[3.0]).unwrap(),
            gradient(&[2.0], &[3.0]).unwrap()
        );
        assert!(saved.backward(old_lease, &[f64::NAN]).is_err());
        assert_eq!(saved.saved, vec![2.0]);
    }

    #[test]
    fn initial_snapshot_rejects_nonfinite_activations() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                SavedForward::new(&[2.0, invalid]).unwrap_err(),
                "nonfinite saved-forward activation"
            );
        }
        assert_eq!(SavedForward::new(&[]).unwrap().generation, 1);
        assert_eq!(SavedForward::new(&[2.0, -3.0]).unwrap().saved, vec![2.0, -3.0]);
    }

    #[test]
    fn same_generation_from_another_instance_never_authorizes_backward() {
        let first = SavedForward::new(&[2.0]).unwrap();
        let second = SavedForward::new(&[7.0]).unwrap();
        let ticket = first.lease();
        assert_eq!(ticket.generation, second.lease().generation);
        assert_ne!(ticket.instance_id, second.lease().instance_id);
        assert_eq!(
            second.backward(ticket, &[1.0]),
            Err("stale saved-forward generation")
        );
        assert_eq!(
            second.backward(second.lease(), &[1.0]).unwrap(),
            gradient(&[7.0], &[1.0]).unwrap()
        );
    }

    #[test]
    fn snapshot_cannot_be_changed_by_caller_mutation() {
        let mut x = vec![4.0];
        let saved = SavedForward::new(&x).unwrap();
        x[0] = 999.0;
        assert_eq!(
            saved.backward(saved.lease(), &[1.0]).unwrap(),
            gradient(&[4.0], &[1.0]).unwrap()
        );
    }
}
