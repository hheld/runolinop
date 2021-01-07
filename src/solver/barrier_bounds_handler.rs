use crate::{ObjectiveSense, VariableBounds};

pub struct BarrierBoundsHandler<'a> {
    pub bounds: &'a [VariableBounds],
    pub sense: &'a ObjectiveSense,
    pub barrier_parameter: f64,
    pub barrier_decrease_factor: f64,
}

impl BarrierBoundsHandler<'_> {
    pub fn adapted_objective_value(&self, xs: &[f64], pure_objective_value: f64) -> f64 {
        xs.iter()
            .zip(self.bounds.iter())
            .fold(pure_objective_value, |sum, (x, bounds)| {
                let mut barrier_term = 0.0;

                if bounds.lb > f64::NEG_INFINITY {
                    barrier_term += self.barrier_parameter * (x - bounds.lb).ln();
                }

                if bounds.ub < f64::INFINITY {
                    barrier_term -= self.barrier_parameter * (bounds.ub - x).ln();
                }

                sum + barrier_term
                    * match self.sense {
                        ObjectiveSense::Min => -1.0,
                        ObjectiveSense::Max => 1.0,
                    }
            })
    }

    pub fn adapted_objective_gradient(&self, xs: &[f64], pure_objective_grad: &[f64]) -> Vec<f64> {
        pure_objective_grad
            .iter()
            .zip(self.bounds.iter())
            .zip(xs.iter())
            .map(|((grad_obj, bounds), x)| {
                let mut grad_barrier_term = 0.0;

                if bounds.lb > f64::NEG_INFINITY {
                    grad_barrier_term += self.barrier_parameter * (1.0 / (x - bounds.lb));
                }

                if bounds.ub < f64::INFINITY {
                    grad_barrier_term -= self.barrier_parameter * (1.0 / (bounds.ub - x));
                }

                grad_obj
                    + grad_barrier_term
                        * match self.sense {
                            ObjectiveSense::Min => -1.0,
                            ObjectiveSense::Max => 1.0,
                        }
            })
            .collect()
    }

    pub fn update_barrier_parameter(&mut self) {
        self.barrier_parameter *= self.barrier_decrease_factor;
    }
}
