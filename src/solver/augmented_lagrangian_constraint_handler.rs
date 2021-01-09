use crate::vec_utils::{inner_product, norm2_sqr};
use crate::ObjectiveSense;

#[allow(dead_code)]
pub struct AugmentedLagrangianConstraintHandler<'a> {
    pub mu: &'a mut [f64],
    pub lambda: &'a mut [f64],
    pub c: f64,
    pub sense: &'a ObjectiveSense,
}

impl AugmentedLagrangianConstraintHandler<'_> {
    #[allow(dead_code)]
    fn inequality_penalization(&self, t: f64, mu: f64) -> f64 {
        0.5 / self.c * (0.0_f64.max(mu + self.c * t).powi(2) - mu.powi(2))
    }

    #[allow(dead_code)]
    pub fn adapted_objective_value(&self, f: f64, g: &[f64], h: &[f64]) -> f64 {
        let mut obj = f;

        if h.len() > 0 {
            obj += inner_product(self.lambda, h).unwrap() + 0.5 * self.c * norm2_sqr(h);
        }

        obj += self.mu.iter().zip(g.iter()).fold(0.0, |sum, (mu, g)| {
            sum + self.inequality_penalization(*g, *mu)
        });

        obj
    }

    #[allow(dead_code)]
    pub fn adapted_objective_grad(
        &self,
        grad_f: &[f64],
        g: &[f64],
        grad_g: &[&[f64]],
        h: &[f64],
        grad_h: &[&[f64]],
    ) -> Vec<f64> {
        let grad_ineq = {
            let mut grad_ineq = vec![0.0; grad_f.len()];

            for (j, grad_g_j) in grad_g.iter().enumerate() {
                let factor = 0.0_f64.max(self.mu[j] + self.c * g[j]);

                for (i, grad_ineq_i) in grad_ineq.iter_mut().enumerate() {
                    *grad_ineq_i += factor * grad_g_j[i];
                }
            }

            grad_ineq
        };

        let grad_eq = {
            let mut grad_eq = vec![0.0; grad_f.len()];

            for (j, grad_h_j) in grad_h.iter().enumerate() {
                let factor = self.lambda[j] + self.c * h[j];

                for (i, grad_eq_i) in grad_eq.iter_mut().enumerate() {
                    *grad_eq_i += factor * grad_h_j[i];
                }
            }

            grad_eq
        };

        grad_f
            .iter()
            .zip(grad_ineq.iter())
            .zip(grad_eq.iter())
            .map(|((grad_f_i, grad_g_i), grad_h_i)| grad_f_i + grad_g_i + grad_h_i)
            .collect()
    }

    #[allow(dead_code)]
    pub fn update_multipliers(&mut self, g: &[f64], h: &[f64]) {
        let c = self.c;
        let sense_factor = match self.sense {
            ObjectiveSense::Min => 1.0,
            ObjectiveSense::Max => -1.0,
        };

        for mu in self.mu.iter_mut() {
            *mu += sense_factor * g.iter().fold(0.0, |sum, g_j| sum + (c * g_j).max(-*mu));
        }

        for (lambda, h_j) in self.lambda.iter_mut().zip(h) {
            *lambda += sense_factor * c * h_j;
        }
    }
}
