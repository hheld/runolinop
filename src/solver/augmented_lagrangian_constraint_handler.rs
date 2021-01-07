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
        f + match inner_product(&self.lambda, h) {
            None => 0.0,
            Some(prod) => prod,
        } + 0.5 * self.c * norm2_sqr(&h)
            + self.mu.iter().zip(g.iter()).fold(0.0, |sum, (mu, g)| {
                sum + self.inequality_penalization(*g, *mu)
            })
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
        grad_f
            .iter()
            .zip(
                grad_g
                    .iter()
                    .zip(self.mu.iter())
                    .zip(g.iter())
                    .map(|((grad_g_j, mu_j), g_j)| {
                        grad_g_j.iter().fold(0.0, |sum, g_g_i| {
                            sum + 0.0_f64.max(*mu_j + self.c * g_j) * g_g_i
                        })
                    }),
            )
            .zip(
                grad_h
                    .iter()
                    .zip(self.lambda.iter())
                    .map(|(grad_h_j, lambda_j)| {
                        grad_h_j
                            .iter()
                            .fold(0.0, |sum, g_h_j| sum + lambda_j * g_h_j)
                    }),
            )
            .zip(
                grad_h.iter().zip(h.iter()).map(|(grad_h_j, h_j)| {
                    grad_h_j.iter().fold(0.0, |sum, g_h_j| sum + h_j * g_h_j)
                }),
            )
            .map(|(((g_f, g_ineq), g_equ), g_aug)| g_f + g_ineq + g_equ + g_aug)
            .collect()
    }

    #[allow(dead_code)]
    pub fn update_multipliers(&mut self, g: &[f64], h: &[f64]) {
        let c = self.c;

        for mu in self.mu.iter_mut() {
            *mu += g.iter().fold(0.0, |sum, g_j| sum + (c * g_j).max(-*mu));
        }

        for (lambda, h_j) in self.lambda.iter_mut().zip(h) {
            *lambda += c * h_j;
        }
    }
}
