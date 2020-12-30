use crate::vec_utils::*;

pub trait StepSizeControl {
    fn do_step(
        &self,
        f: impl Fn(&[f64]) -> f64,
        x: &mut [f64],
        grad_f: &[f64],
        direction: &[f64],
    ) -> f64;
}

pub struct ArmijoGoldsteinRule {
    alpha_0: f64,
    tau: f64,
    c: f64,
}

impl ArmijoGoldsteinRule {
    #[allow(dead_code)]
    pub fn new(alpha_0: f64, tau: f64, c: f64) -> Self {
        ArmijoGoldsteinRule {
            alpha_0: alpha_0.max(1.0E-4),
            tau: tau.max(1.0E-4).min(1.0 - 1.0E-4),
            c: c.max(1.0E-4).min(1.0 - 1.0E-4),
        }
    }
}

impl StepSizeControl for ArmijoGoldsteinRule {
    fn do_step(
        &self,
        f: impl Fn(&[f64]) -> f64,
        x: &mut [f64],
        grad_f: &[f64],
        direction: &[f64],
    ) -> f64 {
        let m = inner_product(grad_f, direction).unwrap();
        let t = -self.c * m;

        let f_x: f64 = f(x);
        let mut x_step = add(&x, &scaled(direction, self.alpha_0)).unwrap();
        let mut f_x_step = f(&x_step);
        let mut alpha_j = self.alpha_0;

        while f_x - f_x_step < alpha_j * t {
            alpha_j *= self.tau;
            x_step = add(&x, &scaled(direction, alpha_j)).unwrap();
            f_x_step = f(&x_step);
        }

        for (x_i, x_step_i) in x.iter_mut().zip(x_step.iter()) {
            *x_i = *x_step_i;
        }

        alpha_j
    }
}
