use crate::{vec_utils, AugmentedLagrangianConstraintHandler, BarrierBoundsHandler, NLP};
use nalgebra::{DMatrix, DVector};

#[allow(dead_code)]
pub struct OptContext {
    pub iteration: u32,
    pub x_current: Vec<f64>,
    pub x_previous: Vec<f64>,
    pub objective_current: f64,
    pub objective_previous: f64,
    pub objective_grad: Vec<f64>,
    pub direction_scale_factor: f64,
}

pub type StepDirection = Vec<f64>;

pub trait Optimizer<Nlp: NLP> {
    fn initialize(
        &mut self,
        nlp: &Nlp,
        bounds_handler: &BarrierBoundsHandler,
        constraint_handler: &AugmentedLagrangianConstraintHandler,
    ) -> OptContext;
    fn iterate(&mut self, nlp: &Nlp, context: &mut OptContext) -> StepDirection;
    fn done(&self, context: &OptContext) -> bool;
}

pub struct SteepestDescent {}

impl<Nlp: NLP> Optimizer<Nlp> for SteepestDescent {
    fn initialize(
        &mut self,
        nlp: &Nlp,
        _bounds_handler: &BarrierBoundsHandler,
        _constraint_handler: &AugmentedLagrangianConstraintHandler,
    ) -> OptContext {
        let nlp_info = nlp.info();

        OptContext {
            iteration: 0,
            x_current: nlp.initial_guess(),
            x_previous: nlp.initial_guess(),
            objective_current: 0.0,
            objective_previous: f64::INFINITY,
            objective_grad: vec![f64::INFINITY; nlp_info.num_variables as usize],
            direction_scale_factor: 1.0,
        }
    }

    fn iterate(&mut self, _nlp: &Nlp, context: &mut OptContext) -> StepDirection {
        vec_utils::scaled(&context.objective_grad, -1.0)
    }

    fn done(&self, context: &OptContext) -> bool {
        (context.objective_current - context.objective_previous).abs() < 1.0E-9
    }
}

#[allow(non_snake_case)]
pub struct Bfgs {
    g_k: DVector<f64>,
    d_k: DVector<f64>,
    H: DMatrix<f64>,
    n: u32,
}

impl Bfgs {
    #[allow(non_snake_case, dead_code)]
    pub fn new(nlp: &impl NLP) -> Self {
        let n = nlp.info().num_variables;
        let H = DMatrix::<f64>::identity(n as usize, n as usize);

        Bfgs {
            g_k: DVector::<f64>::zeros(0),
            d_k: DVector::<f64>::zeros(0),
            H,
            n,
        }
    }
}

impl<Nlp: NLP> Optimizer<Nlp> for Bfgs {
    fn initialize(
        &mut self,
        nlp: &Nlp,
        bounds_handler: &BarrierBoundsHandler,
        constraint_handler: &AugmentedLagrangianConstraintHandler,
    ) -> OptContext {
        let initial_point = nlp.initial_guess();

        let g = nlp.inequality_constraints(&initial_point);
        let h = nlp.equality_constraints(&initial_point);
        let grad_f = bounds_handler
            .adapted_objective_gradient(&initial_point, &nlp.grad_objective(&initial_point));

        let objective_grad = constraint_handler.adapted_objective_grad(
            &grad_f,
            &g,
            &nlp.grad_inequality_constraints(&initial_point)
                .iter()
                .map(|x| &x[..])
                .collect::<Vec<_>>(),
            &h,
            &nlp.grad_equality_constraints(&initial_point)
                .iter()
                .map(|x| &x[..])
                .collect::<Vec<_>>(),
        );

        self.g_k = DVector::<f64>::from_vec(objective_grad);
        self.d_k = -&self.H * &self.g_k;

        OptContext {
            iteration: 0,
            x_current: nlp.initial_guess(),
            x_previous: nlp.initial_guess(),
            objective_current: 0.0,
            objective_previous: f64::INFINITY,
            objective_grad: self.g_k.data.as_vec().clone(),
            direction_scale_factor: 1.0,
        }
    }

    #[allow(non_snake_case)]
    fn iterate(&mut self, _nlp: &Nlp, context: &mut OptContext) -> StepDirection {
        if context.iteration == 1 {
            return self.d_k.as_slice().to_vec();
        }

        let g_k_next = DVector::<f64>::from_vec(context.objective_grad.to_vec());
        let q_k = &g_k_next - &self.g_k;
        let p_k = context.direction_scale_factor * &self.d_k;

        self.g_k = g_k_next;

        let mut H_q_k = DVector::<f64>::zeros(self.n as usize);
        H_q_k.sygemv(1.0, &self.H, &q_k, 0.);

        let p_k_q_k = p_k.dot(&q_k);

        self.H.syger(
            1. / p_k_q_k + q_k.dot(&H_q_k) / p_k_q_k.powi(2),
            &p_k,
            &p_k,
            1.,
        );
        self.H.ger(-1. / p_k_q_k, &p_k, &H_q_k, 1.);
        self.H.ger(-1. / p_k_q_k, &H_q_k, &p_k, 1.);

        self.d_k.sygemv(-1.0, &self.H, &self.g_k, 0.);

        self.d_k.as_slice().to_vec()
    }

    fn done(&self, context: &OptContext) -> bool {
        (context.objective_current - context.objective_previous).abs() < 1.0E-12
    }
}

#[cfg(test)]
mod tests {
    use crate::optimizer::OptContext;

    #[test]
    fn opt_context_can_be_modified() {
        let x_previous = vec![0.0, 0.0, 0.0];
        let x_current = vec![1.0, 2.0, 3.0];
        let obj = 1.1;

        let mut oc = OptContext {
            iteration: 0,
            x_current,
            x_previous,
            objective_current: obj,
            objective_previous: 0.0,
            objective_grad: vec![4.4, 5.5, 6.6],
            direction_scale_factor: 1.0,
        };

        assert_eq!(oc.x_current, [1.0, 2.0, 3.0]);
        assert_eq!(oc.x_previous, [0.0, 0.0, 0.0]);
        assert_eq!(oc.objective_current, obj);

        oc.x_current[0] += 1.1;
        assert_eq!(oc.x_current, [2.1, 2.0, 3.0]);

        oc.x_previous[2] = -1.23;
        assert_eq!(oc.x_previous, [0.0, 0.0, -1.23]);

        oc.objective_current = 2.34;
        assert_eq!(oc.objective_current, 2.34);
    }
}
