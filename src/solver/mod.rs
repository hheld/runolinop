use std::fmt;

pub use augmented_lagrangian_constraint_handler::AugmentedLagrangianConstraintHandler;
pub use barrier_bounds_handler::BarrierBoundsHandler;

use crate::optimizer::Optimizer;
use crate::options::Options;
use crate::output::SolverLogger;
use crate::step_size_control::StepSizeControl;
use crate::vec_utils::norm2_sqr;
use crate::{ArmijoGoldsteinRule, Bfgs, StdoutLogger, NLP};

mod augmented_lagrangian_constraint_handler;
mod barrier_bounds_handler;

#[allow(dead_code)]
pub struct Solver<'a, N, S, O, L>
where
    N: NLP,
    S: StepSizeControl,
    O: Optimizer<N>,
    L: SolverLogger,
{
    nlp: &'a N,
    step_size_control: S,
    optimizer: O,
    bounds_handler: BarrierBoundsHandler,
    constraints_handler: AugmentedLagrangianConstraintHandler,
    logger: Vec<L>,
}

impl<'a, N> Solver<'a, N, ArmijoGoldsteinRule, Bfgs, StdoutLogger>
where
    N: NLP,
{
    pub fn new(nlp: &'a N, options: Options) -> Self {
        Self {
            nlp: &nlp,
            step_size_control: ArmijoGoldsteinRule::new(
                options.step_size_control.alpha_0,
                options.step_size_control.tau,
                options.step_size_control.c,
            ),
            optimizer: Bfgs::new(nlp),
            bounds_handler: BarrierBoundsHandler {
                bounds: nlp.bounds(),
                barrier_parameter: options.bounds_handler.barrier_parameter,
                barrier_decrease_factor: options.bounds_handler.barrier_decrease_factor,
            },
            constraints_handler: AugmentedLagrangianConstraintHandler {
                mu: vec![0.0; nlp.info().num_inequality_constraints as usize],
                lambda: vec![0.0; nlp.info().num_equality_constraints as usize],
                c: options.constraints_handler.c,
            },
            logger: vec![StdoutLogger::new(options.logger.frequency)],
        }
    }
}

impl<N, S, O, L> Solver<'_, N, S, O, L>
where
    N: NLP,
    S: StepSizeControl,
    O: Optimizer<N>,
    L: SolverLogger,
{
    #[allow(dead_code)]
    pub fn solve(&mut self) -> Solution {
        let mut context =
            self.optimizer
                .initialize(&self.nlp, &self.bounds_handler, &self.constraints_handler);

        while !self.optimizer.done(&context) {
            context.objective_previous = context.objective_current;
            context.iteration += 1;

            let g = self.nlp.inequality_constraints(&context.x_current);
            let h = self.nlp.equality_constraints(&context.x_current);
            let grad_f = self.bounds_handler.adapted_objective_gradient(
                &context.x_current,
                &self.nlp.grad_objective(&context.x_current),
            );

            context.objective_grad = self.constraints_handler.adapted_objective_grad(
                &grad_f,
                &g,
                &self
                    .nlp
                    .grad_inequality_constraints(&context.x_current)
                    .iter()
                    .map(|x| &x[..])
                    .collect::<Vec<_>>(),
                &h,
                &self
                    .nlp
                    .grad_equality_constraints(&context.x_current)
                    .iter()
                    .map(|x| &x[..])
                    .collect::<Vec<_>>(),
            );

            let d = self.optimizer.iterate(self.nlp, &mut context);
            context.x_previous = context.x_current.clone();

            if norm2_sqr(&d) < 1.0E-10 {
                break;
            }

            let step_info = self.step_size_control.do_step(
                |xs| {
                    self.constraints_handler.adapted_objective_value(
                        self.bounds_handler
                            .adapted_objective_value(&xs, self.nlp.objective(xs)),
                        &self.nlp.inequality_constraints(&xs),
                        &self.nlp.equality_constraints(&xs),
                    )
                },
                &mut context.x_current,
                &context.objective_grad,
                &d,
            );

            context.objective_current = step_info.obj_value;
            context.direction_scale_factor = step_info.direction_scale_factor;

            self.bounds_handler.update_barrier_parameter();

            let g = self.nlp.inequality_constraints(&context.x_current);
            let h = self.nlp.equality_constraints(&context.x_current);
            self.constraints_handler.update_multipliers(&g, &h);

            for logger in self.logger.iter_mut() {
                logger.log(&context, false);
            }
        }

        for logger in self.logger.iter_mut() {
            logger.log(&context, true);
        }

        Solution {
            best_objective_value: context.objective_current,
            best_solution: context.x_current,
            num_iterations: context.iteration,
        }
    }
}

#[allow(dead_code)]
pub struct Solution {
    pub best_objective_value: f64,
    pub best_solution: Vec<f64>,
    pub num_iterations: u32,
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "best objective value: {}", self.best_objective_value)?;
        writeln!(f, "best solution: {:?}", self.best_solution)?;
        write!(f, "in {} iterations", self.num_iterations)
    }
}

#[cfg(test)]
mod tests {
    use crate::optimizer::{Bfgs, SteepestDescent};
    use crate::output::StdoutLogger;
    use crate::step_size_control::ArmijoGoldsteinRule;
    use crate::{NlpInfo, VariableBounds};

    use super::*;

    #[test]
    fn min_unconstrained_steepest_descent() {
        let step_rule = ArmijoGoldsteinRule::new(1., 0.95, 0.01);

        struct MinXSquared {
            info: NlpInfo,
        };

        let nlp = MinXSquared {
            info: NlpInfo {
                num_variables: 1,
                num_inequality_constraints: 0,
                num_equality_constraints: 0,
            },
        };

        impl NLP for MinXSquared {
            fn info(&self) -> &NlpInfo {
                &self.info
            }

            fn bounds(&self) -> Vec<VariableBounds> {
                vec![VariableBounds { lb: 1.1, ub: 3.213 }]
            }

            fn objective(&self, xs: &[f64]) -> f64 {
                xs[0].powi(2)
            }

            fn grad_objective(&self, xs: &[f64]) -> Vec<f64> {
                vec![2.0 * xs[0]]
            }

            fn initial_guess(&self) -> Vec<f64> {
                vec![2.0]
            }
        }

        let optimizer = SteepestDescent {};

        let mut solver = Solver {
            nlp: &nlp,
            step_size_control: step_rule,
            optimizer,
            bounds_handler: BarrierBoundsHandler {
                bounds: nlp.bounds(),
                barrier_parameter: 1.0E-6,
                barrier_decrease_factor: 0.5,
            },
            constraints_handler: AugmentedLagrangianConstraintHandler {
                mu: vec![0.0; nlp.info().num_variables as usize],
                lambda: vec![0.0; nlp.info.num_variables as usize],
                c: 1.0,
            },
            logger: vec![StdoutLogger::new(1)],
        };

        let solution = solver.solve();

        println!("solution: {}", solution);
        assert!((solution.best_solution[0] - nlp.bounds()[0].lb).abs() < 1.0E-8);
    }

    #[test]
    fn min_unconstrained_bfgs() {
        let step_rule = ArmijoGoldsteinRule::new(1., 0.95, 0.01);

        struct MinXSquared {
            info: NlpInfo,
        };

        let nlp = MinXSquared {
            info: NlpInfo {
                num_variables: 1,
                num_inequality_constraints: 0,
                num_equality_constraints: 0,
            },
        };

        impl NLP for MinXSquared {
            fn info(&self) -> &NlpInfo {
                &self.info
            }

            fn bounds(&self) -> Vec<VariableBounds> {
                vec![VariableBounds { lb: 1.1, ub: 3.213 }]
            }

            fn objective(&self, xs: &[f64]) -> f64 {
                xs[0].powi(2)
            }

            fn grad_objective(&self, xs: &[f64]) -> Vec<f64> {
                vec![2.0 * xs[0]]
            }

            fn initial_guess(&self) -> Vec<f64> {
                vec![2.0]
            }
        }

        let optimizer = Bfgs::new(&nlp);

        let mut solver = Solver {
            nlp: &nlp,
            step_size_control: step_rule,
            optimizer,
            bounds_handler: BarrierBoundsHandler {
                bounds: nlp.bounds(),
                barrier_parameter: 1.0E-6,
                barrier_decrease_factor: 0.5,
            },
            constraints_handler: AugmentedLagrangianConstraintHandler {
                mu: vec![0.0; nlp.info().num_variables as usize],
                lambda: vec![0.0; nlp.info.num_variables as usize],
                c: 1.0,
            },
            logger: vec![StdoutLogger::new(1)],
        };

        let solution = solver.solve();

        println!("solution: {}", solution);
        assert!((solution.best_solution[0] - nlp.bounds()[0].lb).abs() < 1.0E-6);
    }
}
