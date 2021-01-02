use std::fmt;

use barrier_bounds_handler::BarrierBoundsHandler;

use crate::optimizer::Optimizer;
use crate::step_size_control::StepSizeControl;
use crate::NLP;

mod barrier_bounds_handler;

#[allow(dead_code)]
struct Solver<'a, N, S, O>
where
    N: NLP,
    S: StepSizeControl,
    O: Optimizer<N>,
{
    nlp: &'a N,
    step_size_control: &'a S,
    optimizer: &'a O,
    bounds_handler: BarrierBoundsHandler<'a>,
}

impl<N, S, O> Solver<'_, N, S, O>
where
    N: NLP,
    S: StepSizeControl,
    O: Optimizer<N>,
{
    #[allow(dead_code)]
    fn solve(&mut self) -> Solution {
        let mut context = self.optimizer.initialize(&self.nlp);

        while !self.optimizer.done(&context) {
            context.objective_previous = context.objective_current;
            context.x_previous = context.x_current.clone();
            context.iteration += 1;

            context.objective_grad = self.bounds_handler.adapted_objective_gradient(
                &context.x_current,
                &self.nlp.grad_objective(&context.x_current),
            );

            let d = self.optimizer.iterate(self.nlp, &mut context);

            let step_info = self.step_size_control.do_step(
                |xs| {
                    self.bounds_handler
                        .adapted_objective_value(&xs, self.nlp.objective(xs))
                },
                &mut context.x_current,
                &context.objective_grad,
                &d,
                &self.nlp.info().sense,
            );

            context.objective_current = step_info.obj_value;
            context.direction_scale_factor = step_info.direction_scale_factor;

            self.bounds_handler.end_of_iteration();
        }

        Solution {
            best_objective_value: context.objective_current,
            best_solution: context.x_current,
            num_iterations: context.iteration,
        }
    }
}

#[allow(dead_code)]
struct Solution {
    best_objective_value: f64,
    best_solution: Vec<f64>,
    num_iterations: u32,
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
    use crate::optimizer::SteepestDescent;
    use crate::step_size_control::ArmijoGoldsteinRule;
    use crate::{NlpInfo, ObjectiveSense, VariableBounds};

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
                sense: ObjectiveSense::Min,
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
            step_size_control: &step_rule,
            optimizer: &optimizer,
            bounds_handler: BarrierBoundsHandler {
                bounds: &nlp.bounds(),
                sense: &nlp.info().sense,
                barrier_parameter: 1.0E-6,
                barrier_decrease_factor: 0.5,
            },
        };

        let solution = solver.solve();

        println!("solution: {}", solution);
        assert!((solution.best_solution[0] - nlp.bounds()[0].lb).abs() < 1.0E-8);
    }

    #[test]
    fn max_unconstrained_steepest_descent() {
        let step_rule = ArmijoGoldsteinRule::new(1., 0.95, 0.01);

        struct MinXSquared {
            info: NlpInfo,
        };

        let nlp = MinXSquared {
            info: NlpInfo {
                num_variables: 1,
                num_inequality_constraints: 0,
                num_equality_constraints: 0,
                sense: ObjectiveSense::Max,
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
            step_size_control: &step_rule,
            optimizer: &optimizer,
            bounds_handler: BarrierBoundsHandler {
                bounds: &nlp.bounds(),
                sense: &nlp.info().sense,
                barrier_parameter: 1.0E-6,
                barrier_decrease_factor: 0.5,
            },
        };

        let solution = solver.solve();

        println!("solution: {}", solution);
        assert!((solution.best_solution[0] - nlp.bounds()[0].ub).abs() < 1.0E-8);
    }
}
