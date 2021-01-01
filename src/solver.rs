use crate::optimizer::Optimizer;
use crate::step_size_control::StepSizeControl;
use crate::UnconstrainedNlp;
use std::fmt;

#[allow(dead_code)]
struct UnconstrainedSolver<'a, N, S, O>
where
    N: UnconstrainedNlp,
    S: StepSizeControl,
    O: Optimizer<N>,
{
    nlp: &'a N,
    step_size_control: &'a S,
    optimizer: &'a O,
}

impl<N, S, O> UnconstrainedSolver<'_, N, S, O>
where
    N: UnconstrainedNlp,
    S: StepSizeControl,
    O: Optimizer<N>,
{
    #[allow(dead_code)]
    fn solve(&self) -> Solution {
        let mut context = self.optimizer.initialize(&self.nlp);
        let mut barrier_parameter = 1.0E-6;
        let barrier_decrease_factor = 0.5;
        let bounds = self.nlp.bounds();

        while !self.optimizer.done(&context) {
            context.objective_previous = context.objective_current;
            context.x_previous = context.x_current.clone();
            context.iteration += 1;
            context.objective_grad = self
                .nlp
                .grad_objective(&context.x_current)
                .iter()
                .zip(bounds.iter())
                .zip(context.x_current.iter())
                .map(|((grad_obj, bounds), x)| {
                    let mut grad_barrier_term = 0.0;

                    if bounds.lb > f64::NEG_INFINITY {
                        grad_barrier_term += barrier_parameter * (1.0 / (x - bounds.lb));
                    }

                    if bounds.ub < f64::INFINITY {
                        grad_barrier_term -= barrier_parameter * (1.0 / (bounds.ub - x));
                    }

                    grad_obj - grad_barrier_term
                })
                .collect();

            let d = self.optimizer.iterate(self.nlp, &mut context);

            context.objective_current = self.step_size_control.do_step(
                |xs| {
                    self.nlp.objective(xs)
                        + xs.iter().zip(bounds.iter()).fold(0.0, |sum, (x, bounds)| {
                            let mut barrier_term = 0.0;

                            if bounds.lb > f64::NEG_INFINITY {
                                barrier_term += barrier_parameter * (x - bounds.lb).ln();
                            }

                            if bounds.ub < f64::INFINITY {
                                barrier_term -= barrier_parameter * (bounds.ub - x).ln();
                            }

                            sum - barrier_term
                        })
                },
                &mut context.x_current,
                &context.objective_grad,
                &d,
                &self.nlp.info().sense,
            );

            barrier_parameter *= barrier_decrease_factor;
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
    use super::*;
    use crate::optimizer::SteepestDescent;
    use crate::step_size_control::ArmijoGoldsteinRule;
    use crate::{NlpInfo, ObjectiveSense, VariableBounds};

    #[test]
    fn min_unconstrained_steepeset_descent() {
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

        impl UnconstrainedNlp for MinXSquared {
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

        let solver = UnconstrainedSolver {
            nlp: &nlp,
            step_size_control: &step_rule,
            optimizer: &optimizer,
        };

        let solution = solver.solve();

        println!("solution: {}", solution);
        assert!((solution.best_solution[0] - nlp.bounds()[0].lb).abs() < 1.0E-8);
    }

    #[test]
    fn max_unconstrained_steepset_descent() {
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

        impl UnconstrainedNlp for MinXSquared {
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

        let solver = UnconstrainedSolver {
            nlp: &nlp,
            step_size_control: &step_rule,
            optimizer: &optimizer,
        };

        let solution = solver.solve();

        println!("solution: {}", solution);
        assert!((solution.best_solution[0] - nlp.bounds()[0].ub).abs() < 1.0E-8);
    }
}
