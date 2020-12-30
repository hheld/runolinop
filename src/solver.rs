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

        while !self.optimizer.done(&context) {
            self.optimizer.iterate(self.nlp, &mut context);
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
    fn construct_unconstrained_solver() {
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
                vec![VariableBounds {
                    lb: 1.1,
                    ub: f64::INFINITY,
                }]
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
    }
}
