use runolinop::{
    ArmijoGoldsteinRule, AugmentedLagrangianConstraintHandler, BarrierBoundsHandler, Bfgs, NlpInfo,
    Solver, StdoutLogger, VariableBounds, NLP,
};

fn f(xs: &[f64]) -> f64 {
    xs[0].powi(2) + xs[1].powi(2)
}

fn grad_f(xs: &[f64]) -> Vec<f64> {
    let mut grad = vec![0.0; 2];

    grad[0] = 2.0 * xs[0];
    grad[1] = 2.0 * xs[1];

    grad
}

#[test]
fn inequality_constrained_min_problem() {
    let step_rule = ArmijoGoldsteinRule::new(1., 0.5, 0.2);

    struct Prob {
        info: NlpInfo,
    };

    let nlp = Prob {
        info: NlpInfo {
            num_variables: 2,
            num_inequality_constraints: 1,
            num_equality_constraints: 0,
        },
    };

    impl NLP for Prob {
        fn info(&self) -> &NlpInfo {
            &self.info
        }

        fn bounds(&self) -> Vec<VariableBounds> {
            vec![
                VariableBounds {
                    lb: 0.0,
                    ub: f64::INFINITY,
                };
                self.info.num_variables as usize
            ]
        }

        fn objective(&self, xs: &[f64]) -> f64 {
            f(xs)
        }

        fn grad_objective(&self, xs: &[f64]) -> Vec<f64> {
            grad_f(xs)
        }

        fn inequality_constraints(&self, xs: &[f64]) -> Vec<f64> {
            vec![-xs[0] - xs[1] + 0.5]
        }

        fn grad_inequality_constraints(&self, _xs: &[f64]) -> Vec<Vec<f64>> {
            vec![vec![-1.0, -1.0]]
        }

        fn initial_guess(&self) -> Vec<f64> {
            vec![1.0; self.info.num_variables as usize]
        }
    }

    let mut optimizer = Bfgs::new(&nlp);
    let mut solver = Solver {
        nlp: &nlp,
        step_size_control: &step_rule,
        optimizer: &mut optimizer,
        bounds_handler: BarrierBoundsHandler {
            bounds: &nlp.bounds(),
            barrier_parameter: 1.0E-6,
            barrier_decrease_factor: 0.5,
        },
        constraints_handler: AugmentedLagrangianConstraintHandler {
            mu: &mut vec![0.0; nlp.info().num_variables as usize],
            lambda: &mut vec![0.0; nlp.info.num_variables as usize],
            c: 1.0E9,
        },
        logger: vec![StdoutLogger::new(1)],
    };

    let solution = solver.solve();
    println!("solution: {}", solution);

    assert!(
        nlp.inequality_constraints(&solution.best_solution)[0] <= 0.0,
        "sum of variable values: {}",
        solution.best_solution.iter().sum::<f64>()
    );
}

#[test]
fn inequality_constrained_max_problem() {
    let step_rule = ArmijoGoldsteinRule::new(1., 0.5, 0.2);

    struct Prob {
        info: NlpInfo,
    };

    let nlp = Prob {
        info: NlpInfo {
            num_variables: 2,
            num_inequality_constraints: 1,
            num_equality_constraints: 0,
        },
    };

    impl NLP for Prob {
        fn info(&self) -> &NlpInfo {
            &self.info
        }

        fn bounds(&self) -> Vec<VariableBounds> {
            vec![
                VariableBounds {
                    lb: f64::NEG_INFINITY,
                    ub: f64::INFINITY,
                };
                self.info.num_variables as usize
            ]
        }

        fn objective(&self, xs: &[f64]) -> f64 {
            f(xs)
        }

        fn grad_objective(&self, xs: &[f64]) -> Vec<f64> {
            grad_f(xs)
        }

        fn inequality_constraints(&self, xs: &[f64]) -> Vec<f64> {
            vec![xs[0] + xs[1] - 0.5]
        }

        fn grad_inequality_constraints(&self, _xs: &[f64]) -> Vec<Vec<f64>> {
            vec![vec![1.0, 1.0]]
        }

        fn initial_guess(&self) -> Vec<f64> {
            vec![1.0; self.info.num_variables as usize]
        }
    }

    let mut optimizer = Bfgs::new(&nlp);
    let mut solver = Solver {
        nlp: &nlp,
        step_size_control: &step_rule,
        optimizer: &mut optimizer,
        bounds_handler: BarrierBoundsHandler {
            bounds: &nlp.bounds(),
            barrier_parameter: 1.0E-6,
            barrier_decrease_factor: 0.5,
        },
        constraints_handler: AugmentedLagrangianConstraintHandler {
            mu: &mut vec![0.0; nlp.info().num_variables as usize],
            lambda: &mut vec![0.0; nlp.info.num_variables as usize],
            c: 1.0E9,
        },
        logger: vec![StdoutLogger::new(1)],
    };

    let solution = solver.solve();
    println!("solution: {}", solution);

    assert!(
        nlp.inequality_constraints(&solution.best_solution)[0] <= 0.0,
        "sum of variable values: {}",
        solution.best_solution.iter().sum::<f64>()
    );
}

#[test]
fn equality_constrained_min_problem() {
    let step_rule = ArmijoGoldsteinRule::new(1., 0.5, 0.2);

    struct Prob {
        info: NlpInfo,
    };

    let nlp = Prob {
        info: NlpInfo {
            num_variables: 2,
            num_inequality_constraints: 0,
            num_equality_constraints: 1,
        },
    };

    impl NLP for Prob {
        fn info(&self) -> &NlpInfo {
            &self.info
        }

        fn bounds(&self) -> Vec<VariableBounds> {
            vec![
                VariableBounds {
                    lb: f64::NEG_INFINITY,
                    ub: f64::INFINITY,
                };
                self.info.num_variables as usize
            ]
        }

        fn objective(&self, xs: &[f64]) -> f64 {
            f(xs)
        }

        fn grad_objective(&self, xs: &[f64]) -> Vec<f64> {
            grad_f(xs)
        }

        fn equality_constraints(&self, xs: &[f64]) -> Vec<f64> {
            vec![xs[0] + xs[1] - 0.5]
        }

        fn grad_equality_constraints(&self, _xs: &[f64]) -> Vec<Vec<f64>> {
            vec![vec![1.0, 1.0]]
        }

        fn initial_guess(&self) -> Vec<f64> {
            vec![1.0; self.info.num_variables as usize]
        }
    }

    let mut optimizer = Bfgs::new(&nlp);
    let mut solver = Solver {
        nlp: &nlp,
        step_size_control: &step_rule,
        optimizer: &mut optimizer,
        bounds_handler: BarrierBoundsHandler {
            bounds: &nlp.bounds(),
            barrier_parameter: 1.0E-6,
            barrier_decrease_factor: 0.5,
        },
        constraints_handler: AugmentedLagrangianConstraintHandler {
            mu: &mut vec![0.0; nlp.info().num_inequality_constraints as usize],
            lambda: &mut vec![0.0; nlp.info().num_equality_constraints as usize],
            c: 1.0E9,
        },
        logger: vec![StdoutLogger::new(1)],
    };

    let solution = solver.solve();
    println!("solution: {}", solution);

    assert!(
        nlp.equality_constraints(&solution.best_solution)[0].abs() <= 1.0E-3,
        "sum of variable values: {}",
        solution.best_solution.iter().sum::<f64>()
    );
}
