use runolinop::{
    ArmijoGoldsteinRule, AugmentedLagrangianConstraintHandler, BarrierBoundsHandler, Bfgs, NlpInfo,
    ObjectiveSense, Solver, StdoutLogger, VariableBounds, NLP,
};

fn rosenbrock(xs: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;

    for i in 0..n - 1 {
        sum += 100.0 * (xs[i + 1] - xs[i] * xs[i]).powi(2) + (1.0 - xs[i]).powi(2);
    }

    sum
}

fn grad_rosenbrock(xs: &[f64], n: usize) -> Vec<f64> {
    let mut grad = vec![0.0; n];

    for i in 0..n - 1 {
        grad[i] += -400.0 * (xs[i + 1] - xs[i] * xs[i]) * xs[i] - 2.0 * (1.0 - xs[i]);
        grad[i + 1] += 200.0 * (xs[i + 1] - xs[i] * xs[i]);
    }

    grad
}

#[test]
fn rosenbrock_bfgs_benchmark() {
    let step_rule = ArmijoGoldsteinRule::new(1., 0.5, 0.2);

    struct Rosenbrock {
        info: NlpInfo,
    };

    let nlp = Rosenbrock {
        info: NlpInfo {
            num_variables: 1000,
            num_inequality_constraints: 0,
            num_equality_constraints: 0,
            sense: ObjectiveSense::Min,
        },
    };

    impl NLP for Rosenbrock {
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
            rosenbrock(xs, self.info.num_variables as usize)
        }

        fn grad_objective(&self, xs: &[f64]) -> Vec<f64> {
            grad_rosenbrock(xs, self.info.num_variables as usize)
        }

        fn initial_guess(&self) -> Vec<f64> {
            vec![0.0; self.info.num_variables as usize]
        }
    }

    let mut optimizer = Bfgs::new(&nlp);
    let mut solver = Solver {
        nlp: &nlp,
        step_size_control: &step_rule,
        optimizer: &mut optimizer,
        bounds_handler: BarrierBoundsHandler {
            bounds: &nlp.bounds(),
            sense: &nlp.info().sense,
            barrier_parameter: 1.0E-6,
            barrier_decrease_factor: 0.5,
        },
        constraints_handler: AugmentedLagrangianConstraintHandler {
            mu: &mut vec![0.0; nlp.info().num_variables as usize],
            lambda: &mut vec![0.0; nlp.info.num_variables as usize],
            c: 1.0,
            sense: &ObjectiveSense::Min,
        },
        logger: vec![StdoutLogger::new(100)],
    };

    let solution = solver.solve();
    println!("solution: {}", solution);

    for i in 0..nlp.info.num_variables as usize {
        assert!(
            (solution.best_solution[i] - 1.0).abs() < 1.0E-6,
            "failing component {}: {}",
            i,
            solution.best_solution[i]
        );
    }
}
