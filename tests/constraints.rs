use runolinop::{
    ArmijoGoldsteinRule, AugmentedLagrangianConstraintHandler, BarrierBoundsHandler, Bfgs, NlpInfo,
    ObjectiveSense, Solver, StdoutLogger, VariableBounds, NLP,
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
fn constrained_min_problem() {
    let step_rule = ArmijoGoldsteinRule::new(1., 0.5, 0.2);

    struct Prob {
        info: NlpInfo,
    };

    let nlp = Prob {
        info: NlpInfo {
            num_variables: 2,
            num_inequality_constraints: 0,
            num_equality_constraints: 0,
            sense: ObjectiveSense::Min,
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
        logger: vec![StdoutLogger::new(1)],
    };

    let solution = solver.solve();
    println!("solution: {}", solution);

    for i in 0..nlp.info.num_variables as usize {
        assert!(
            (solution.best_solution[i] - 0.0).abs() < 1.0E-6,
            "failing component {}: {}",
            i,
            solution.best_solution[i]
        );
    }
}
