use runolinop::*;

#[test]
fn nlp_definition() {
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

    dump_nlp(&nlp);
}
