use core::fmt;

#[allow(dead_code)]
pub enum ObjectiveSense {
    Min,
    Max,
}

impl fmt::Display for ObjectiveSense {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ObjectiveSense::Min => write!(f, "Min"),
            ObjectiveSense::Max => write!(f, "Max"),
        }
    }
}

pub struct NlpInfo {
    pub num_variables: u32,
    pub num_inequality_constraints: u32,
    pub num_equality_constraints: u32,
    pub sense: ObjectiveSense,
}

impl fmt::Display for NlpInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "number of variables: {}", self.num_variables)?;
        writeln!(
            f,
            "number of inequality constraints: {}",
            self.num_inequality_constraints
        )?;
        writeln!(
            f,
            "number of equality constraints: {}",
            self.num_equality_constraints
        )?;
        writeln!(f, "objective sense: {}", self.sense)
    }
}

pub struct VariableBounds {
    pub lb: f64,
    pub ub: f64,
}

impl fmt::Display for VariableBounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "({}, {})", self.lb, self.ub)
    }
}

pub trait UnconstrainedNlp {
    fn info(&self) -> &NlpInfo;
    fn bounds(&self) -> Vec<VariableBounds>;

    fn objective(&self, xs: &[f64]) -> f64;
    fn grad_objective(&self, xs: &[f64]) -> Vec<f64>;

    fn initial_guess(&self) -> Vec<f64>;
}

pub fn dump_unconstrained_nlp(nlp: &dyn UnconstrainedNlp) {
    println!("NLP information: {}", nlp.info());

    for (v, b) in nlp.bounds().iter().enumerate() {
        println!("bounds for variable no. {}: {}", v, b);
    }
}

pub trait ConstrainedNlp: UnconstrainedNlp {
    fn equality_constraints(xs: &[f64]) -> Vec<f64>;
    fn grad_equality_constraints(xs: &[f64]) -> Vec<Vec<f64>>;

    fn inequality_constraints(xs: &[f64]) -> Vec<f64>;
    fn grad_inequality_constraints(xs: &[f64]) -> Vec<Vec<f64>>;
}

#[cfg(test)]
mod tests {
    use crate::nlp::{dump_unconstrained_nlp, ObjectiveSense};

    use super::*;

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

        dump_unconstrained_nlp(&nlp);
    }
}
