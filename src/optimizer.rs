use crate::{vec_utils, ObjectiveSense, UnconstrainedNlp};

#[allow(dead_code)]
pub struct OptContext {
    pub iteration: u32,
    pub x_current: Vec<f64>,
    pub x_previous: Vec<f64>,
    pub objective_current: f64,
    pub objective_previous: f64,
    pub objective_grad: Vec<f64>,
}

pub type StepDirection = Vec<f64>;

pub trait Optimizer<Nlp: UnconstrainedNlp> {
    fn initialize(&self, nlp: &Nlp) -> OptContext;
    fn iterate(&self, nlp: &Nlp, context: &mut OptContext) -> StepDirection;
    fn done(&self, context: &OptContext) -> bool;
}

pub struct SteepestDescent {}

impl<Nlp: UnconstrainedNlp> Optimizer<Nlp> for SteepestDescent {
    fn initialize(&self, nlp: &Nlp) -> OptContext {
        let nlp_info = nlp.info();

        OptContext {
            iteration: 0,
            x_current: nlp.initial_guess(),
            x_previous: nlp.initial_guess(),
            objective_current: 0.0,
            objective_previous: match nlp_info.sense {
                ObjectiveSense::Min => f64::INFINITY,
                ObjectiveSense::Max => f64::NEG_INFINITY,
            },
            objective_grad: vec![f64::INFINITY; nlp_info.num_variables as usize],
        }
    }

    fn iterate(&self, nlp: &Nlp, context: &mut OptContext) -> StepDirection {
        let nlp_info = nlp.info();

        match nlp_info.sense {
            ObjectiveSense::Min => vec_utils::scaled(&context.objective_grad, -1.0),
            ObjectiveSense::Max => context.objective_grad.clone(),
        }
    }

    fn done(&self, context: &OptContext) -> bool {
        (context.objective_current - context.objective_previous).abs() < 1.0E-9
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
