use crate::{ConstrainedNlp, UnconstrainedNlp};

#[allow(dead_code)]
struct OptContext {
    iteration: u32,
    x_current: Vec<f64>,
    x_previous: Vec<f64>,
    objective_current: f64,
    objective_previous: f64,
    objective_grad: Vec<f64>,
}

trait Optimizer {
    type Nlp: UnconstrainedNlp + ConstrainedNlp;

    fn initialize(nlp: &Self::Nlp) -> &mut OptContext;
    fn iterate(nlp: &Self::Nlp, context: &mut OptContext);
    fn done(context: &OptContext) -> bool;
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
