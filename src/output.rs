use crate::optimizer::OptContext;
use crate::output::Frequency::EveryNthIteration;

#[allow(dead_code)]
pub enum Frequency {
    EveryNthIteration(u32),
}

pub trait SolverLogger {
    fn log(&mut self, context: &OptContext, ignore_frequency: bool);
}

pub struct StdoutLogger {
    frequency: Frequency,
    last_output_iteration: u32,
}

impl StdoutLogger {
    #[allow(dead_code)]
    pub fn new(n: u32) -> Self {
        StdoutLogger {
            frequency: EveryNthIteration(n),
            last_output_iteration: 0,
        }
    }
}

impl SolverLogger for StdoutLogger {
    fn log(&mut self, context: &OptContext, ignore_frequency: bool) {
        match self.frequency {
            EveryNthIteration(freq) => {
                if (context.iteration % freq == 0 && !ignore_frequency)
                    || (ignore_frequency && self.last_output_iteration != context.iteration)
                {
                    println!(
                        "iteration {:5} | objective {:20.8} (actual: {:14.8}) | change: {:14.8}",
                        context.iteration,
                        context.objective_current,
                        context.pure_objective,
                        (context.objective_current - context.objective_previous).abs()
                    );

                    self.last_output_iteration = context.iteration;
                }
            }
        };
    }
}
