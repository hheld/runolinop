use crate::optimizer::OptContext;
use crate::output::Frequency::EveryNthIteration;

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
                        "iteration {:5} | objective {:14.8} | change: {:14.8}",
                        context.iteration,
                        context.objective_current,
                        (context.objective_current - context.objective_previous).abs()
                    );

                    self.last_output_iteration = context.iteration;
                }
            }
        };
    }
}
