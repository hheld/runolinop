pub struct Options {
    pub step_size_control: StepSizeControl,
    pub bounds_handler: BoundsHandler,
    pub constraints_handler: ConstraintsHandler,
    pub logger: Logger,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            step_size_control: Default::default(),
            bounds_handler: Default::default(),
            constraints_handler: Default::default(),
            logger: Default::default(),
        }
    }
}

pub struct StepSizeControl {
    pub alpha_0: f64,
    pub tau: f64,
    pub c: f64,
}

impl Default for StepSizeControl {
    fn default() -> Self {
        Self {
            alpha_0: 1.0,
            tau: 0.5,
            c: 0.2,
        }
    }
}

pub struct BoundsHandler {
    pub barrier_parameter: f64,
    pub barrier_decrease_factor: f64,
}

impl Default for BoundsHandler {
    fn default() -> Self {
        Self {
            barrier_parameter: 1.0E-6,
            barrier_decrease_factor: 0.5,
        }
    }
}

pub struct ConstraintsHandler {
    pub c: f64,
}

impl Default for ConstraintsHandler {
    fn default() -> Self {
        Self { c: 1.0E9 }
    }
}

pub struct Logger {
    pub frequency: u32,
}

impl Default for Logger {
    fn default() -> Self {
        Self { frequency: 1 }
    }
}
