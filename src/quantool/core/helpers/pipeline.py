from .logger_factory import LoggerFactory
from .metrics import MetricRecorder


class PipelineBase:
    """Base class for composable processing pipelines."""

    def __init__(self, steps=None, logger=None):
        self.steps = steps or []
        self.logger = logger or LoggerFactory.get_logger(self.__class__.__name__)
        self.metrics = MetricRecorder(self.logger)

    def add_step(self, func, name=None):
        """Register a processing step (callable)."""
        step_name = name or func.__name__
        self.steps.append((step_name, func))
        return self

    def run(self, data):
        """Execute all steps in order, passing data through."""
        current = data
        for name, func in self.steps:
            self.logger.info(f"Pipeline step START {name}")
            with self.metrics.timeit(name):
                current = func(current)
            self.logger.info(f"Pipeline step END {name}")
        return current
