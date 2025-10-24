import time
from .logger_factory import LoggerFactory

class MetricRecorder:
    """Context manager for recording metrics."""

    def __init__(self, logger=None):
        self.logger = logger or LoggerFactory.get_logger(self.__class__.__name__)

    def timeit(self, op_name):
        """Context manager to log start/end timestamps and duration."""

        class TimerCtx:
            def __enter__(inner_self):
                inner_self.start = time.time()
                self.logger.info(f"START {op_name}")
                return inner_self

            def __exit__(inner_self, exc_type, exc, tb):
                dur = time.time() - inner_self.start
                self.logger.info(f"END {op_name} (duration={dur:.3f}s)")

        return TimerCtx()

    def count(self, metric_name, value=1):
        """Log an incrementing counter metric."""
        self.logger.info(f"METRIC counter {metric_name}={value}")
