
import logging
import threading
import time
from pythonjsonlogger.jsonlogger import JsonFormatter

class LoggerFactory:
    _lock = threading.Lock()
    _configured = False

    @classmethod
    def configure(cls, level=logging.INFO, fmt=None, json=False):
        """Configure root logger once."""
        with cls._lock:
            if cls._configured:
                return
            handler = logging.StreamHandler()
            if json:
                handler.setFormatter(JsonFormatter(fmt))
            else:
                fmt = fmt or '%(asctime)s %(levelname)s [%(name)s] %(message)s'
                handler.setFormatter(logging.Formatter(fmt))
            handler.setFormatter(logging.Formatter(fmt))
            root = logging.getLogger()
            root.setLevel(level)
            root.addHandler(handler)
            cls._configured = True

    @classmethod
    def get_logger(cls, name=None):
        """Get a module-level logger after ensuring configuration."""
        cls.configure()
        return logging.getLogger(name)

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
