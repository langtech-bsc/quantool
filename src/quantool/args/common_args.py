from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class CommonArguments:
    """
    Common arguments shared across different quantization methods.
    """
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )

    device: Optional[str] = field(
        default=None,
        metadata={"help": "Device to use for computation (cuda, cpu, auto)."}
    )

    verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose logging."}
    )




@dataclass
class LoggingArguments:
    """
    Arguments for experiment logging and tracking.
    """
    report_to: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={"help": "Where to report the results (e.g., 'mlflow', 'wandb')."}
    )
    experiment_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the experiment for logging."}
    )
    log_level: str = field(
        default="INFO",
        metadata={"help": "Logging level (DEBUG, INFO, WARNING, ERROR)."}
    )

    save_logs: bool = field(
        default=True,
        metadata={"help": "Save logs to file."}
    )

    log_dir: Optional[str] = field(
        default="./logs",
        metadata={"help": "Directory to save log files."}
    )
