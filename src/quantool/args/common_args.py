from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, NoneType


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
    
    log_level: str = field(
        default="INFO",
        metadata={"help": "Logging level (DEBUG, INFO, WARNING, ERROR)."}
    )
    
    dry_run: bool = field(
        default=False,
        metadata={"help": "Perform a dry run without actual quantization."}
    )
    
    force: bool = field(
        default=False,
        metadata={"help": "Force overwrite existing output files."}
    )

    report_to: Union[NoneType, str, List[str]] = field(
        default=None,
        metadata={"help": "Where to report the results (e.g., 'mlflow', 'wandb')."}
    )


@dataclass
class EvaluationArguments:
    """
    Arguments for model evaluation.
    """
    enable_evaluation: bool = field(
        default=True,
        metadata={"help": "Enable model evaluation after quantization."}
    )
    
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset to use for evaluation."}
    )
    
    eval_split: str = field(
        default="test",
        metadata={"help": "Dataset split to use for evaluation."}
    )
    
    eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for evaluation."}
    )
    
    eval_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to evaluate (None for all)."}
    )
    
    metrics: List[str] = field(
        default_factory=lambda: ["perplexity", "accuracy"],
        metadata={"help": "Metrics to compute during evaluation."}
    )
    
    compare_with_original: bool = field(
        default=True,
        metadata={"help": "Compare quantized model performance with original."}
    )