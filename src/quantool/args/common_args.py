from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


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


@dataclass  
class LoggingArguments:
    """
    Arguments for experiment logging integrations.
    """
    enable_mlflow: bool = field(
        default=False,
        metadata={"help": "Enable MLflow logging."}
    )
    
    mlflow_tracking_uri: Optional[str] = field(
        default=None,
        metadata={"help": "MLflow tracking server URI."}
    )
    
    mlflow_experiment_name: str = field(
        default="quantization",
        metadata={"help": "MLflow experiment name."}
    )
    
    enable_wandb: bool = field(
        default=False,
        metadata={"help": "Enable Weights & Biases logging."}
    )
    
    wandb_project: str = field(
        default="quantization",
        metadata={"help": "W&B project name."}
    )
    
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "W&B entity (username or team name)."}
    )
    
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Custom name for the experiment run."}
    )
    
    tags: List[str] = field(
        default_factory=list,
        metadata={"help": "Tags to add to the experiment run."}
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