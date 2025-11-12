from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class ModelArguments:
    """
    Model loading and I/O paths.
    """

    model_id: str = field(
        metadata={"help": "Path, HF repo ID or alias of the pretrained model."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name/path if different from model_name_or_path."},
    )
    # TODO: add support for local_dir argument (similar to transformers snapshot_download)
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for model and tokenizer."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to use your HF auth token (needed for private models)."
        },
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Model Git revision (branch, tag, commit)."},
    )


@dataclass
class QuantizationArguments:
    """
    Quantizer selection and precision.
    """

    method: str = field(
        default="gguf",
        metadata={"help": "Quantization method: gptq, awq, gguf, higgs, aqml, etc."},
    )

    quant_level: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={
            "help": (
                "Specific quant set label, e.g. Q2, Q4_K_M for gguf; "
                "or list of labels for methods that support multiple quantization levels. "
                "Overrides bit_width if provided."
            )
        },
    )
    # Inner config for quantization
    quantization_config: dict = field(
        default_factory=dict,
        metadata={"help": "Quantization specific configuration."},
    )


@dataclass
class EvaluationArguments:
    """
    Arguments for model evaluation.
    """

    enable_evaluation: bool = field(
        default=True, metadata={"help": "Enable model evaluation after quantization."}
    )

    eval_dataset: Optional[str] = field(
        default=None, metadata={"help": "Dataset to use for evaluation."}
    )

    metrics: List[str] = field(
        default_factory=lambda: ["perplexity"],
        metadata={"help": "Metrics to compute during evaluation."},
    )


@dataclass
class ExportArguments:
    """
    Export format and output path.
    """

    output_path: str = field(
        default="quantized_model",
        metadata={"help": "Path (including filename) for the exported model."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Push the quantized model to Hugging Face Hub."},
    )
    repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository ID on Hugging Face Hub (username/repo-name)."},
    )
    private: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether the repository should be private on the Hub."},
    )


@dataclass
class CalibrationArguments:
    """
    Arguments for calibration dataset loading and preprocessing.
    """

    dataset_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "HF datasets id (or dataset identifier) used for calibration (e.g. 'wikitext')."
        },
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Local dataset path (file or directory) to use for calibration."
        },
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config name for HF datasets if required."},
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Dataset split to use (e.g. 'train', 'validation')."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for dataset downloads (for HF datasets)."},
    )
    sample_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Number of examples to use for calibration (None = full split)."
        },
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle before sampling for calibration data."},
    )
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed used for sampling/shuffling the calibration dataset."
        },
    )
    load_in_pipeline: bool = field(
        default=False,
        metadata={
            "help": "If True, the pipeline will load and preprocess the dataset and pass objects to the quantizer."
        },
    )
    preprocess_fn: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional name of a preprocessing function (module.func) to run on examples."
        },
    )
    # Method-specific calibration options
    calibration_config: dict = field(
        default_factory=dict,
        metadata={
            "help": "Method specific calibration config that is passed to quantizer.calibrate(...) if needed."
        },
    )
