from dataclasses import dataclass, field
from typing import Optional, List


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
    quant_level: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Specific quant set label, e.g. Q2, Q4_K_M for gguf; "
                "overrides bit_width if provided."
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
        default=True,
        metadata={"help": "Enable model evaluation after quantization."}
    )

    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset to use for evaluation."}
    )

    metrics: List[str] = field(
        default_factory=lambda: ["perplexity"],
        metadata={"help": "Metrics to compute during evaluation."}
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
