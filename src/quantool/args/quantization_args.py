# src/quantool/args.py
from dataclasses import dataclass, field
from typing import Optional, List



@dataclass
class ModelArguments:
    """
    Model loading and I/O paths.
    Mirrors llm-compressor’s ModelArguments for pretrained model, tokenizer, etc. :contentReference[oaicite:0]{index=0}
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
    # bit_width: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "Uniform bit-width (e.g. 4, 8); leave None for method defaults."},
    # )
    quant_level: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Specific quant set label, e.g. Q2, Q4_K_M for gguf; "
                "overrides bit_width if provided."
            )
        },
    )
    # inner config for quantization
    quantization_config: dict = field(
        default_factory=dict,
        metadata={"help": "Quantization specific configuration."},
    )


# not all quantization methods require calibration, refactoring to make it optional

# @dataclass
# class CalibrationArguments:
#     """
#     Calibration and data loading.
#     """
#     calibration_data: Optional[str] = field(
#         default=None,
#         metadata={"help": "Path to calibration data file or dataset name."},
#     )
#     batch_size: int = field(
#         default=16, metadata={"help": "Batch size for calibration forward passes."}
#     )

#     num_calibration_samples: int = field(
#         default=512,
#         metadata={"help": "Number of samples to use for calibration."},
#     )
#     max_seq_length: int = field(
#         default=2048,
#         metadata={"help": "Maximum sequence length for calibration data."},
#     )

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
