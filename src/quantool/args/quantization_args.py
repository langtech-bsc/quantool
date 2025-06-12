# src/quantool/args.py
from dataclasses import dataclass, field
from typing import Optional, List



@dataclass
class ModelArguments:
    """
    Model loading and I/O paths.
    Mirrors llm-compressor’s ModelArguments for pretrained model, tokenizer, etc. :contentReference[oaicite:0]{index=0}
    """
    model_name_or_path: str = field(
        metadata={"help": "Path, HF repo ID or alias of the pretrained model."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Config name/path if different from model_name_or_path."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name/path if different from model_name_or_path."},
    )
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
    revision: str = field(
        default="main",
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
    bit_width: Optional[int] = field(
        default=None,
        metadata={"help": "Uniform bit-width (e.g. 4, 8); leave None for method defaults."},
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
    quant_set: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Allowed quant_set choices; defaults to plugin’s supported_levels."
            )
        },
    )
    # inner config for quantization
    quantization_config: dict = field(
        default_factory=dict,
        metadata={"help": "Quantization specific configuration."},
    )


# not all quantization methods require calibration, refactoring to make it optional

@dataclass
class CalibrationArguments:
    """
    Calibration and data loading.
    """
    calibrator: str = field(
        default="default",
        metadata={"help": "Calibration plugin: default or imatrix."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face dataset identifier for calibration."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Which split of the dataset to load."},
    )
    batch_size: int = field(
        default=16, metadata={"help": "Batch size for calibration forward passes."}
    )
    num_batches: int = field(
        default=128,
        metadata={"help": "Number of batches to sample for calibration."},
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
