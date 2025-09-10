from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
from pathlib import Path
from quantool.core.base import BaseQuantizer
from quantool.core.registry import QuantizerRegistry
from quantool.core.meta import TemplateQuantizationCard


@QuantizerRegistry.register
class Higgs(BaseQuantizer):
    """
    Quantizer for HIGGS-based models.
    HIGGS is a zero-shot quantization algorithm that combines Hadamard preprocessing with MSE-Optimal quantization grids to achieve lower quantization error and state-of-the-art performance.
    """
    name = "higgs"
    supported_levels = ["4bit", "8bit"]
    template_card = TemplateQuantizationCard(
        title="HIGGS",
        description="HIGGS quantization method for efficient model inference",
        hyperparameters={
            "bits": 4,
            "method": "higgs"
        },
        intended_use="Efficient inference for large language models",
        limitations="Currently optimized for specific model architectures",
        citations=["https://arxiv.org/abs/2411.17525"]
    )
    

    def quantize(self, model, level: str = "4bit", **kwargs):
        """
        Apply HIGGS quantization at specified level.
        """
        bits = 4 if level == "4bit" else 8
        print(f"Quantizing model to {bits}-bit using HIGGS method.")

        # Resolve model id/path from either a string or a loaded model
        model_id = (
            model
            if isinstance(model, str)
            else getattr(model, "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None)
        )
        if not model_id:
            raise ValueError("Unable to resolve model identifier/path from input 'model'.")

        try:
            from transformers import HiggsConfig as _HiggsConfig
        except Exception as e:
            raise ImportError(
                "HiggsConfig is not available. Ensure the HIGGS backend is installed "
                "and exposes 'HiggsConfig' importable as quantool.methods.higgs.config.HiggsConfig "
                "or transformers.HiggsConfig."
            ) from e

        higgs_cfg = _HiggsConfig(bits=bits, **kwargs.pop("higgs_config_kwargs", {}))

        device_map = kwargs.pop("device_map", "auto")
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        torch_dtype = kwargs.pop("torch_dtype", None)

        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=higgs_cfg,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            **kwargs
        )

        # Best-effort: load and attach tokenizer for export convenience
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, use_fast=True, trust_remote_code=trust_remote_code
                )
            except Exception:
                tokenizer = None
        if tokenizer is not None:
            setattr(quantized_model, "tokenizer", tokenizer)

        # Cache last quantized model for ExportMixin
        self.last_model = quantized_model

        return quantized_model

    def _save_model_files(self, save_directory: Union[str, Path]):
        """
        Save the last HIGGS-quantized transformers model and its tokenizer.
        If no cached model is available, perform a default quantization first.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        model = getattr(self, "last_model", None)
        if model is None:
            self.logger.warning("No cached HIGGS model found; performing default quantization.")
            # Derive default level from template hyperparameters (bits)
            bits = int(self.template_card.hyperparameters.get("bits", 4))
            default_level = "4bit" if bits <= 4 else "8bit"
            source = getattr(self, "source_model", self.model_id)
            model = self.quantize(source, level=default_level)

        # Save model
        model.save_pretrained(save_dir)

        # Save tokenizer if present; otherwise try to load it
        tok = getattr(model, "tokenizer", None)
        if tok is None:
            model_id = getattr(model, "name_or_path", None) or getattr(getattr(model, "config", None), "_name_or_path", None) or self.model_id
            try:
                tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
            except Exception:
                tok = None
        if tok is not None:
            tok.save_pretrained(save_dir)