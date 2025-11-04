"""AWQ (Activation-Aware Weight Quantization) via llm-compressor."""

from typing import Any, Dict, Optional, Tuple

from quantool.core.meta import TemplateQuantizationCard
from quantool.core.registry import QuantizerRegistry
from quantool.methods.llm_compressor.base import (LLMCompressorQuantizer,
                                                  RecipeType)


@QuantizerRegistry.register
class AWQ(LLMCompressorQuantizer):
    """AWQ quantizer using llm-compressor backend.

    Activation-Aware Weight Quantization (AWQ) protects salient weights based on
    activation patterns observed from calibration data.
    """

    name = "awq"
    supported_levels = [
        "W4A16",
        "W4A16_ASYM",
        "W8A16",
    ]

    template_card = TemplateQuantizationCard(
        title="AWQ Quantization",
        description="Activation-aware weight quantization preserving salient weights",
        hyperparameters={
            "method": "awq",
            "scheme": "W4A16",
            "targets": "Linear",
            "ignore": ["lm_head"],
            "num_calibration_samples": 512,
        },
        intended_use="Weight-only quantization with better accuracy than naive PTQ",
        limitations="Requires calibration dataset; weight-only (activations remain fp16)",
        citations=["https://arxiv.org/abs/2306.00978"],
    )

    def _build_recipe(
        self, level: Optional[str], method_kwargs: Dict[str, Any]
    ) -> Tuple[RecipeType, str]:
        """Build AWQ recipe from level and kwargs."""
        try:
            from compressed_tensors.quantization import is_preset_scheme
            from llmcompressor.modifiers.awq import AWQModifier
        except ImportError as exc:
            raise ImportError(
                "AWQModifier not available. Ensure llmcompressor is installed correctly."
            ) from exc

        scheme = level or method_kwargs.get("scheme", "W4A16")

        # Validate scheme against compressed-tensors presets
        if not is_preset_scheme(scheme):
            raise ValueError(
                f"Scheme '{scheme}' is not a valid compressed-tensors preset scheme. "
                f"Valid schemes include: W8A16, W4A16, W4A16_ASYM, W8A8, INT8, W4A8, "
                f"FP8, FP8_DYNAMIC, FP8_BLOCK, NVFP4A16, NVFP4, UNQUANTIZED"
            )

        # AWQ only supports weight-only quantization with 16-bit activations
        if scheme not in self.supported_levels:
            self.logger.warning(
                f"AWQ only supports weight-only quantization with 16-bit activations. "
                f"Scheme '{scheme}' may not be compatible. Supported: {self.supported_levels}"
            )

        modifier_kwargs = {
            "scheme": scheme,
            "targets": method_kwargs.get("targets", "Linear"),
            "ignore": method_kwargs.get("ignore", ["lm_head"]),
        }

        # AWQ-specific parameters
        for key in ["mappings", "smoothing_strength"]:
            if key in method_kwargs:
                modifier_kwargs[key] = method_kwargs[key]

        recipe = AWQModifier(**modifier_kwargs)
        self.logger.info(f"Built AWQ recipe with scheme={scheme}")

        return recipe, scheme
