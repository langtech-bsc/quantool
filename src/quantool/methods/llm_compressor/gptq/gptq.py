"""GPTQ quantization via llm-compressor."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from quantool.core.meta import TemplateQuantizationCard
from quantool.core.registry import QuantizerRegistry
from quantool.methods.llm_compressor.base import LLMCompressorQuantizer, RecipeType


@QuantizerRegistry.register
class GPTQ(LLMCompressorQuantizer):
    """GPTQ quantizer using llm-compressor backend.
    
    GPTQ (Generative Pre-trained Transformer Quantization) is a post-training
    quantization method that uses calibration data to minimize quantization error.
    Supports weight-only and weight+activation quantization.
    """
    
    name = "gptq"
    supported_levels = [
        "W4A16",      # 4-bit weights, 16-bit activations
        "W8A8",       # 8-bit weights, 8-bit activations (int8)
        "W8A8_FP8",   # 8-bit weights, 8-bit activations (fp8)
        "W8A16",      # 8-bit weights, 16-bit activations
        "W4A16_ASYM", # 4-bit asymmetric weights
        "W8A8_INT8",  # Explicit int8 scheme
    ]
    
    template_card = TemplateQuantizationCard(
        title="GPTQ Quantization",
        description="Post-training quantization using GPTQ algorithm with calibration data",
        hyperparameters={
            "method": "gptq",
            "scheme": "W4A16",
            "targets": "Linear",
            "ignore": ["lm_head"],
            "num_calibration_samples": 512,
        },
        intended_use="Efficient inference for LLMs with minimal accuracy loss",
        limitations="Requires calibration dataset; quantization time scales with model size",
        citations=["https://arxiv.org/abs/2210.17323"],
    )
    
    def _build_recipe(
        self, level: Optional[str], method_kwargs: Dict[str, Any]
    ) -> Tuple[RecipeType, str]:
        """Build GPTQ recipe from level and kwargs."""
        try:
            from llmcompressor.modifiers.quantization import GPTQModifier
        except ImportError as exc:
            raise ImportError(
                "GPTQModifier not available. Ensure llmcompressor is installed correctly."
            ) from exc
        
        # Determine quantization scheme
        scheme = level or method_kwargs.get("scheme", "W4A16")
        if scheme not in self.supported_levels:
            self.logger.warning(
                f"Level '{scheme}' not in supported list, using anyway: {self.supported_levels}"
            )
        
        # Build modifier kwargs
        modifier_kwargs = {
            "scheme": scheme,
            "targets": method_kwargs.get("targets", "Linear"),
            "ignore": method_kwargs.get("ignore", ["lm_head"]),
        }
        
        # Pass through additional GPTQ-specific parameters
        for key in ["block_size", "dampening_frac", "sequential_targets"]:
            if key in method_kwargs:
                modifier_kwargs[key] = method_kwargs[key]
        
        recipe = GPTQModifier(**modifier_kwargs)
        self.logger.info(f"Built GPTQ recipe with scheme={scheme}, targets={modifier_kwargs['targets']}")
        
        return recipe, scheme