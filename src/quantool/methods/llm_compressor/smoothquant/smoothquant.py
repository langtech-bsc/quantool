"""SmoothQuant via llm-compressor."""
from typing import Any, Dict, List, Optional, Tuple

from quantool.core.meta import TemplateQuantizationCard
from quantool.core.registry import QuantizerRegistry
from quantool.methods.llm_compressor.base import LLMCompressorQuantizer, RecipeType


@QuantizerRegistry.register
class SmoothQuant(LLMCompressorQuantizer):
    """SmoothQuant quantizer using llm-compressor backend.
    
    SmoothQuant migrates quantization difficulty from activations to weights by
    smoothing activation outliers, enabling efficient W8A8 quantization.
    
    Note: SmoothQuant is a two-stage process that combines:
        1. SmoothQuantModifier - Pre-processing to smooth activation outliers
        2. GPTQModifier - Actual quantization of weights and activations
    
    This is why SmoothQuant only supports schemes with quantized activations
    (W8A8, W4A8) - the smoothing step is only beneficial when quantizing activations.
    """
    
    name = "smoothquant"
    supported_levels = [
        "W8A8",       # 8-bit weights, 8-bit activations (int8)
        "INT8",       # Alias for W8A8
        "W4A8",       # 4-bit weights, 8-bit activations
    ]
    
    template_card = TemplateQuantizationCard(
        title="SmoothQuant",
        description="Smoothing-based activation quantization for W8A8",
        hyperparameters={
            "method": "smoothquant",
            "scheme": "W8A8",
            "smoothing_strength": 0.5,
            "targets": "Linear",
            "ignore": ["lm_head"],
            "num_calibration_samples": 512,
        },
        intended_use="W8A8 quantization with activation smoothing for better accuracy",
        limitations="Requires calibration dataset; best for W8A8 schemes",
        citations=["https://arxiv.org/abs/2211.10438"],
    )
    
    def _build_recipe(
        self, level: Optional[str], method_kwargs: Dict[str, Any]
    ) -> Tuple[RecipeType, str]:
        """Build SmoothQuant + quantization recipe."""
        try:
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
            from llmcompressor.modifiers.quantization import GPTQModifier
            from compressed_tensors.quantization import is_preset_scheme
        except ImportError as exc:
            raise ImportError(
                "SmoothQuant modifiers not available. Ensure llmcompressor is installed."
            ) from exc
        
        scheme = level or method_kwargs.get("scheme", "W8A8")
        
        # Validate scheme against compressed-tensors presets
        if not is_preset_scheme(scheme):
            raise ValueError(
                f"Scheme '{scheme}' is not a valid compressed-tensors preset scheme. "
                f"Valid schemes include: W8A16, W4A16, W4A16_ASYM, W8A8, INT8, W4A8, "
                f"FP8, FP8_DYNAMIC, FP8_BLOCK, NVFP4A16, NVFP4, UNQUANTIZED"
            )
        smoothing_strength = method_kwargs.get("smoothing_strength", 0.5)
        
        # SmoothQuant is a two-stage process:
        # 1. SmoothQuantModifier smooths activation outliers
        # 2. GPTQModifier quantizes the smoothed weights and activations
        # This combination provides better accuracy for W8A8/W4A8 schemes
        recipe: List[Any] = [
            SmoothQuantModifier(smoothing_strength=smoothing_strength),
            GPTQModifier(
                scheme=scheme,
                targets=method_kwargs.get("targets", "Linear"),
                ignore=method_kwargs.get("ignore", ["lm_head"]),
            ),
        ]
        
        self.logger.info(
            f"Built SmoothQuant recipe with scheme={scheme}, smoothing_strength={smoothing_strength}"
        )
        
        return recipe, scheme
