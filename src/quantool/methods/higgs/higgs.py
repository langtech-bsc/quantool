from transformers import AutoModelForCausalLM, AutoTokenizer, HiggsConfig
from quantool.core.base import BaseQuantizer
from quantool.core.registry import QuantizerRegistry
from quantool.core.meta import TemplateQuantizationCard


@QuantizerRegistry.register
class Higgs(BaseQuantizer):
    """
    Quantizer for HIGGS-based models.
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
        citations=["https://github.com/huggingface/transformers"]
    )

    def quantize(self, model, level: str = "4bit", **kwargs):
        """
        Apply HIGGS quantization at specified level.
        """
        bits = 4 if level == "4bit" else 8
        
        print(f"Quantizing model to {bits}-bit using HIGGS method.")
        
        # Apply HIGGS quantization config
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model.name_or_path if hasattr(model, "name_or_path") else "model",
            quantization_config=HiggsConfig(bits=bits),
            device_map="auto",
            **kwargs
        )
        
        return quantized_model

    def export(self, model, out_path: str):
        """
        Save quantized model.
        """
        print(f"Exporting HIGGS-quantized model to {out_path}")
        model.save_pretrained(out_path)
        
        # If tokenizer is available, save it too
        if hasattr(model, "tokenizer"):
            model.tokenizer.save_pretrained(out_path)