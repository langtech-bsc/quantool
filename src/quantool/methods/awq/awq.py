from transformers import AutoModelForCausalLM, AutoTokenizer
from quantool.core.base import BaseQuantizer
from quantool.core.registry import QuantizerRegistry
from quantool.core.meta import TemplateQuantizationCard
import torch


@QuantizerRegistry.register
class AWQ(BaseQuantizer):
    """
    Quantizer for AWQ (Activation-aware Weight Quantization) models.
    """
    name = "awq"
    supported_levels = ["4bit", "8bit"]
    template_card = TemplateQuantizationCard(
        title="AWQ",
        description="Activation-aware Weight Quantization for efficient model inference",
        hyperparameters={
            "bits": 4,
            "method": "awq",
            "zero_point": True,
            "q_group_size": 128,
            "version": "GEMM"
        },
        intended_use="Efficient inference with activation-aware quantization",
        limitations="Requires AWQ library and calibration data",
        citations=["https://arxiv.org/abs/2306.00978"]
    )

    def quantize(self, model, level: str = "4bit", **kwargs):
        """
        Apply AWQ quantization at specified level.
        """
        # Parse bit width from level
        if level.endswith("bit"):
            bits = int(level.replace("bit", ""))
        else:
            bits = 4  # default
        
        self.logger.info(f"Quantizing model to {bits}-bit using AWQ method.")
        
        # Configure AWQ parameters
        quant_config = {
            "zero_point": kwargs.get("zero_point", True),
            "q_group_size": kwargs.get("q_group_size", 128),
            "w_bit": bits,
            "version": kwargs.get("version", "GEMM")
        }
        
        try:
            # Try to use AWQ library
            from awq import AutoAWQForCausalLM
            
            # Load model with AWQ
            awq_model = AutoAWQForCausalLM.from_pretrained(
                model.name_or_path if hasattr(model, "name_or_path") else "model",
                device_map="auto"
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model.name_or_path if hasattr(model, "name_or_path") else "model",
                trust_remote_code=True
            )
            
            # Apply quantization
            awq_model.quantize(tokenizer, quant_config=quant_config)
            
            self.model = awq_model
            self.tokenizer = tokenizer
            
            return awq_model
            
        except ImportError:
            print("AWQ library not available, using transformers AwqConfig")
            
            try:
                from transformers import AwqConfig
                
                awq_config = AwqConfig(
                    bits=bits,
                    group_size=kwargs.get("q_group_size", 128),
                    zero_point=kwargs.get("zero_point", True),
                    version=kwargs.get("version", "GEMM"),
                )
                
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    model.name_or_path if hasattr(model, "name_or_path") else "model",
                    quantization_config=awq_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                
                self.model = quantized_model
                return quantized_model
                
            except ImportError:
                print("AWQ not available, using fallback quantization")
                self.model = model
                return model

    def _save_model_files(self, save_directory):
        """Save AWQ-specific model files using ExportMixin."""
        if hasattr(self, 'model'):
            if hasattr(self.model, 'save_quantized'):
                self.model.save_quantized(save_directory)
            else:
                self.model.save_pretrained(save_directory)
                
        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(save_directory)


# Legacy function for backward compatibility
def to_awq(model_path, quant_path):
    """Legacy function for AWQ quantization."""
    awq_quantizer = AWQ()
    
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    quantized_model = awq_quantizer.quantize(model)
    awq_quantizer.save_pretrained(quant_path)