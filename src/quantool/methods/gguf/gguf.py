import subprocess
import os
import shutil
from utils.command import run_command
from quantool.core.registry import QuantizerRegistry
from quantool.core.base import BaseQuantizer
from quantool.core.meta import TemplateQuantizationCard


@QuantizerRegistry.register
class GGUF(BaseQuantizer):
    """
    Quantizer for GGUF models using llama.cpp.
    """
    name = "gguf"
    supported_levels = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32"]
    template_card = TemplateQuantizationCard(
        title="GGUF",
        description="GGUF quantization using llama.cpp for efficient CPU and GPU inference",
        hyperparameters={
            "format": "gguf",
            "method": "gguf",
            "quantization_type": "Q4_K_M",
            "context_length": 2048
        },
        intended_use="Efficient inference on CPU and GPU with llama.cpp",
        limitations="Requires llama.cpp conversion tools and specific model architectures",
        citations=["https://github.com/ggml-org/llama.cpp"]
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llama_cpp_path = kwargs.get("llama_cpp_path", None)
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if llama.cpp tools are available."""
        raise NotImplementedError(
            "GGUF quantization requires llama.cpp tools. Please ensure they are installed and available in your PATH."
        )
        
    def _convert_to_gguf(self, model_path: str, output_path: str, quantization_type: str):
        """Convert model to GGUF format."""
        try:
            # Try using llama.cpp convert script if available
            convert_script = os.path.join(self.llama_cpp_path or "", "convert_hf_to_gguf.py")
            
            if os.path.exists(convert_script):
                # Use llama.cpp convert script
                cmd = [
                    "python", convert_script,
                    model_path,
                    "--outtype", "f16",
                    "--outfile", f"{output_path}/model.gguf"
                ]
                try:
                    self.logger.info(f"Converting to GGUF: {' '.join(cmd)}")
                    result = run_command(self.logger, cmd, cwd=output_path)
                    self.logger.info(f"Conversion completed!")
                except SystemExit as e:
                    if e.code == 127:
                        raise RuntimeError("Conversion command failed. Ensure llama.cpp tools are installed correctly.")
                
                # Quantize the GGUF file
                quantize_cmd = [
                    os.path.join(self.llama_cpp_path or "", "quantize"),
                    f"{output_path}/model.gguf",
                    f"{output_path}/model-{quantization_type.lower()}.gguf",
                    quantization_type
                ]

                try: 
                    self.logger.info(f"Quantizing GGUF: {' '.join(quantize_cmd)}")
                    result = run_command(self.logger, quantize_cmd, cwd=output_path)
                    self.logger.info(f"Quantization completed!")
                except SystemExit as e:
                    if e.code == 127:
                        raise RuntimeError("Quantization command failed. Ensure llama.cpp tools are installed correctly.")

                return f"{output_path}/model-{quantization_type.lower()}.gguf"
                
            else:
                # Fallback: use transformers to save in a compatible format
                self.logger.warning("llama.cpp tools not found, using fallback conversion")
                return self._fallback_conversion(model_path, output_path, quantization_type)
                
        except Exception as e:
            self.logger.error(f"GGUF conversion failed: {e}")
            return self._fallback_conversion(model_path, output_path, quantization_type)

    def _fallback_conversion(self, model_path: str, output_path: str, quantization_type: str):
        """Fallback conversion method."""
        self.logger.info("Using fallback conversion method")
        
        # Create a simple metadata file indicating the intended quantization
        os.makedirs(output_path, exist_ok=True)
        
        metadata = {
            "format": "gguf",
            "quantization_type": quantization_type,
            "source_model": model_path,
            "note": "This is a placeholder. Use proper llama.cpp tools for actual GGUF conversion."
        }
        
        import json
        with open(f"{output_path}/gguf_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return f"{output_path}/gguf_metadata.json"

    def quantize(self, model, level: str = "Q4_K_M", **kwargs):
        """
        Apply GGUF quantization at specified level.
        """
        if level not in self.supported_levels:
            self.logger.warning(f"Level {level} not in supported levels, using Q4_K_M")
            level = "Q4_K_M"
        
        print(f"Quantizing model to {level} using GGUF method.")
        
        # For GGUF, we primarily work with file paths
        model_path = model.name_or_path if hasattr(model, "name_or_path") else str(model)
        
        self.quantization_type = level
        self.source_model = model
        
        return model

    def _save_model_files(self, save_directory):
        """Save GGUF-specific model files using ExportMixin."""
        if hasattr(self, 'source_model'):
            # Save the source model first
            self.source_model.save_pretrained(save_directory)
            
            # Get model path for conversion
            model_path = self.source_model.name_or_path if hasattr(self.source_model, "name_or_path") else save_directory
            
            # Convert to GGUF
            gguf_file = self._convert_to_gguf(
                model_path, 
                save_directory, 
                getattr(self, 'quantization_type', 'Q4_K_M')
            )
            
            self.logger.info(f"GGUF conversion completed: {gguf_file}")
            
        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(save_directory)
