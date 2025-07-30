import os
import shutil
import sys
import tempfile
from idlelib.window import registry
from pathlib import Path
from typing import List, Optional, Union

from quantool.utils import run_command
from quantool.core.registry import QuantizerRegistry
from quantool.core.base import BaseQuantizer
from quantool.core.meta import TemplateQuantizationCard
from enum import Enum


class QuantType(Enum):
    Q2_K = "Q2_K"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "f16"
    F32 = "f32"

    def __str__(self):
        return self.value


@QuantizerRegistry.register
class GGUF(BaseQuantizer):
    name = "gguf"
    supported_levels = list(QuantType)

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

    CONVERT_OUTTYPES = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}

    def __init__(
            self,
            *args,
            llama_cpp_path: Optional[Union[str, Path]] = None,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.llama_cpp_path = Path(llama_cpp_path) if llama_cpp_path else None
        # TODO: add optional path for executable
        self.use_module_import = False  # Flag to track if we should use module import
        self._check_dependencies()

    def _check_dependencies(self):
        self.logger.info("llama.cpp path: %s", self.llama_cpp_path)
        self._locate_converter_script()
        self._locate_quantize_binary()
        
        if not self.use_module_import:
            self.logger.info(f"Found converter: {self.convert_script}")
        self.logger.info(f"Found quantizer: {self.quantize_bin}")

    def _locate_converter_script(self):
        script = (self.llama_cpp_path / "convert_hf_to_gguf.py") if self.llama_cpp_path else Path(
            "convert_hf_to_gguf.py")

        if not script.exists():
            try:
                self.logger.info("convert_hf_to_gguf.py not found as file, using module import instead")
                self.use_module_import = True
                self.convert_script = None
            except ImportError:
                raise RuntimeError(f"convert_hf_to_gguf.py not found at {script} and module import failed.")
        else:
            self.convert_script = script
            self.use_module_import = False

    def _locate_quantize_binary(self):
        bin_path = (self.llama_cpp_path / "llama-quantize") if self.llama_cpp_path else None
        if bin_path and bin_path.exists():
            quantize_bin = bin_path
        else:
            alt = shutil.which("llama-quantize") or shutil.which("quantize")
            if not alt:
                raise RuntimeError("Could not find llama-quantize in PATH.")
            quantize_bin = Path(alt)

        self.quantize_bin = quantize_bin

    def _ensure_output_directory(self, output_dir: Optional[Union[str, Path]]) -> Path:
        output_path = Path(output_dir or tempfile.mkdtemp())
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _convert_hf(self, model_path: str, output_path: str, outtype: str) -> str:
        out_file = Path(output_path) / f"model.{outtype}.gguf"

        if self.use_module_import:
            # If we can't find the script, use module import
            import convert_hf_to_gguf

            # Save original sys.argv
            original_argv = sys.argv.copy()

            try:
                sys.argv = [
                    "convert_hf_to_gguf.py",  # this placeholder becomes sys.argv[0]
                    model_path,  # input model path
                    "--outfile", str(out_file),  # output filename
                    "--outtype", outtype,  # quantization type
                ]
                self.logger.info(f"Running converter via module import with args: {' '.join(sys.argv[1:])}")
                convert_hf_to_gguf.main()
            finally:
                # Restore original sys.argv
                sys.argv = original_argv
        else:
            # Use subprocess approach with script file
            cmd = [
                sys.executable, str(self.convert_script),
                model_path,
                "--outfile", str(out_file),
                "--outtype", outtype
            ]
            self.logger.info(f"Running converter: {' '.join(cmd)}")
            run_command(self.logger, cmd)

        return str(out_file)

    def _quantize_gguf(self, input_gguf: str, output_path: str, quant: str) -> str:
        model_name = Path(self.model_id).name if isinstance(self.model_id, str) else str(self.model_id)
        self.logger.info(f"Model name extracted: {model_name}")
        out_file = Path(output_path) / f"{model_name}-{quant}.gguf"
        cmd = [
            str(self.quantize_bin),
            input_gguf,
            str(out_file),
            quant
        ]
        self.logger.info(f"Running quantizer: {' '.join(cmd)}")
        run_command(self.logger, cmd)
        return str(out_file)
    
    def _validate_and_convert_level(self, level) -> QuantType:
        if not isinstance(level, QuantType):
            try:
                level = QuantType[level]
            except KeyError:
                try:
                    level = QuantType(level)
                except ValueError:
                    level = QuantType.Q4_K_M
                    self.logger.warning(f"Invalid quantization level '{level}', defaulting to Q4_K_M.")
        return level
    
    def quantize(
            self,
            model: Union[str, Path],
            level: QuantType = QuantType.Q4_K_M,
            output_dir: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> str:
        level = self._validate_and_convert_level(level)
        output_path = self._ensure_output_directory(output_dir)

        model_path = getattr(model, "name_or_path", str(model))
        self.logger.info(f"Quantizing model: {self.model_id} at level {level}")

        lvl = str(level)

        # If converter supports this outtype, do it directly
        if lvl in self.CONVERT_OUTTYPES:
            final = self._convert_hf(model_path, output_path, lvl)
        else:
            # First convert to FP16
            base = self._convert_hf(model_path, output_path, "f16")
            # Then apply k-quant or i-quant
            final = self._quantize_gguf(base, output_path, lvl)

        self.logger.info(f"Quantization complete: {final}")
        self.last_gguf = final
        return final

    def _save_model_files(self, save_directory: Union[str, Path]):
        if hasattr(self, "last_gguf"):
            self.logger.info(f"Saving last GGUF file: {self.last_gguf}")
            shutil.copy(self.last_gguf, save_directory)
        else:
            # Fall back to default quantization
            self.logger.warning("No GGUF file found, using default quantization method.")
            self.quantize(self.source_model, QuantType(self.template_card.hyperparameters["quantization_type"]),
                          save_directory)