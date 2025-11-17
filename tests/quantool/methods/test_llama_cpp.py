from pathlib import Path
from unittest.mock import patch

from quantool.core.registry import QuantizerRegistry
from quantool.methods.llama_cpp.llama_cpp import GGUF, QuantType


class TestGGUFQuantizer:
	"""Test suite for the GGUF quantizer."""

	def test_registration(self):
		"""Ensure GGUF quantizer is registered in the registry."""
		assert "gguf" in QuantizerRegistry.list()

	def test_quantize_multiple_levels_returns_list(self, tmp_path):
		"""Quantizing with multiple levels should return one artifact per level."""

		with patch.object(GGUF, "_check_dependencies"):
			quantizer = QuantizerRegistry.create("gguf", model_id="test/model")

		output_dir = tmp_path / "gguf"
		levels = ["Q4_K_M", QuantType.Q3_K_S]

		convert_calls = []
		quantize_calls = []

		def fake_convert(model_path, out_path, outtype):
			convert_calls.append((model_path, out_path, outtype))
			assert outtype == "f16"
			return "base_f16.gguf"

		def fake_quantize(base_file, out_path, quant_level):
			quantize_calls.append((base_file, out_path, quant_level))
			return str(Path(out_path) / f"model-{quant_level}.gguf")

		with (
			patch.object(quantizer, "_ensure_output_directory", return_value=output_dir) as mock_ensure,
			patch.object(quantizer, "_convert_hf", side_effect=fake_convert) as mock_convert,
			patch.object(quantizer, "_quantize_gguf", side_effect=fake_quantize) as mock_quantize,
		):
			artifacts = quantizer.quantize(model="hf-repo/model", level=levels)

		expected_artifacts = [
			str(output_dir / "model-Q4_K_M.gguf"),
			str(output_dir / "model-Q3_K_S.gguf"),
		]

		assert artifacts == expected_artifacts
		assert quantizer.last_gguf == expected_artifacts

		assert mock_ensure.called
		mock_convert.assert_called_once()
		assert convert_calls == [("hf-repo/model", output_dir, "f16")]

		assert mock_quantize.call_count == 2
		assert quantize_calls == [
			("base_f16.gguf", output_dir, "Q4_K_M"),
			("base_f16.gguf", output_dir, "Q3_K_S"),
		]
