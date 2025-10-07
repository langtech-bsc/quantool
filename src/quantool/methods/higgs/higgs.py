from __future__ import annotations

import importlib

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, HiggsConfig

from quantool.core.base import BaseQuantizer
from quantool.core.registry import QuantizerRegistry
from quantool.core.meta import TemplateQuantizationCard


class HiggsPrecision(Enum):
    """Supported precision levels for HIGGS quantization."""

    B2 = 2
    B3 = 3
    B4 = 4

    def __str__(self) -> str:
        return f"{self.value}bit"

    @property
    def bits(self) -> int:
        return int(self.value)

    @classmethod
    def from_value(cls, value: Union[int, str, "HiggsPrecision", None]) -> "HiggsPrecision":
        """Resolve an incoming value into a :class:`HiggsPrecision` member."""

        if isinstance(value, HiggsPrecision):
            return value

        if value is None:
            return cls.B4

        if isinstance(value, int):
            for member in cls:
                if member.value == value:
                    return member
            raise ValueError(f"Unsupported HIGGS precision bits: {value}")

        normalized = str(value).strip().lower().replace("-", "").replace("_", "")

        # Allow values like "4bit", "bits4", "4", etc.
        if normalized.endswith("bit"):
            normalized = normalized[:-3]

        if normalized.startswith("bits"):
            normalized = normalized[4:]

        if normalized.isdigit():
            return cls.from_value(int(normalized))

        for member in cls:
            if normalized == str(member.value).lower():
                return member

        raise ValueError(f"Unrecognized HIGGS precision level: {value}")


@QuantizerRegistry.register
class Higgs(BaseQuantizer):
    """Quantizer that leverages the Transformers ``HiggsConfig`` and FLUTE backend."""

    name = "higgs"
    supported_levels = list(HiggsPrecision)
    template_card = TemplateQuantizationCard(
        title="HIGGS",
        description=(
            "HIGGS quantization using the FLUTE runtime with Hadamard preprocessing "
            "and MSE-optimal grids."
        ),
        hyperparameters={
            "method": "higgs",
            "bits": 4,
            "p": 2,
            "group_size": 256,
            "hadamard_size": 512,
        },
        intended_use=(
            "Zero-shot weight-only quantization for select large language models with "
            "runtime provided by FLUTE."
        ),
        limitations=(
            "Currently limited to specific model families (Llama 3/3.1 70B & 405B, "
            "Gemma 2 8B & 27B) and inference-only workloads."
        ),
        citations=[
            "https://arxiv.org/abs/2411.17525",
            "https://github.com/HanGuo97/flute",
        ],
    )

    def __init__(
        self,
        model_id: str,
        *,
        bits: int = 4,
        p: int = 2,
        group_size: int = 256,
        hadamard_size: int = 512,
        modules_to_not_convert: Optional[list[str]] = None,
        tune_metadata: Optional[Dict[str, Any]] = None,
        device_map: Union[str, Dict[str, Union[str, int]]] = "auto",
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id)

        self.logger.debug(
            "Initializing HIGGS quantizer with bits=%s, p=%s, group_size=%s",
            bits,
            p,
            group_size,
        )

        self.default_level = HiggsPrecision.from_value(bits)

        self.quant_config_defaults: Dict[str, Any] = {
            "bits": self.default_level.bits,
            "p": p,
            "group_size": group_size,
            "hadamard_size": hadamard_size,
            "modules_to_not_convert": modules_to_not_convert,
            "tune_metadata": tune_metadata,
        }

        self.load_defaults: Dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "use_auth_token": use_auth_token,
        }

        self.extra_init_kwargs = kwargs

        self.last_model = None
        self.last_tokenizer = None
        self.last_quantization_config: Optional[HiggsConfig] = None

        self._check_dependencies()

    # Helpers
    def _check_dependencies(self) -> None:
        try:
            importlib.import_module("flute")
            self.logger.info("FLUTE backend available for HIGGS quantization")
        except ImportError as e:
            raise RuntimeError(
                "HIGGS quantization requires the 'flute-kernel' package. "
                "Install it via 'pip install quantool[higgs]'"
            ) from e

    @staticmethod
    def _normalize_model_identifier(model: Union[str, Path, Any]) -> str:
        if isinstance(model, (str, Path)):
            return str(model)

        candidate = getattr(model, "name_or_path", None)
        if candidate:
            return candidate

        config = getattr(model, "config", None)
        candidate = getattr(config, "_name_or_path", None)
        if candidate:
            return candidate

        raise ValueError("Unable to resolve model identifier/path from provided 'model'.")

    def _resolve_level(self, level: Union[str, int, HiggsPrecision, None]) -> HiggsPrecision:
        if level is None:
            return self.default_level

        resolved = HiggsPrecision.from_value(level)
        self.logger.info("Using HIGGS precision level: %s", resolved)
        return resolved

    # Public API
    def quantize(
        self,
        model: Union[str, Path, Any],
        level: Union[str, int, HiggsPrecision, None] = None,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        tokenizer=None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Apply HIGGS quantization.

        Parameters
        ----------
        model:
            Model identifier (HF repo ID or local path) or an already loaded model instance.
        level:
            Desired precision level. Can be an instance of :class:`HiggsPrecision`,
            the string representation (e.g., ``"4bit"``), or the integer number of bits.
        output_dir:
            Optional directory where the resulting model should be written immediately.
            If omitted, defer saving to :meth:`save_pretrained` or :meth:`push_to_hub`.
        tokenizer:
            Optional tokenizer instance to attach. If ``None``, one will be loaded
            automatically when possible.
        tokenizer_kwargs:
            Additional keyword arguments forwarded to :func:`AutoTokenizer.from_pretrained`.
        **kwargs:
            Extra arguments forwarded to :func:`AutoModelForCausalLM.from_pretrained` or used
            to override default Higgs configuration (``bits``, ``p``, ``group_size``,
            ``hadamard_size``, ``modules_to_not_convert``, ``tune_metadata``).
        """

        tokenizer_kwargs = tokenizer_kwargs or {}

        # Split kwargs between HiggsConfig overrides and from_pretrained overrides
        quant_overrides: Dict[str, Any] = {}
        for key in [
            "bits",
            "p",
            "group_size",
            "hadamard_size",
            "modules_to_not_convert",
            "tune_metadata",
        ]:
            if key in kwargs:
                quant_overrides[key] = kwargs.pop(key)

        load_overrides: Dict[str, Any] = {}
        for key in [
            "device_map",
            "torch_dtype",
            "trust_remote_code",
            "revision",
            "cache_dir",
            "use_auth_token",
        ]:
            if key in kwargs:
                load_overrides[key] = kwargs.pop(key)

        resolved_level = self._resolve_level(quant_overrides.pop("bits", level))

        quant_config_kwargs = {**self.quant_config_defaults, **quant_overrides}
        quant_config_kwargs["bits"] = resolved_level.bits
        quant_config = HiggsConfig(
            **{k: v for k, v in quant_config_kwargs.items() if v is not None}
        )

        load_kwargs = {**self.load_defaults, **load_overrides, **self.extra_init_kwargs}
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
        load_kwargs.update(kwargs)

        model_id = self._normalize_model_identifier(model)
        self.source_model = model_id
        self.logger.info(
            "Starting HIGGS quantization | model=%s | config=%s | load_kwargs=%s",
            model_id,
            {k: v for k, v in quant_config_kwargs.items() if k != "tune_metadata"},
            {k: v for k, v in load_kwargs.items() if k != "use_auth_token"},
        )

        self.logger.info("Loading quantized model...")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            **load_kwargs,
        )
        self.logger.info("Quantized model loaded successfully.")

        self.logger.info("Attaching tokenizer...")
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    use_fast=True,
                    trust_remote_code=load_kwargs.get("trust_remote_code", True),
                    **tokenizer_kwargs,
                )
            except Exception as e:
                self.logger.warning("Failed to load tokenizer for %s: %s", model_id, e)
                tokenizer = None

        if tokenizer is not None:
            setattr(quantized_model, "tokenizer", tokenizer)

        self.last_model = quantized_model
        self.last_tokenizer = tokenizer
        self.last_quantization_config = quant_config

        if output_dir is not None:
            self.logger.info("Saving quantized model directly to %s", output_dir)
            self._save_model_files(output_dir)

        return quantized_model

    # ExportMixin hooks
    def _save_model_files(self, save_directory: Union[str, Path]):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        model = getattr(self, "last_model", None)
        if model is None:
            self.logger.warning("No cached HIGGS model found; performing default quantization.")
            source = getattr(self, "source_model", self.model_id)
            model = self.quantize(source, level=self.default_level)

        model.save_pretrained(save_dir)

        tokenizer = getattr(model, "tokenizer", None) or self.last_tokenizer
        if tokenizer is None:
            model_id = getattr(model, "name_or_path", None) or getattr(
                getattr(model, "config", None),
                "_name_or_path",
                None,
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id or self.model_id,
                    use_fast=True,
                    trust_remote_code=True,
                )
            except Exception as e:
                self.logger.warning("Unable to save tokenizer: %s", e)
                tokenizer = None

        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)