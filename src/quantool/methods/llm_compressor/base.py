from __future__ import annotations

import inspect
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from llmcompressor import logger as llmcompressor_logger
from quantool.core.helpers import LoggerFactory

# Configure llmcompressor's logger to use quantool's format
LoggerFactory.configure_external_logger(
    llmcompressor_logger,
    level="INFO",
    fmt="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level: <8}</level> <magenta>[{name}:{module}:{function}:{line}]</magenta> <level>{message}</level>"
)

# Get our own logger for this module
logger = LoggerFactory.get_logger(__name__)

from quantool.core.base import BaseQuantizer


RecipeType = Union[Any, List[Any]]  # llmcompressor recipe can be a single modifier or a list


class LLMCompressorQuantizer(BaseQuantizer):
    """Shared logic for quantizers backed by llm-compressor."""

    # Cache for oneshot parameter names (extracted once from function signature)
    _ONESHOT_PARAMS_CACHE: Optional[set] = None

    def __init__(self, model_id, *args, **kwargs):
        super().__init__(model_id)
        self.last_output_dir: Optional[Path] = None
        self.last_model = None
        self.last_tokenizer = None
        self.source_model = None
        self._last_recipe: Optional[RecipeType] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @classmethod
    def _get_oneshot_params(cls) -> set:
        """Extract parameter names from llmcompressor.oneshot function signature.
        
        This dynamically inspects the oneshot function to get all valid parameters,
        avoiding the need to maintain a hardcoded list that could become outdated.
        
        Returns:
            Set of parameter names accepted by llmcompressor.oneshot
        """
        if cls._ONESHOT_PARAMS_CACHE is None:
            try:
                from llmcompressor import oneshot
                sig = inspect.signature(oneshot)
                # Get all parameter names except 'self' if it exists
                cls._ONESHOT_PARAMS_CACHE = {
                    param_name for param_name in sig.parameters.keys()
                    if param_name != 'self'
                }
            except Exception as e:
                logger.warning(f"Could not extract oneshot parameters: {e}. Using empty set.")
                cls._ONESHOT_PARAMS_CACHE = set()
        
        return cls._ONESHOT_PARAMS_CACHE
    
    def quantize(
        self,
        model: Union[str, Path, Any],
        level: Optional[str] = None,
        recipe: Optional[RecipeType] = None,
        oneshot_kwargs: Optional[Dict[str, Any]] = None,
        method_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Run llm-compressor oneshot flow and return the output directory path.

        Parameters
        ----------
        model:
            Hugging Face model identifier/path or an already loaded model instance.
        level:
            Quantization level hint (e.g. "W4A16"). When provided, subclasses can use
            it as default scheme.
        recipe:
            Optional explicit recipe (modifier instance, list of modifiers, or path).
        oneshot_kwargs:
            Optional dict of arguments forwarded to ``llmcompressor.oneshot``.
        method_kwargs:
            Optional dict of modifier-specific overrides consumed by subclasses.
        **kwargs:
            Convenience overrides merged into ``oneshot_kwargs`` (for calibration
            dataset, etc.) or ``method_kwargs`` when prefixed accordingly.
        """

        oneshot_kwargs = dict(oneshot_kwargs or {})
        method_kwargs = dict(method_kwargs or {})

        # Get valid oneshot parameters dynamically from function signature
        valid_oneshot_params = self._get_oneshot_params()
        
        # Allow top-level convenience keys to flow into oneshot kwargs automatically
        # if they match oneshot function parameters
        for key in list(kwargs.keys()):
            if key in valid_oneshot_params:
                oneshot_kwargs.setdefault(key, kwargs.pop(key))

        # Allow "method_kwargs__foo" style to populate method_kwargs automatically
        for key in list(kwargs.keys()):
            if key.startswith("method_kwargs__"):
                target_key = key.split("__", 1)[1]
                method_kwargs[target_key] = kwargs.pop(key)

        if recipe is None:
            recipe, inferred_level = self._build_recipe(level, method_kwargs)
        else:
            inferred_level = level or getattr(self, "default_level", "default")

        self._last_recipe = recipe

        oneshot_fn = self._import_llmcompressor_oneshot()
        oneshot_kwargs = self._prepare_oneshot_kwargs(model, oneshot_kwargs, inferred_level)
        oneshot_kwargs.setdefault("recipe", recipe)

        if not self._has_calibration_source(oneshot_kwargs):
            raise ValueError(
                "llm-compressor integrations require calibration data. "
                "Provide `dataset`, `dataset_path`, or a custom `calibration_dataloader` "
                "through `oneshot_kwargs`."
            )

        self.logger.info(
            f"Running llm-compressor oneshot with output_dir={oneshot_kwargs.get('output_dir')}"
        )

        # Store source model reference
        self.source_model = model
        
         # Try to get tokenizer if available
        self.logger.info("Trying to load tokenizer...")
        tokenizer_name = oneshot_kwargs.get("tokenizer")
        if self.last_tokenizer is None:
            from transformers import AutoTokenizer
            if tokenizer_name is None and isinstance(model, str):
                tokenizer_name = model
            self.last_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            self.last_tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
            self.logger.info("Loaded tokenizer from model_id")
        # Run oneshot - it returns the quantized model
        try:
            self.logger.info("Building calibration dataset...")
            ds = self._build_calibration_data(oneshot_kwargs["dataset"], oneshot_kwargs["num_calibration_samples"], oneshot_kwargs["max_seq_length"], oneshot_kwargs.get("shuffle_calibration_samples", True))
            oneshot_kwargs["dataset"] = ds
            self.logger.info("Calibration dataset built successfully.")

            self.last_model = oneshot_fn(**oneshot_kwargs)
        except Exception as e:
            self.logger.error(f"llm-compressor oneshot failed: {e}")
            raise e
        self.logger.info("llm-compressor oneshot completed successfully.")
        self.last_output_dir = Path(oneshot_kwargs["output_dir"]).resolve()
        
        self.logger.info(f"Quantization complete. Model ready at: {self.last_output_dir}")

        return str(self.last_output_dir)

    # ExportMixin hooks
    def _save_model_files(self, save_directory: Union[str, Path]):
        """Save quantized model and tokenizer using their native save_pretrained methods."""
        if not self.last_model:
            raise RuntimeError(
                "No quantized model available. Call `quantize()` before saving."
            )

        dest = Path(save_directory)
        dest.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving quantized model to: {dest}")
        
        # Save model using its native save_pretrained method
        self.last_model.save_pretrained(str(dest), save_compressed=True)
        
        # Save tokenizer if available
        if self.last_tokenizer is not None:
            self.logger.info(f"Saving tokenizer to: {dest}")
            self.last_tokenizer.save_pretrained(str(dest))
        else:
            # Try to load and save tokenizer from model_id
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                tokenizer.save_pretrained(str(dest))
                self.logger.info("Loaded and saved tokenizer from model_id")
            except Exception as e:
                self.logger.warning(f"Could not save tokenizer: {e}")
        
        self.logger.info(f"Model and tokenizer saved successfully to: {dest}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_recipe(
        self, level: Optional[str], method_kwargs: Dict[str, Any]
    ) -> Tuple[RecipeType, str]:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _default_output_dir(self, level_hint: Optional[str]) -> Path:
        model_name = str(self.model_id).replace("/", "_") if self.model_id else "model"
        level_fragment = (level_hint or "default").replace("/", "_")
        return Path("./output") / f"{self.name}_{model_name}_{level_fragment}"

    def _prepare_oneshot_kwargs(
        self,
        model: Union[str, Path, Any],
        oneshot_kwargs: Dict[str, Any],
        level_hint: Optional[str],
    ) -> Dict[str, Any]:
        prepared = dict(oneshot_kwargs)
        prepared.setdefault("model", model)
        prepared.setdefault("save_compressed", True)
        prepared.setdefault("trust_remote_code_model", True)

        output_dir = prepared.get("output_dir")
        if not output_dir:
            output_dir = self._default_output_dir(level_hint)
            prepared["output_dir"] = str(output_dir)
        else:
            prepared["output_dir"] = str(output_dir)

        Path(prepared["output_dir"]).mkdir(parents=True, exist_ok=True)
        return prepared

    def _has_calibration_source(self, oneshot_kwargs: Dict[str, Any]) -> bool:
        calibration_keys = ("dataset", "dataset_path", "calibration_dataloader")
        return any(oneshot_kwargs.get(key) for key in calibration_keys)

    def _import_llmcompressor_oneshot(self):
        try:
            from llmcompressor import oneshot
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "llmcompressor is required for this quantizer. Install it via "
                "pip install llmcompressor."
            ) from exc
        return oneshot

    def _build_calibration_data(self,dataset: str, num_calibration_samples: int, max_seq_length: int, shuffle_calibration_samples:bool = True, **kwargs):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets library is required for loading calibration data.") from exc
        
        # Load the dataset
        ds = load_dataset(dataset)
        
        # Select a subset for calibration
        if num_calibration_samples > 0:
            ds = ds["train_sft"].select(range(num_calibration_samples))
        else:
            ds = ds["train_sft"]
        if shuffle_calibration_samples:
            ds = ds.shuffle(seed=42)

        # Preprocess the data into the format the model is trained with.
        def preprocess(example):
            return {"text": self.last_tokenizer.apply_chat_template(example["messages"], tokenize=False,)}
        ds = ds.map(preprocess)

        # Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
        def tokenize(sample):
            return self.last_tokenizer(sample["text"], padding=False, max_length=max_seq_length, truncation=True, add_special_tokens=False)
        ds = ds.map(tokenize, remove_columns=ds.column_names)

        return ds