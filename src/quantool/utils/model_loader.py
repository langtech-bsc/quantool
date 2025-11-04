import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from quantool.core.helpers import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class ModelDownloader:
    """
    Provides functionality to get/download models from HuggingFace Hub or local directories.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        cache_dir: Directory to cache downloaded models.
                      If None, uses HF_HOME or ~/.cache/huggingface
        """
        self.cache_dir = cache_dir or os.getenv(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )

    def load_model(
        self,
        model_name_or_path: Union[str, Path],
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        resume_download: bool = True,
        proxies: Optional[Dict[str, str]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        **kwargs,
    ) -> str:
        """
        Download or locate a pretrained model, similar to HuggingFace's from_pretrained.

        Args:
            model_name_or_path: Either:
                - A string, the model id of a pretrained model hosted on huggingface.co
                - A path to a directory containing model files
            revision: The specific model version to use (branch name, tag name, or commit id)
            cache_dir: Path to a directory where downloaded models will be cached
            force_download: Whether to force re-downloading even if cached
            resume_download: Whether to resume incomplete downloads
            proxies: Dictionary of proxy servers to use
            use_auth_token: Token to use for authentication
            **kwargs: Additional arguments passed to snapshot_download

        Returns:
            str: Path to the directory containing the model files

        Raises:
            ValueError: If model cannot be found locally or downloaded
            HfHubHTTPError: If there's an error downloading from HuggingFace Hub
        """

        # Use provided cache_dir or fall back to instance/default cache_dir
        effective_cache_dir = cache_dir or self.cache_dir

        # Convert to Path object for easier manipulation
        model_path = Path(model_name_or_path)

        try:
            # Case 1: Local directory path
            if self._is_model_dir(model_path):
                return self._validate_local_path(model_path)
            # Case 2: Trying to download HuggingFace model identifier
            else:
                logger.info(
                    f"Directory does not contain recognizable model files: {model_name_or_path},"
                    f" trying to download from HuggingFace Hub"
                )

                # First try to get cached model
                cached_path = self._get_cached_path(
                    model_name_or_path, effective_cache_dir, revision
                )
                if cached_path and os.path.exists(cached_path):
                    logger.info(f"Loading cached model from {cached_path}")
                    return cached_path

                # Else fall to download from HuggingFace Hub
                return self._download_hf_model(
                    model_name_or_path,
                    revision=revision,
                    cache_dir=effective_cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=use_auth_token,
                    **kwargs,
                )
        except Exception as e:
            logger.error(f"Error processing model '{model_name_or_path}': {e}")
            raise ValueError(f"Could not load model '{model_name_or_path}': {e}")

    def _is_model_dir(self, path: Path) -> bool:
        """Check if the path is a local directory containing model files."""
        if not path.exists():
            logger.info(f"Local path does not exist: {path}")
            return False

        if not path.is_dir():
            logger.info(f"Path is not a directory: {path}")
            return False

        # Check for common model files
        model_files = [
            "config.json",
            "*.bin",
            "*.safetensors",
            "tf_model.h5",
            "flax_model.msgpack",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        has_model_files = any(glob.glob(str(path / file)) for file in model_files)

        if has_model_files:
            logger.info(f"Found local model directory: {path}")
            return True

        logger.info(f"No recognizable model files found in directory: {path}")
        return False

    def _validate_local_path(self, path: Path) -> str:
        """Validate local directory path."""
        if not path.exists():
            raise ValueError(f"Local directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        return str(path.resolve())

    def _download_hf_model(
        self,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: str = None,
        force_download: bool = False,
        resume_download: bool = True,  # TODO: check if its legacy
        proxies: Optional[Dict[str, str]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        **kwargs,
    ) -> str:
        """Handle HuggingFace model download."""

        try:
            logger.info(
                f"Downloading model '{model_id}' from HuggingFace Hub using snapshot_download"
            )

            # Use snapshot_download for full model download
            model_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                # resume_download=resume_download,
                proxies=proxies,
                use_auth_token=use_auth_token,
                **kwargs,
            )
            logger.info(f"Model downloaded successfully to: {model_path}")
            return model_path

        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{model_id}' not found on HuggingFace Hub")
            else:
                raise ValueError(f"Error downloading model '{model_id}': {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error downloading model '{model_id}': {e}")

    def _get_cached_path(
        self, model_id: str, cache_dir: str, revision: Optional[str] = None
    ) -> Optional[Path]:
        """Get the cached path for a model if it exists."""
        clean_model_id = model_id.replace("/", "--")
        cache_dir = Path(cache_dir)
        model_cache_dir = cache_dir / f"models--{clean_model_id}/snapshots/"

        if revision:
            logger.info(
                f"Looking for cached model '{model_id}' at revision '{revision}'"
            )
            model_cache_dir = model_cache_dir / revision
        else:
            logger.info(f"Looking for cached model '{model_id}'")
            if not model_cache_dir.exists():
                logger.info(f"Model cache directory does not exist: {model_cache_dir}")
                return None

            subdir = self._find_latest_subdir(model_cache_dir)
            if subdir:
                model_cache_dir = model_cache_dir / subdir
                logger.info(f"Using latest revision: {model_cache_dir}")
            else:
                return None

        return model_cache_dir if model_cache_dir.exists() else None

    def _find_latest_subdir(self, directory: Path) -> Optional[str]:
        """Find the most recently modified subdirectory."""
        if not directory.exists():
            return None

        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        if not subdirs:
            return None

        latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
        return latest_subdir.name

    def list_cached_models(self) -> List[str]:
        """List all cached models."""
        if not os.path.exists(self.cache_dir):
            return []

        cached_models = []
        for item in os.listdir(self.cache_dir):
            if item.startswith("models--"):
                # Convert back to model name format
                model_name = item.replace("models--", "").replace("--", "/")
                cached_models.append(model_name)

        return cached_models

    def clear_cache(self, model_name_or_path: Optional[str] = None):
        """Clear cache for a specific model or all models."""
        if model_name_or_path:
            clean_name = model_name_or_path.replace("/", "--")
            cache_path = os.path.join(self.cache_dir, f"models--{clean_name}")
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for {model_name_or_path}")
        else:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                logger.info("Cleared all model cache")


# Convenience function for direct usage
def load_model(model_name_or_path: Union[str, Path], **kwargs) -> str:
    """
    Wrapper function to load a pretrained model.

    Args:
        model_name_or_path: Model identifier or local path
        **kwargs: Additional arguments passed to ModelDownloader.load()

    Returns:
        str: Path to the directory containing the model files
    """
    downloader = ModelDownloader()
    return downloader.load_model(model_name_or_path, **kwargs)
