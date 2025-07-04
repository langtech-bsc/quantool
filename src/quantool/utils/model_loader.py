import os
import json
import glob
import shutil
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Provides functionality to get/download models from HuggingFace Hub or local directories.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        cache_dir: Directory to cache downloaded models.
                      If None, uses HF_HOME or ~/.cache/huggingface
        """
        self.cache_dir = cache_dir or os.getenv('HF_HOME',
                                                os.path.expanduser('~/.cache/huggingface'))

    def load_model(
            self,
            model_name_or_path: Union[str, Path],
            revision: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: bool = False,
            resume_download: bool = True,
            proxies: Optional[Dict[str, str]] = None,
            use_auth_token: Optional[Union[bool, str]] = None,
            **kwargs
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
            if self._is_local_path(model_path):
                return self._handle_local_path(model_path)
            # Case 2: Trying to download HuggingFace model identifier
            else:
                logger.info(f"Directory does not contain recognizable model files: {model_name_or_path},"
                            f" trying to download from HuggingFace Hub")

                # First try to get cached model
                cached_path = self._get_cached_path(model_name_or_path, effective_cache_dir, revision)
                if cached_path and os.path.exists(cached_path):
                    logger.info(f"Loading cached model from {cached_path}")
                    return cached_path

                # Else fall to download from HuggingFace Hub
                return self._handle_hf_model(
                    model_name_or_path,
                    revision=revision,
                    cache_dir=effective_cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=use_auth_token,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Error processing model '{model_name_or_path}': {e}")
            raise ValueError(f"Could not load model '{model_name_or_path}': {e}")

    def _is_local_path(self, path: Path) -> bool:
        """Check if the path is a local directory containing model files."""
        if not path.exists():
            logger.info(f"Local path does not exist: {path}")
            return False

        if not path.is_dir():
            logger.info(f"Path is not a directory: {path}")
            return False

        # Check for common model files
        model_files = [
            'config.json', '*.bin', '*.safetensors',
            'tf_model.h5', 'flax_model.msgpack', 'tokenizer.json',
            'tokenizer_config.json'
        ]

        has_model_files = any(glob.glob(str(path / file)) for file in model_files)

        if has_model_files:
            logger.info(f"Found local model directory: {path}")
            return True

        logger.info(f"No recognizable model files found in directory: {path}")
        return False

    def _handle_local_path(self, path: Path) -> str:
        """Handle local directory path."""
        if not path.exists():
            raise ValueError(f"Local directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        return str(path.resolve())

    def _handle_hf_model(
            self,
            model_id: str,
            revision: Optional[str] = None,
            cache_dir: str = None,
            force_download: bool = False,
            resume_download: bool = True, # TODO: check if its legacy
            proxies: Optional[Dict[str, str]] = None,
            use_auth_token: Optional[Union[bool, str]] = None,
            **kwargs
    ) -> str:
        """Handle HuggingFace model download."""

        try:
            logger.info(f"Downloading model '{model_id}' from HuggingFace Hub using snapshot_download")

            # Use snapshot_download for full model download
            model_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                # resume_download=resume_download,
                proxies=proxies,
                use_auth_token=use_auth_token,
                **kwargs
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

    def _get_cached_path(self, model_id: str, cache_dir: str, revision: Optional[str] = None) -> Optional[str]:
        """Get the cached path for a model if it exists."""

        # Clean model_id for filesystem
        clean_model_id = model_id.replace('/', '--')

        if revision:
            logger.info(f"Looking for cached model '{model_id}' at revision '{revision}'")
            model_cache_dir = os.path.join(cache_dir, f"models--{clean_model_id}/snapshots/{revision}")
        else:
            logger.info(f"Looking for cached model '{model_id}'")
            model_cache_dir = os.path.join(cache_dir, f"models--{clean_model_id}/snapshots/")
            # Check if the model cache directory exists
            if not os.path.exists(model_cache_dir):
                logger.info(f"Model cache directory does not exist: {model_cache_dir}")
                return None
            # Get latest revision dir from model_cache_dir (go through all subdirs)
            subdirs = [d for d in os.listdir(model_cache_dir) if os.path.isdir(os.path.join(model_cache_dir, d))]
            if subdirs:
                # Sort subdirs by modification time and take the most recent one
                subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(model_cache_dir, d)), reverse=True)
                model_cache_dir = os.path.join(model_cache_dir, subdirs[0])
                logger.info(f"Using latest revision: {model_cache_dir}")

        if os.path.exists(model_cache_dir):
            return model_cache_dir

        return None

    def list_cached_models(self) -> List[str]:
        """List all cached models."""
        if not os.path.exists(self.cache_dir):
            return []

        cached_models = []
        for item in os.listdir(self.cache_dir):
            if item.startswith('models--'):
                # Convert back to model name format
                model_name = item.replace('models--', '').replace('--', '/')
                cached_models.append(model_name)

        return cached_models

    def clear_cache(self, model_name_or_path: Optional[str] = None):
        """Clear cache for a specific model or all models."""
        if model_name_or_path:
            clean_name = model_name_or_path.replace('/', '--')
            cache_path = os.path.join(self.cache_dir, f"models--{clean_name}")
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for {model_name_or_path}")
        else:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                logger.info("Cleared all model cache")


# Convenience function for direct usage
def load_model(
        model_name_or_path: Union[str, Path],
        **kwargs
) -> str:
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


# Example usage
if __name__ == "__main__":
    # Example 1: Load from local directory
    print("Loading model from local directory...")
    try:
        model_path = from_pretrained("/Users/macbook/.cache/huggingface/models--bert-base-cased/")
        print(f"Local model loaded from: {model_path}")
    except ValueError as e:
        print(f"Local model not found: {e}")

    print("\nDownloading model from HuggingFace Hub...")
    # Example 2: Download from HuggingFace
    try:
        model_path = load_model("bert-base-cased", revision="cd5ef92a9fb2f889e972770a36d4ed042daf221e")
        print(f"HF model downloaded to: {model_path}")
    except ValueError as e:
        print(f"HF model download failed: {e}")

    print("\nDownloading model with specific revision...")
    # Example 3: Use with custom cache directory
    try:
        model_path = load_model(
            "openai-community/openai-gpt",
            cache_dir="./custom_cache",
            revision="main"
        )
        print(f"Model with custom cache: {model_path}")
    except ValueError as e:
        print(f"Custom cache download failed: {e}")

    print("\nListing cached models...")
    # Example 4: Using the class directly
    downloader = ModelDownloader(cache_dir="./my_models")

    # List cached models
    cached = downloader.list_cached_models()
    print(f"Cached models: {cached}")

    # Clear specific model cache
    downloader.clear_cache("bert-base-uncased")
