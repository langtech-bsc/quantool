import os
import json
import glob
import shutil
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from quantool.core.helpers import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class DatasetDownloader:
    """
    Provides functionality to get/download datasets from HuggingFace Hub or local directories.
    Wraps around datasets.load_dataset with additional caching and utility features.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        cache_dir: Directory to cache downloaded datasets.
                      If None, uses HF_HOME or ~/.cache/huggingface
        """
        self.cache_dir = cache_dir or os.getenv('HF_HOME',
                                                os.path.expanduser('~/.cache/huggingface'))

    def load_dataset(
            self,
            dataset_name_or_path: Union[str, Path],
            revision: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: bool = False,
            **kwargs
    ) -> Any:
        """
        Load a dataset using datasets.load_dataset with enhanced caching and error handling.

        Args:
            dataset_name_or_path: Either:
                - A string, the dataset id of a dataset hosted on huggingface.co
                - A path to a directory containing dataset files
            revision: The specific dataset version to use
            cache_dir: Path to a directory where downloaded datasets will be cached
            force_download: Whether to force re-downloading even if cached
            **kwargs: Additional arguments passed to datasets.load_dataset

        Returns:
            Dataset or DatasetDict: The loaded dataset object

        Raises:
            ValueError: If dataset cannot be found or loaded
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("datasets library is required. Install with `pip install datasets`.") from e

        # Use provided cache_dir or fall back to instance/default cache_dir
        effective_cache_dir = cache_dir or self.cache_dir

        # Convert to Path object for easier manipulation
        dataset_path = Path(dataset_name_or_path)

        try:
            # Case 1: Local directory path
            if self._is_dataset_dir(dataset_path):
                logger.info(f"Loading local dataset from: {dataset_path}")
                # For local datasets, use datasets.load_dataset with data_files
                if dataset_path.is_dir():
                    # Find dataset files in the directory
                    data_files = self._find_dataset_files(dataset_path)
                    if data_files:
                        return load_dataset(
                            "json",  # Assume JSON format, could be enhanced to detect format
                            data_files=data_files,
                            cache_dir=effective_cache_dir,
                            **kwargs
                        )
                    else:
                        raise ValueError(f"No dataset files found in {dataset_path}")
                else:
                    return load_dataset(str(dataset_path), cache_dir=effective_cache_dir, **kwargs)

            # Case 2: HuggingFace dataset identifier
            else:
                logger.info(f"Loading dataset '{dataset_name_or_path}' from HuggingFace Hub")

                return load_dataset(
                    dataset_name_or_path,
                    revision=revision,
                    cache_dir=effective_cache_dir,
                    download_mode="force_redownload" if force_download else "reuse_dataset_if_exists",
                    **kwargs
                )

        except Exception as e:
            logger.error(f"Error loading dataset '{dataset_name_or_path}': {e}")
            raise ValueError(f"Could not load dataset '{dataset_name_or_path}': {e}")

    def _is_dataset_dir(self, path: Path) -> bool:
        """Check if the path is a local directory containing dataset files."""
        if not path.exists():
            return False

        if not path.is_dir():
            return False

        # Check for common dataset files
        dataset_files = [
            '*.json', '*.jsonl', '*.csv', '*.tsv', '*.txt',
            '*.parquet', '*.arrow', 'dataset_info.json'
        ]

        has_dataset_files = any(glob.glob(str(path / file)) for file in dataset_files)
        return has_dataset_files

    def _find_dataset_files(self, directory: Path) -> List[str]:
        """Find dataset files in a directory."""
        dataset_extensions = ['*.json', '*.jsonl', '*.csv', '*.tsv', '*.txt', '*.parquet', '*.arrow']

        files = []
        for ext in dataset_extensions:
            files.extend(glob.glob(str(directory / ext)))

        return [str(Path(f).relative_to(directory)) for f in files]

    def list_cached_datasets(self) -> List[str]:
        """List all cached datasets in the datasets cache directory."""
        try:
            from datasets import config
            cache_dir = Path(config.HF_DATASETS_CACHE)
        except ImportError:
            cache_dir = Path(self.cache_dir) / "datasets"

        if not cache_dir.exists():
            return []

        cached_datasets = []
        try:
            for item in cache_dir.iterdir():
                if item.is_dir() and item.name.startswith('___'):
                    # datasets library uses ___ naming for cached datasets
                    dataset_name = item.name.replace('___', '').replace('___', '/')
                    cached_datasets.append(dataset_name)
        except Exception:
            pass

        return cached_datasets

    def clear_cache(self, dataset_name: Optional[str] = None):
        """Clear cache for a specific dataset or all datasets."""
        try:
            from datasets import config
            cache_dir = Path(config.HF_DATASETS_CACHE)
        except ImportError:
            cache_dir = Path(self.cache_dir) / "datasets"

        if dataset_name:
            # Clear specific dataset cache
            clean_name = dataset_name.replace('/', '___')
            cache_path = cache_dir / f"___{clean_name}"
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for {dataset_name}")
        else:
            # Clear all dataset caches
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info("Cleared all dataset cache")


# Convenience function for direct usage
def load_dataset(
        dataset_name_or_path: Union[str, Path],
        **kwargs
) -> Any:
    """
    Wrapper function to load a dataset.

    Args:
        dataset_name_or_path: Dataset identifier or local path
        **kwargs: Additional arguments passed to DatasetDownloader.load_dataset()

    Returns:
        Dataset or DatasetDict: The loaded dataset object
    """
    downloader = DatasetDownloader()
    return downloader.load_dataset(dataset_name_or_path, **kwargs)