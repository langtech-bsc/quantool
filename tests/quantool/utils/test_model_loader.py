import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from huggingface_hub.utils import HfHubHTTPError

from quantool.utils.model_loader import ModelDownloader, load_model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model_dir(temp_dir):
    """Create a mock model directory with proper model files."""
    model_dir = temp_dir / "mock_model"
    model_dir.mkdir(parents=True)
    
    # Create mock model files
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "tokenizer.json").write_text('{"tokenizer": "test"}')
    (model_dir / "pytorch_model.bin").touch()
    
    return model_dir


@pytest.fixture
def mock_cache_dir(temp_dir):
    """Create a mock cache directory structure."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True)
    
    # Create a cached model structure
    model_cache = cache_dir / "models--facebook--opt-125m" / "snapshots"
    model_cache.mkdir(parents=True)
    
    # Create a revision directory
    revision_dir = model_cache / "abc123def456"
    revision_dir.mkdir()
    (revision_dir / "config.json").write_text('{"model_type": "opt"}')
    (revision_dir / "pytorch_model.bin").touch()
    
    return cache_dir


@pytest.fixture
def downloader(temp_dir):
    """Create a ModelDownloader instance with temp cache directory."""
    return ModelDownloader(cache_dir=str(temp_dir / "cache"))


class TestModelDownloaderInitialization:
    """Test ModelDownloader initialization."""
    
    def test_init_with_cache_dir(self, temp_dir):
        """Test initialization with explicit cache directory."""
        cache_dir = str(temp_dir / "custom_cache")
        downloader = ModelDownloader(cache_dir=cache_dir)
        assert downloader.cache_dir == cache_dir
    
    def test_init_without_cache_dir(self):
        """Test initialization without explicit cache directory."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('os.path.expanduser') as mock_expanduser:
                mock_expanduser.return_value = "/home/user/.cache/huggingface"
                downloader = ModelDownloader()
                assert downloader.cache_dir == "/home/user/.cache/huggingface"
    
    def test_init_with_hf_home_env(self):
        """Test initialization with HF_HOME environment variable."""
        with patch.dict(os.environ, {'HF_HOME': '/custom/hf/path'}):
            downloader = ModelDownloader()
            assert downloader.cache_dir == '/custom/hf/path'


class TestIsModelDir:
    """Test _is_model_dir method."""
    
    def test_is_model_dir_valid(self, downloader, mock_model_dir):
        """Test with valid model directory."""
        assert downloader._is_model_dir(mock_model_dir) is True
    
    def test_is_model_dir_nonexistent(self, downloader, temp_dir):
        """Test with non-existent directory."""
        nonexistent = temp_dir / "nonexistent"
        assert downloader._is_model_dir(nonexistent) is False
    
    def test_is_model_dir_not_directory(self, downloader, temp_dir):
        """Test with file instead of directory."""
        file_path = temp_dir / "not_a_dir.txt"
        file_path.touch()
        assert downloader._is_model_dir(file_path) is False
    
    def test_is_model_dir_empty_directory(self, downloader, temp_dir):
        """Test with empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        assert downloader._is_model_dir(empty_dir) is False
    
    def test_is_model_dir_with_safetensors(self, downloader, temp_dir):
        """Test with safetensors model files."""
        model_dir = temp_dir / "safetensors_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{}')
        (model_dir / "model.safetensors").touch()
        assert downloader._is_model_dir(model_dir) is True


class TestValidateLocalPath:
    """Test _validate_local_path method."""
    
    def test_validate_local_path_valid(self, downloader, mock_model_dir):
        """Test with valid local path."""
        result = downloader._validate_local_path(mock_model_dir)
        assert result == str(mock_model_dir.resolve())
    
    def test_validate_local_path_nonexistent(self, downloader, temp_dir):
        """Test with non-existent path."""
        nonexistent = temp_dir / "nonexistent"
        with pytest.raises(ValueError, match="Local directory not found"):
            downloader._validate_local_path(nonexistent)
    
    def test_validate_local_path_not_directory(self, downloader, temp_dir):
        """Test with file instead of directory."""
        file_path = temp_dir / "not_a_dir.txt"
        file_path.touch()
        with pytest.raises(ValueError, match="Path is not a directory"):
            downloader._validate_local_path(file_path)


class TestDownloadHfModel:
    """Test _download_hf_model method."""
    
    @patch('quantool.utils.model_loader.snapshot_download')
    def test_download_hf_model_success(self, mock_snapshot, downloader):
        """Test successful HF model download."""
        mock_snapshot.return_value = "/path/to/downloaded/model"
        
        result = downloader._download_hf_model(
            "facebook/opt-125m",
            revision="main",
            cache_dir="/cache",
            force_download=False
        )
        
        assert result == "/path/to/downloaded/model"
        mock_snapshot.assert_called_once_with(
            repo_id="facebook/opt-125m",
            revision="main",
            cache_dir="/cache",
            force_download=False,
            proxies=None,
            use_auth_token=None
        )
    
    @patch('quantool.utils.model_loader.snapshot_download')
    def test_download_hf_model_404_error(self, mock_snapshot, downloader):
        """Test HF model download with 404 error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_error = HfHubHTTPError("Not found", response=mock_response)
        mock_snapshot.side_effect = mock_error
        
        with pytest.raises(ValueError, match="Model 'nonexistent/model' not found on HuggingFace Hub"):
            downloader._download_hf_model("nonexistent/model")
    
    @patch('quantool.utils.model_loader.snapshot_download')
    def test_download_hf_model_other_http_error(self, mock_snapshot, downloader):
        """Test HF model download with other HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_error = HfHubHTTPError("Server error", response=mock_response)
        mock_snapshot.side_effect = mock_error
        
        with pytest.raises(ValueError, match="Error downloading model 'some/model'"):
            downloader._download_hf_model("some/model")
    
    @patch('quantool.utils.model_loader.snapshot_download')
    def test_download_hf_model_unexpected_error(self, mock_snapshot, downloader):
        """Test HF model download with unexpected error."""
        mock_snapshot.side_effect = RuntimeError("Unexpected error")
        
        with pytest.raises(ValueError, match="Unexpected error downloading model 'some/model'"):
            downloader._download_hf_model("some/model")


class TestGetCachedPath:
    """Test _get_cached_path method."""
    
    def test_get_cached_path_with_revision(self, downloader, temp_dir):
        """Test getting cached path with specific revision."""
        # Create cache structure
        cache_dir = temp_dir / "cache"
        model_cache = cache_dir / "models--facebook--opt-125m" / "snapshots" / "abc123"
        model_cache.mkdir(parents=True)
        (model_cache / "config.json").touch()
        
        result = downloader._get_cached_path(
            "facebook/opt-125m", 
            str(cache_dir), 
            revision="abc123"
        )
        
        assert result == model_cache
        assert result.exists()
    
    def test_get_cached_path_without_revision(self, downloader, temp_dir):
        """Test getting cached path without specific revision."""
        # Create cache structure with multiple revisions
        cache_dir = temp_dir / "cache"
        snapshots_dir = cache_dir / "models--facebook--opt-125m" / "snapshots"
        snapshots_dir.mkdir(parents=True)
        
        # Create older revision
        old_revision = snapshots_dir / "old_revision"
        old_revision.mkdir()
        (old_revision / "config.json").touch()
        
        # Create newer revision (sleep to ensure different mtime)
        import time
        time.sleep(0.01)
        new_revision = snapshots_dir / "new_revision" 
        new_revision.mkdir()
        (new_revision / "config.json").touch()
        
        result = downloader._get_cached_path("facebook/opt-125m", str(cache_dir))
        
        assert result == new_revision
    
    def test_get_cached_path_nonexistent_model(self, downloader, temp_dir):
        """Test getting cached path for non-existent model."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        
        result = downloader._get_cached_path("nonexistent/model", str(cache_dir))
        assert result is None
    
    def test_get_cached_path_no_snapshots(self, downloader, temp_dir):
        """Test getting cached path when snapshots directory is empty."""
        cache_dir = temp_dir / "cache" 
        snapshots_dir = cache_dir / "models--facebook--opt-125m" / "snapshots"
        snapshots_dir.mkdir(parents=True)
        
        result = downloader._get_cached_path("facebook/opt-125m", str(cache_dir))
        assert result is None


class TestFindLatestSubdir:
    """Test _find_latest_subdir method."""
    
    def test_find_latest_subdir_success(self, downloader, temp_dir):
        """Test finding latest subdirectory."""
        test_dir = temp_dir / "test"
        test_dir.mkdir()
        
        # Create subdirectories with different modification times
        old_dir = test_dir / "old"
        old_dir.mkdir()
        
        import time
        time.sleep(0.01)
        
        new_dir = test_dir / "new"
        new_dir.mkdir()
        
        result = downloader._find_latest_subdir(test_dir)
        assert result == "new"
    
    def test_find_latest_subdir_nonexistent(self, downloader, temp_dir):
        """Test finding latest subdirectory in non-existent directory."""
        nonexistent = temp_dir / "nonexistent"
        result = downloader._find_latest_subdir(nonexistent)
        assert result is None
    
    def test_find_latest_subdir_empty(self, downloader, temp_dir):
        """Test finding latest subdirectory in empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        result = downloader._find_latest_subdir(empty_dir)
        assert result is None
    
    def test_find_latest_subdir_only_files(self, downloader, temp_dir):
        """Test finding latest subdirectory when only files exist."""
        test_dir = temp_dir / "test"
        test_dir.mkdir()
        (test_dir / "file1.txt").touch()
        (test_dir / "file2.txt").touch()
        
        result = downloader._find_latest_subdir(test_dir)
        assert result is None


class TestLoadModel:
    """Test load_model method."""
    
    def test_load_model_local_path(self, downloader, mock_model_dir):
        """Test loading model from local path."""
        result = downloader.load_model(str(mock_model_dir))
        assert result == str(mock_model_dir.resolve())
    
    @patch('quantool.utils.model_loader.snapshot_download')
    def test_load_model_hf_download(self, mock_snapshot, downloader):
        """Test loading model by downloading from HF."""
        mock_snapshot.return_value = "/path/to/downloaded"
        
        result = downloader.load_model("facebook/opt-125m")
        
        assert result == "/path/to/downloaded"
        mock_snapshot.assert_called_once()
    
    def test_load_model_cached(self, downloader, temp_dir):
        """Test loading model from cache."""
        # Create cached model
        cache_dir = temp_dir / "cache"
        model_cache = cache_dir / "models--facebook--opt-125m" / "snapshots" / "abc123"
        model_cache.mkdir(parents=True)
        (model_cache / "config.json").touch()
        
        downloader.cache_dir = str(cache_dir)
        
        result = downloader.load_model("facebook/opt-125m", revision="abc123")
        assert str(result) == str(model_cache)

    @patch('quantool.utils.model_loader.snapshot_download')
    def test_load_model_with_custom_cache_dir(self, mock_snapshot, downloader, temp_dir):
        """Test loading model with custom cache directory."""
        mock_snapshot.return_value = "/path/to/downloaded"
        custom_cache = str(temp_dir / "custom_cache")
        
        downloader.load_model("facebook/opt-125m", cache_dir=custom_cache)
        
        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs['cache_dir'] == custom_cache
    
    def test_load_model_error_handling(self, downloader, temp_dir):
        """Test error handling in load_model."""
        with pytest.raises(ValueError, match="Could not load model"):
            downloader.load_model("nonexistent/path")


class TestListCachedModels:
    """Test list_cached_models method."""
    
    def test_list_cached_models_success(self, downloader, temp_dir):
        """Test listing cached models."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        
        # Create cached models
        (cache_dir / "models--facebook--opt-125m").mkdir()
        (cache_dir / "models--microsoft--DialoGPT-medium").mkdir()
        (cache_dir / "not-a-model").mkdir()  # Should be ignored
        
        downloader.cache_dir = str(cache_dir)
        
        result = downloader.list_cached_models()
        
        assert "facebook/opt-125m" in result
        assert "microsoft/DialoGPT-medium" in result
        assert len(result) == 2
    
    def test_list_cached_models_empty_cache(self, downloader, temp_dir):
        """Test listing cached models with empty cache."""
        cache_dir = temp_dir / "cache"
        downloader.cache_dir = str(cache_dir)
        
        result = downloader.list_cached_models()
        assert result == []
    
    def test_list_cached_models_nonexistent_cache(self, downloader, temp_dir):
        """Test listing cached models with non-existent cache directory."""
        downloader.cache_dir = str(temp_dir / "nonexistent")
        
        result = downloader.list_cached_models()
        assert result == []


class TestClearCache:
    """Test clear_cache method."""
    
    def test_clear_cache_specific_model(self, downloader, temp_dir):
        """Test clearing cache for specific model."""
        cache_dir = temp_dir / "cache"
        model_cache = cache_dir / "models--facebook--opt-125m"
        model_cache.mkdir(parents=True)
        (model_cache / "test_file").touch()
        
        downloader.cache_dir = str(cache_dir)
        
        downloader.clear_cache("facebook/opt-125m")
        
        assert not model_cache.exists()
    
    def test_clear_cache_all_models(self, downloader, temp_dir):
        """Test clearing all cached models."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "models--facebook--opt-125m").mkdir()
        (cache_dir / "models--microsoft--DialoGPT-medium").mkdir()
        
        downloader.cache_dir = str(cache_dir)
        
        downloader.clear_cache()
        
        assert not cache_dir.exists()
    
    def test_clear_cache_nonexistent_model(self, downloader, temp_dir):
        """Test clearing cache for non-existent model."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        
        downloader.cache_dir = str(cache_dir)
        
        # Should not raise error
        downloader.clear_cache("nonexistent/model")
        assert cache_dir.exists()


class TestLoadModelConvenienceFunction:
    """Test the convenience load_model function."""
    
    @patch('quantool.utils.model_loader.ModelDownloader')
    def test_load_model_function(self, mock_downloader_class):
        """Test the convenience load_model function."""
        mock_downloader = MagicMock()
        mock_downloader.load_model.return_value = "/path/to/model"
        mock_downloader_class.return_value = mock_downloader
        
        result = load_model("facebook/opt-125m", revision="main")
        
        assert result == "/path/to/model"
        mock_downloader_class.assert_called_once_with()
        mock_downloader.load_model.assert_called_once_with(
            "facebook/opt-125m", 
            revision="main"
        )


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_load_model_with_path_object(self, downloader, mock_model_dir):
        """Test load_model with Path object instead of string."""
        result = downloader.load_model(mock_model_dir)
        assert result == str(mock_model_dir.resolve())
    
    @patch('quantool.utils.model_loader.snapshot_download')
    def test_load_model_with_kwargs(self, mock_snapshot, downloader):
        """Test load_model with additional kwargs."""
        mock_snapshot.return_value = "/path/to/model"
        
        downloader.load_model(
            "facebook/opt-125m",
            custom_arg="custom_value",
            another_arg=123
        )
        
        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs['custom_arg'] == "custom_value"
        assert call_kwargs['another_arg'] == 123
    
    def test_cached_path_type_conversion(self, downloader, temp_dir):
        """Test that _get_cached_path returns Path object consistently."""
        cache_dir = temp_dir / "cache"
        model_cache = cache_dir / "models--facebook--opt-125m" / "snapshots" / "abc123"
        model_cache.mkdir(parents=True)
        
        result = downloader._get_cached_path("facebook/opt-125m", str(cache_dir), "abc123")
        
        assert isinstance(result, Path)
        assert result == model_cache


if __name__ == "__main__":
    pytest.main([__file__])