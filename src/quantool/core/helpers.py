import logging
import threading
import time
from pythonjsonlogger.jsonlogger import JsonFormatter
import loguru



class LoggerFactory:
    _lock = threading.Lock()
    _configured = False

    @classmethod
    def configure(cls, level=logging.INFO, fmt=None, json=False):
        """Configure root logger once."""
        with cls._lock:
            if cls._configured:
                return
            handler = logging.StreamHandler()
            if json:
                handler.setFormatter(JsonFormatter(fmt))
            else:
                fmt = fmt or '%(asctime)s %(levelname)s [%(name)s] %(message)s'
                handler.setFormatter(logging.Formatter(fmt))
            root = logging.getLogger()
            root.setLevel(level)
            root.addHandler(handler)
            cls._configured = True

    @classmethod
    def get_logger(cls, name=None):
        """Get a module-level logger after ensuring configuration."""
        cls.configure()
        return logging.getLogger(name)

class MetricRecorder:
    """Context manager for recording metrics."""
    def __init__(self, logger=None):
        self.logger = logger or LoggerFactory.get_logger(self.__class__.__name__)

    def timeit(self, op_name):
        """Context manager to log start/end timestamps and duration."""
        class TimerCtx:
            def __enter__(inner_self):
                inner_self.start = time.time()
                self.logger.info(f"START {op_name}")
                return inner_self
            def __exit__(inner_self, exc_type, exc, tb):
                dur = time.time() - inner_self.start
                self.logger.info(f"END {op_name} (duration={dur:.3f}s)")
        return TimerCtx()

    def count(self, metric_name, value=1):
        """Log an incrementing counter metric."""
        self.logger.info(f"METRIC counter {metric_name}={value}")

class PipelineBase:
    """Base class for composable processing pipelines."""
    def __init__(self, steps=None, logger=None):
        self.steps = steps or []
        self.logger = logger or LoggerFactory.get_logger(self.__class__.__name__)
        self.metrics = MetricRecorder(self.logger)

    def add_step(self, func, name=None):
        """Register a processing step (callable)."""
        step_name = name or func.__name__
        self.steps.append((step_name, func))
        return self

    def run(self, data):
        """Execute all steps in order, passing data through."""
        current = data
        for name, func in self.steps:
            self.logger.info(f"Pipeline step START {name}")
            with self.metrics.timeit(name):
                current = func(current)
            self.logger.info(f"Pipeline step END {name}")
        return current

from typing import Optional, Union
import tempfile
import os
from huggingface_hub import upload_folder, create_repo, ModelCard, ModelCardData
from .meta import TemplateQuantizationCard

def create_model_card_from_template(template: TemplateQuantizationCard) -> ModelCard:
    """Create a Hugging Face ModelCard from a TemplateQuantizationCard."""
    return ModelCard(
        model_id=template.title,
        model_name=template.title,
        tags=["quantization"],
        description=template.description,
        metrics=template.hyperparameters,
        intended_use=template.intended_use,
        limitations=template.limitations,
        citations=template.citations
    )

class ExportMixin:
    """Mixin for exporting models to local filesystem or Hugging Face Hub.

    Provides functionality to save model files and model cards, and push them to the Hub.
    This class should be inherited by model classes that need to support exporting.
    
    Note: This mixin expects the inheriting class to provide a 'logger' attribute.
    """

    def _save_model_files(self, save_directory: Union[str, os.PathLike]):
        """Save model-specific files to the specified directory.
        
        Args:
            save_directory: Directory where the model files should be saved
        """
        raise NotImplementedError("Subclasses must implement _save_model_files method")

    def _save_model_card(self, save_directory: Union[str, os.PathLike]):
        """Save model card to the specified directory."""
        if not hasattr(self, "template_card"):
            self.logger.warning("No template_card attribute found, skipping model card generation")
            return

        model_card = create_model_card_from_template(self.template_card)
        model_card_path = os.path.join(save_directory, "README.md")
        model_card.save(model_card_path)
        self.logger.info(f"Model card saved to {model_card_path}")

    def _upload_folder(
            self,
            working_dir: Union[str, os.PathLike],
            repo_id: str,
            token: Optional[str] = None,
            commit_message: Optional[str] = None,
            create_pr: bool = False
        ):
        """Upload a folder to Hugging Face Hub."""
        if commit_message is None:
            commit_message = f"Upload {self.__class__.__name__} model"

        return upload_folder(
            folder_path=working_dir,
            path_in_repo=".",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
            create_pr=create_pr
        )

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike]
        ):
        """Save the model and configuration files to a directory.
        
        Args:
            save_directory: Directory where the model/files should be saved
        """
        os.makedirs(save_directory, exist_ok=True)
        self.logger.info(f"Saving model files to {save_directory}")
        # Save model specific files - to be implemented by subclasses
        self._save_model_files(save_directory)
        
        # Save model card
        self._save_model_card(save_directory)
        

    def push_to_hub(
            self,
            repo_id: Optional[str] = None,
            commit_message: Optional[str] = None,
            private: Optional[bool] = None,
            token: Optional[str] = None,
            create_pr: bool = False,
            safe_serialization: bool = False,
            variant: Optional[str] = None,
        ):
        """Push the model to the Hugging Face Model Hub.
        
        Args:
            repo_id: The name of the repository to push to
            commit_message: Message to commit while pushing
            private: Whether the repository should be private
            token: The token to use as HTTP bearer authorization
            create_pr: Whether to create a PR instead of pushing directly
            safe_serialization: Whether to use safe serialization
            variant: The variant name for this model
            
        Returns:
            The url of the commit on the hub
        """
        if repo_id is None:
            if hasattr(self, "repo_id") and self.repo_id:
                repo_id = self.repo_id
            else:
                if not hasattr(self, "name"):
                    raise ValueError("repo_id must be specified if the model doesn't have a name attribute")
                # Use model name if available
                repo_id = self.name

        token = token if token is not None else os.environ.get("HF_TOKEN", None)
        
        # Create a temporary directory to save files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save all model files and model card
            self.save_pretrained(tmpdir)
            
            # Create repo (or get existing)
            repo = create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                exist_ok=True
            )
            
            # Upload the files
            self.logger.info(f"Pushing model to {repo_id}")
            return self._upload_folder(
                working_dir=tmpdir,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr
            )
