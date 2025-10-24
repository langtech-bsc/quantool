from typing import Optional, Union
import os
import tempfile
from huggingface_hub import upload_folder, create_repo
from .modelcard_generator import create_model_card_from_template

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

    def save_model_card(
            self,
            save_directory: Union[str, os.PathLike]
    ):
        """Save the model card to a directory.
        Args:
            save_directory: Directory where the model card should be saved
        """
        os.makedirs(save_directory, exist_ok=True)
        self.logger.info(f"Saving model card to {save_directory}")
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
            self.save_model_card(tmpdir)

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