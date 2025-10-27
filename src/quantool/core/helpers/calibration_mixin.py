class CalibrationMixin:
    """Mixin for calibration-related functionalities."""


    def require_calibration(self) -> bool:
        """Check if the model requires calibration.

        Returns:
            bool: True if calibration is required, False otherwise.
        """
        return False

    def prepare_calibration_data(self, dataset, tokenizer=None):
        """Prepare calibration data for this specific quantization method.

        This method is called after basic dataset loading but before quantization.
        Subclasses can override this to apply method-specific preprocessing.

        Args:
            dataset: The loaded dataset (Dataset or DatasetDict)
            tokenizer: Optional tokenizer for preprocessing that requires tokenization

        Returns:
            The prepared dataset ready for quantization
        """
        # Default implementation: return dataset unchanged
        return dataset

    def run_calibration(self):
        """ Optionally run the calibration process. """
        return None