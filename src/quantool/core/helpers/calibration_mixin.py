
class CalibrationMixin:
    """Mixin for calibration-related functionalities."""
    
    def require_calibration(self) -> bool:
        """Check if the model requires calibration.
        
        Returns:
            bool: True if calibration is required, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement require_calibration method")
    
    def prepare_calibration_data(self, data_loader):
        """Prepare data for calibration.
        
        Args:
            data_loader: DataLoader providing calibration data.
        """
        raise NotImplementedError("Subclasses must implement prepare_calibration_data method")
    
    def run_calibration(self):
        """ Optionally run the calibration process. """
        return None