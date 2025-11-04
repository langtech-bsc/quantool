from abc import ABC, abstractmethod

from quantool.core.helpers import CalibrationMixin, ExportMixin, LoggerFactory
from quantool.core.meta import TemplateQuantizationCard


class BaseQuantizer(ABC, ExportMixin, CalibrationMixin):
    """Base class for quantization methods."""

    name: str  # e.g. "gguf"
    supported_levels: list  # e.g. ["Q2","Q3_K_S","Q4_K_M",...]
    template_card: TemplateQuantizationCard  # e.g. TemplateQuantizationCard(title="GGUF", description="GGUF quantization method")

    def __init__(self, model_id, *args, **kwargs):
        """Initialize the quantizer with its own logger.

        This initializes the logger before calling ExportMixin's __init__ method,
        so the mixin will use this logger rather than creating its own.
        """
        self.model_id = model_id
        # Create logger first so ExportMixin will use it
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        # Call parent's __init__ (including ExportMixin)
        super().__init__(*args, **kwargs)

    @abstractmethod
    def quantize(self, model, level: str, **kwargs):
        """Apply quantization at specified level."""
        pass
