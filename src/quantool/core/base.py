from abc import ABC, abstractmethod
from typing import Union, List
from quantool.core.helpers import ExportMixin, LoggerFactory
from quantool.core.meta import TemplateQuantizationCard


class BaseQuantizer(ABC, ExportMixin):
    """Base class for quantization methods."""
    name: str                 # e.g. "gguf"
    supported_levels: list    # e.g. ["Q2","Q3_K_S","Q4_K_M",...]
    supports_multiple_levels: bool = False  # Whether the method can quantize to multiple levels at once
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
    def quantize(self, model, level: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Apply quantization at specified level(s)."""
        if isinstance(level, list) and not self.supports_multiple_levels:
            raise ValueError(f"Method '{self.name}' does not support multiple quantization levels. "
                           f"Please specify a single level.")