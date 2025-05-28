from abc import ABC, abstractmethod
from helpers import ExportMixin, LoggerFactory
from meta import TemplateQuantizationCard


class BaseQuantizer(ABC, ExportMixin):
    """Base class for quantization methods."""
    name: str                 # e.g. "gguf"
    supported_levels: list    # e.g. ["Q2","Q3_K_S","Q4_K_M",...]
    template_card: TemplateQuantizationCard  # e.g. TemplateQuantizationCard(title="GGUF", description="GGUF quantization method")

    def __init__(self, *args, **kwargs):
        """Initialize the quantizer with its own logger.
        
        This initializes the logger before calling ExportMixin's __init__ method,
        so the mixin will use this logger rather than creating its own.
        """
        # Create logger first so ExportMixin will use it
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)
        
        # Call parent's __init__ (including ExportMixin)
        super().__init__(*args, **kwargs)

    @abstractmethod
    def quantize(self, model, level: str, **kwargs):
        """Apply quantization at specified level."""
        pass

        

from abc import Environment, PackageLoader, select_autoescape

class BaseTemplateRenderer:
    """Template Method pattern skeleton."""
    name: str  # e.g. "gptq"
    template_file: str  # e.g. "gptq_readme.md.j2"

    def __init__(self):
        self.env = Environment(
            loader=PackageLoader("quantool", "templates"),
            autoescape=select_autoescape(["md", "j2"])
        )  # single shared env :contentReference[oaicite:7]{index=7}

    def render(self, context: dict, output_path: str):
        ctx = self.build_context(context)
        tpl = self.env.get_template(self.template_file)
        content = tpl.render(**ctx)
        with open(output_path, "w") as f:
            f.write(content)

    def build_context(self, base_ctx: dict) -> dict:
        """Hook for subclasses to extend context."""
        return base_ctx