from abc import ABC, abstractmethod
from meta import TemplateMeta

class BaseQuantizer(ABC):
    """Base class for quantization methods."""
    name: str                 # e.g. "gguf"
    supported_levels: list    # e.g. ["Q2","Q3_K_S","Q4_K_M",...]
    template_meta:


    @abstractmethod
    def quantize(self, model, level: str, **kwargs):
        """Apply quantization at specified level."""
        pass

    @abstractmethod
    def export(self, model, out_path: str):
        """Save quantized model."""
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