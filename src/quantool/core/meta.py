from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TemplateQuantizationCard:
    """Used by the generic README template."""

    title: str = field(
        metadata={"help": "Human-readable title for this quantized model"}
    )
    description: str = field(metadata={"help": "Brief description of the method"})
    hyperparameters: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Key/value pairs of method-specific settings"},
    )
    intended_use: str = field(default="", metadata={"help": "Recommended use cases"})
    limitations: str = field(default="", metadata={"help": "Known limitations"})
    citations: List[str] = field(
        default_factory=list, metadata={"help": "List of citation strings"}
    )
