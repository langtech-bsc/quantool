from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class TemplateQuantizationCard:
    """Used by the generic README template."""
    title: str = field(metadata={"help": "Human-readable title for this quantized model"})
    description: str = field(metadata={"help": "Brief description of the method"})
    hyperparameters: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Key/value pairs of method-specific settings"}
    )
    intended_use: str = field(default="", metadata={"help": "Recommended use cases"})
    limitations: str = field(default="", metadata={"help": "Known limitations"})
    citations: List[str] = field(default_factory=list, metadata={"help": "List of citation strings"})
    
    # Additional fields for better README generation
    base_model: Optional[str] = field(default=None, metadata={"help": "Base model ID or path"})
    model_id: Optional[str] = field(default=None, metadata={"help": "Quantized model ID"})
    quantization_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Detailed quantization configuration"}
    )
    usage_example: Optional[str] = field(
        default=None, 
        metadata={"help": "Code example for using the quantized model"}
    )
    language: Optional[str] = field(default="en", metadata={"help": "Model language"})
    license: Optional[str] = field(default=None, metadata={"help": "Model license"})
    library_name: Optional[str] = field(default=None, metadata={"help": "Library used (e.g., transformers, llama.cpp)"})
