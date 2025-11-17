from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CalibrationArtifact:
    """Lightweight wrapper passed through pipeline state for calibration data.

    type: identifies the kind of payload. Supported values:
      - "dataset_id": payload is HF dataset id string
      - "dataset_path": payload is local path string
      - "hf_dataset": payload is a `datasets.Dataset` instance
      - "dataloader": payload is a DataLoader or iterator
      - "custom": payload is user-defined

    payload: the actual object
    meta: optional metadata (sample_size, split, etc.)
    """

    type: str
    payload: Any
    meta: Optional[Dict[str, Any]] = field(default_factory=dict)
