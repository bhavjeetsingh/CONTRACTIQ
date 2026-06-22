"""
JSON Exporter for ContractIQ
==============================
Exports contract analysis and key-term extraction results as JSON.
"""

import json
from typing import Dict, Any
from pathlib import Path

from logger import GLOBAL_LOGGER as log


def export_json(data: Dict[str, Any], output_path: str = None) -> str:
    """
    Export analysis results as formatted JSON.

    Args:
        data: Analysis result dictionary
        output_path: Optional file path to save to

    Returns:
        JSON string
    """
    json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        log.info("JSON exported", path=output_path, size=len(json_str))

    return json_str
