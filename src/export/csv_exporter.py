"""
CSV Exporter for ContractIQ
==============================
Exports key-term extraction results as CSV for spreadsheet use.
"""

import csv
import io
from typing import Dict, Any, List
from pathlib import Path

from logger import GLOBAL_LOGGER as log


def export_csv(data: Dict[str, Any], output_path: str = None) -> str:
    """
    Export key terms as CSV.

    Flattens nested structures into a tabular format.

    Args:
        data: Extraction result dictionary
        output_path: Optional file path to save to

    Returns:
        CSV string
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Category", "Field", "Value", "Confidence"])

    key_terms = data.get("key_terms", data.get("extracted_terms", data))
    scores = data.get("confidence_scores", {})

    # Parties
    for party in key_terms.get("parties", []):
        if isinstance(party, dict):
            writer.writerow(["Parties", party.get("name", ""), party.get("role", ""), scores.get("parties", "")])

    # Dates
    writer.writerow(["Dates", "Effective Date", key_terms.get("effective_date", ""), scores.get("effective_date", "")])
    writer.writerow(["Dates", "Expiration Date", key_terms.get("expiration_date", ""), scores.get("expiration_date", "")])
    writer.writerow(["Dates", "Auto Renewal", str(key_terms.get("auto_renewal", "")), ""])

    # Payment terms
    for pt in key_terms.get("payment_terms", []):
        if isinstance(pt, dict):
            val = f"{pt.get('amount', '')} {pt.get('currency', '')} - {pt.get('frequency', '')}"
            writer.writerow(["Payment", val, pt.get("due_date", ""), scores.get("payment_terms", "")])

    # Obligations
    for ob in key_terms.get("obligations", []):
        if isinstance(ob, dict):
            writer.writerow(["Obligation", ob.get("party", ""), ob.get("description", ""), scores.get("obligations", "")])

    # Risk flags
    for rf in key_terms.get("risk_flags", []):
        if isinstance(rf, dict):
            writer.writerow(["Risk", rf.get("risk_level", ""), rf.get("plain_english", ""), scores.get("risk_flags", "")])

    # Other fields
    writer.writerow(["Legal", "Governing Law", key_terms.get("governing_law", ""), scores.get("governing_law", "")])
    writer.writerow(["Legal", "Liability Cap", key_terms.get("liability_cap", ""), scores.get("liability_cap", "")])
    writer.writerow(["Legal", "Confidentiality Period", key_terms.get("confidentiality_period", ""), ""])

    csv_str = output.getvalue()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(csv_str)
        log.info("CSV exported", path=output_path)

    return csv_str
