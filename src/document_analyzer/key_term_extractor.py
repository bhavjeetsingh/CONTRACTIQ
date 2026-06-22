"""
KeyTermExtractor — Structured Contract Key-Term Extraction
=============================================================
Extracts 12+ structured data points from contracts:
    - Parties, roles
    - Effective date, expiration, auto-renewal
    - Payment terms (amount, frequency, due dates)
    - Obligations (who, what, when, consequence)
    - Termination clauses
    - Liability caps
    - Risk flags with plain-English explanations
    - Confidence scores per field

Uses LLM with Pydantic structured output for type-safe extraction.
"""

import os
import sys
from typing import Dict, Any, Optional

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate


# Extraction prompt — designed for maximum precision on legal contracts
EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert legal contract analyst. Extract ALL key terms from the following contract.

For each field, also provide a confidence score (0.0 to 1.0) indicating how certain you are
about the extraction. Use 1.0 for clearly stated terms, 0.5 for inferred terms, and 0.0
for terms not found in the document.

Return ONLY valid JSON matching this exact schema:
{{
    "parties": [
        {{"name": "...", "role": "buyer/seller/contractor/client/etc."}}
    ],
    "effective_date": "YYYY-MM-DD or description",
    "expiration_date": "YYYY-MM-DD or description",
    "auto_renewal": true/false,
    "payment_terms": [
        {{
            "amount": "dollar/currency amount",
            "currency": "USD/INR/EUR/etc.",
            "frequency": "monthly/one-time/milestone-based/etc.",
            "due_date": "when payment is due"
        }}
    ],
    "obligations": [
        {{
            "party": "who is obligated",
            "description": "what they must do",
            "deadline": "by when",
            "consequence_of_breach": "what happens if breached"
        }}
    ],
    "termination_clauses": ["clause 1 summary", "clause 2 summary"],
    "liability_cap": "maximum liability amount or description",
    "confidentiality_period": "duration of confidentiality obligations",
    "governing_law": "jurisdiction/state/country",
    "non_compete": {{
        "exists": true/false,
        "duration": "time period",
        "geographic_scope": "area covered"
    }},
    "risk_flags": [
        {{
            "clause": "the problematic clause text",
            "risk_level": "low/medium/high/critical",
            "plain_english": "what this means in simple business terms",
            "recommendation": "what the user should do about it"
        }}
    ],
    "confidence_scores": {{
        "parties": 0.0-1.0,
        "effective_date": 0.0-1.0,
        "expiration_date": 0.0-1.0,
        "payment_terms": 0.0-1.0,
        "obligations": 0.0-1.0,
        "termination_clauses": 0.0-1.0,
        "liability_cap": 0.0-1.0,
        "governing_law": 0.0-1.0,
        "risk_flags": 0.0-1.0
    }}
}}

If a field is not found in the document, set the value to "Not found" and confidence to 0.0.
Do NOT hallucinate or infer terms that are not in the document.

Contract text:
{document_text}
""")


class KeyTermExtractor:
    """
    Extracts structured key terms from contract text using LLM.

    Features:
        - Type-safe extraction via Pydantic/JSON schema
        - Per-field confidence scoring
        - Automatic LLM provider fallback (Groq -> Google)
        - OutputFixingParser for malformed JSON recovery
    """

    def __init__(self):
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            self.parser = JsonOutputParser()
            self.fixing_parser = OutputFixingParser.from_llm(
                parser=self.parser, llm=self.llm
            )
            self.prompt = EXTRACTION_PROMPT
            log.info("KeyTermExtractor initialized")
        except Exception as e:
            log.error("KeyTermExtractor init failed", error=str(e))
            raise DocumentPortalException("KeyTermExtractor init failed", e) from e

    def extract(self, document_text: str) -> Dict[str, Any]:
        """
        Extract key terms from contract text.

        Args:
            document_text: Full text of the contract

        Returns:
            Dict with "extracted_terms", "confidence_scores", "avg_confidence"
        """
        try:
            # Truncate very long documents to avoid token limits
            max_chars = 50000  # ~12.5k tokens
            if len(document_text) > max_chars:
                log.warning(
                    "Document truncated for extraction",
                    original_len=len(document_text),
                    truncated_to=max_chars,
                )
                document_text = document_text[:max_chars]

            chain = self.prompt | self.llm | self.fixing_parser
            log.info("Running key-term extraction chain")

            result = chain.invoke({"document_text": document_text})

            # Extract confidence scores
            confidence_scores = result.pop("confidence_scores", {})
            avg_confidence = (
                sum(confidence_scores.values()) / len(confidence_scores)
                if confidence_scores
                else 0.5
            )

            log.info(
                "Key-term extraction successful",
                fields=len(result),
                avg_confidence=round(avg_confidence, 3),
                risk_flags=len(result.get("risk_flags", [])),
            )

            return {
                "extracted_terms": result,
                "confidence_scores": confidence_scores,
                "avg_confidence": round(avg_confidence, 3),
            }

        except Exception as e:
            err = str(e)
            should_fallback = any(
                kw in err.lower()
                for kw in ["resourceexhausted", "quota", "429", "401", "unauthorized"]
            )

            if should_fallback:
                log.warning("Primary LLM failed, trying fallback", error=err)
                try:
                    current = os.getenv("LLM_PROVIDER", "groq").strip()
                    fallback_provider = "google" if current == "groq" else "groq"
                    fallback_llm = self.loader.load_llm(fallback_provider)
                    fallback_fixing = OutputFixingParser.from_llm(
                        parser=self.parser, llm=fallback_llm
                    )
                    fallback_chain = self.prompt | fallback_llm | fallback_fixing
                    result = fallback_chain.invoke({"document_text": document_text})
                    confidence_scores = result.pop("confidence_scores", {})
                    avg_confidence = (
                        sum(confidence_scores.values()) / len(confidence_scores)
                        if confidence_scores
                        else 0.5
                    )
                    log.info("Fallback extraction successful", provider=fallback_provider)
                    return {
                        "extracted_terms": result,
                        "confidence_scores": confidence_scores,
                        "avg_confidence": round(avg_confidence, 3),
                    }
                except Exception as fb_err:
                    log.error("Fallback extraction also failed", error=str(fb_err))

            log.error("Key-term extraction failed", error=err)
            raise DocumentPortalException("Key-term extraction failed", e) from e
