import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import SummaryResponse,PromptType

class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.fixing_parser
        log.info("DocumentComparatorLLM initialized", model=self.llm)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }

            log.info("Invoking document comparison LLM chain")
            response = self.chain.invoke(inputs)
            log.info("Chain invoked successfully", response_preview=str(response)[:200])
            return self._format_response(response)
        except Exception as e:
            log.error("Error in compare_documents", error=str(e))
            raise DocumentPortalException("Error comparing documents", sys)

    def _is_no_change_text(self, text: str) -> bool:
        """Helper to detect if a text represents a 'no change' statement."""
        clean = text.lower().replace("•", "").replace("-", "").strip().rstrip(".")
        no_change_phrases = {
            "no change",
            "no changes",
            "no changes detected",
            "no changes found",
            "no change detected",
            "no change found",
            "no deviation",
            "no deviations",
            "no modification",
            "no modifications",
            "no difference",
            "no differences",
            "remains the same",
            "remain the same",
            "remains unchanged",
            "remain unchanged",
            "same as reference",
            "same as the reference",
            "same as original",
            "same as the original",
            "unchanged",
            "nothing changed",
            "not changed"
        }
        if clean in no_change_phrases:
            return True
        if clean.endswith("remains the same") or clean.endswith("remain the same") or clean.endswith("remains unchanged") or clean.endswith("remain unchanged"):
            return True
        return False

    def _format_response(self, response_parsed) -> pd.DataFrame:
        try:
            # First, check if it's a Pydantic object and convert it to native Python
            if hasattr(response_parsed, "model_dump"):
                response_parsed = response_parsed.model_dump()
            elif hasattr(response_parsed, "dict"):
                response_parsed = response_parsed.dict()

            data = []
            if isinstance(response_parsed, list):
                data = response_parsed
            elif isinstance(response_parsed, dict):
                # Check for common list wrapper keys case-insensitively
                for key in ["changes", "Changes", "rows", "Rows", "root", "Root"]:
                    if key in response_parsed and isinstance(response_parsed[key], list):
                        data = response_parsed[key]
                        break
                if not data:
                    data = [response_parsed]
            else:
                data = []

            cleaned = []
            for item in data:
                if isinstance(item, dict):
                    page_val = item.get("Page", item.get("page", ""))
                    page_val = str(page_val).strip() if page_val is not None else ""
                    changes_val = item.get("Changes", item.get("changes", ""))
                    # If changes_val is a dictionary (common parser quirk), extract its value
                    if isinstance(changes_val, dict):
                        changes_val = changes_val.get("value", changes_val.get("Changes", changes_val.get("changes", str(changes_val))))

                    # If changes_val is a list, format each as a bullet point
                    if isinstance(changes_val, list):
                        formatted_list = []
                        for x in changes_val:
                            x_str = str(x).strip()
                            sub_parts = [part.strip() for part in x_str.split("•") if part.strip()]
                            if not sub_parts:
                                sub_parts = [x_str]
                            for part in sub_parts:
                                clean_part = part.lstrip("•- ")
                                if clean_part and not self._is_no_change_text(clean_part):
                                    formatted_list.append(f"• {clean_part}")
                        changes_val = "\n".join(formatted_list)
                    elif isinstance(changes_val, str):
                        raw_lines = [line.strip() for line in changes_val.split("\n") if line.strip()]
                        formatted_lines = []
                        for raw_line in raw_lines:
                            sub_lines = [part.strip() for part in raw_line.split("•") if part.strip()]
                            if not sub_lines:
                                sub_lines = [raw_line]
                            for sub_line in sub_lines:
                                clean_line = sub_line.lstrip("•- ")
                                if clean_line and not self._is_no_change_text(clean_line):
                                    formatted_lines.append(f"• {clean_line}")
                        changes_val = "\n".join(formatted_lines)

                    if page_val and changes_val.strip():
                        cleaned.append({
                            "Page": str(page_val) if page_val is not None else "",
                            "Changes": str(changes_val) if changes_val is not None else ""
                        })

            df = pd.DataFrame(cleaned, columns=["Page", "Changes"])
            return df
        except Exception as e:
            log.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)
