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
                    changes_val = item.get("Changes", item.get("changes", ""))
                    
                    # If changes_val is a list of strings, join it with newlines
                    if isinstance(changes_val, list):
                        changes_val = "\n".join(f"• {str(x).lstrip('•- ')}" for x in changes_val)
                    elif isinstance(changes_val, str):
                        # Ensure formatting has clean bullet points
                        lines = [line.strip() for line in changes_val.split("\n") if line.strip()]
                        formatted_lines = []
                        for line in lines:
                            if not line.startswith("•") and not line.startswith("-"):
                                formatted_lines.append(f"• {line}")
                            else:
                                formatted_lines.append(f"• {line.lstrip('•- ')}")
                        changes_val = "\n".join(formatted_lines)

                    if page_val or changes_val:
                        cleaned.append({
                            "Page": str(page_val) if page_val is not None else "",
                            "Changes": str(changes_val) if changes_val is not None else ""
                        })

            df = pd.DataFrame(cleaned, columns=["Page", "Changes"])
            return df
        except Exception as e:
            log.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)
