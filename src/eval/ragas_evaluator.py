"""
Evaluation Pipeline for ContractIQ
==========================================
Generates answers to a standard evaluation suite using your RAG pipeline
and optionally scores them using RAGAS.
"""

import os
import sys
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

# Attempt to load RAGAS (optional depending on installation)
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.run_config import RunConfig
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    log.warning("ragas or datasets library not installed. Scoring will be skipped.")

class ContractEvalSuite:
    """
    Evaluation suite for legal contract QnA using a provided golden dataset.
    """

    @staticmethod
    def run_eval_with_rag(
        rag_instance,
        test_cases: Optional[List[Dict]] = None,
        save_results: bool = True,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Run evaluation by passing test questions through your RAG pipeline,
        then optionally score them via RAGAS.
        """
        if not test_cases:
            raise ValueError("No evaluation dataset provided. You MUST provide a custom golden dataset file (e.g., 40 questions).")
            
        cases = test_cases
        results = []
        
        # Data dictionary constructed for RAGAS
        data_for_ragas = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        log.info("Running ContractIQ eval suite (Generation Phase)", cases=len(cases))

        for case in cases:
            try:
                # Run question through RAG pipeline
                result = rag_instance.invoke(
                    question=case["question"],
                    chat_history=[],
                )

                answer = result.get("answer", "")
                
                # Extract contexts from the source docs
                source_docs = result.get("source_documents", [])
                contexts = [doc.page_content for doc in source_docs] if source_docs else [""]

                results.append({
                    "question": case["question"],
                    "answer": answer,
                    "ground_truth": case.get("ground_truth", "")
                })
                
                # Append for scoring
                data_for_ragas["question"].append(case["question"])
                data_for_ragas["answer"].append(answer)
                data_for_ragas["contexts"].append(contexts)
                data_for_ragas["ground_truth"].append(case.get("ground_truth", ""))

            except Exception as e:
                log.warning("Test case failed - skipping", question=case["question"], error=str(e))
                results.append({
                    "question": case["question"],
                    "answer": f"ERROR: {str(e)}",
                    "ground_truth": case.get("ground_truth", "")
                })
                
            # Sleep 60 seconds to avoid hitting Gemini free-tier rate limits (15 RPM)
            # time.sleep(60)

        scores = {}
        
        if RAGAS_AVAILABLE and len(data_for_ragas["question"]) > 0:
            log.info("Starting RAGAS Scoring Phase...")
            try:
                dataset = Dataset.from_dict(data_for_ragas)
                metrics = [faithfulness, answer_relevancy] # Reduced metrics to limit API usage
                
                # Use HuggingFace for evaluation
                from langchain_huggingface import HuggingFaceEndpoint
                
                # Retrieve Hugging Face API key
                hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
                if not hf_token:
                    log.warning("Hugging Face API key not found. Using OpenAI as fallback to avoid n>1 errors.")
                    from langchain_openai import ChatOpenAI
                    eval_llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0,
                        max_retries=10
                    )
                else:
                    eval_llm = HuggingFaceEndpoint(
                        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                        task="text-generation",
                        huggingfacehub_api_token=hf_token,
                        temperature=0.1,
                        max_new_tokens=1024,
                    )
                
                os.environ["LANGCHAIN_TRACING_V2"] = "false"

                eval_kwargs = {
                    "dataset": dataset,
                    "metrics": metrics,
                    "llm": eval_llm,
                    "embeddings": rag_instance.model_loader.load_embeddings(),
                    # limit max_workers to 1 to reduce parallel rate limit errors during RAGAS evaluations
                    "run_config": RunConfig(max_workers=1, max_retries=10)
                }
                
                # By specifying ChatGroq, Ragas will automatically use it (via GROQ_API_KEY).

                evaluation = evaluate(**eval_kwargs)
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                
                try:
                    scores = {k: v for k, v in evaluation.items()}
                except Exception:
                    scores = {"metrics": str(evaluation)}
                log.info("RAGAS numerical scores calculated successfully", scores=scores)
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                log.error("RAGAS Evaluation failed.", error=str(e), exc_info=True)
                scores = {"error": f"Evaluation crashed: {str(e)}", "traceback": error_trace}
        else:
            log.info("Skipping RAGAS scoring (ragas not installed or no valid answers)")

        if save_results:
            results_dir = Path("eval_results")
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"eval_output_{timestamp}.json"
            
            output_data = {
                "generated_results": results,
                "ragas_scores": scores
            }
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            log.info("Evaluation results and scores saved", file=str(filename))

        return results, scores
