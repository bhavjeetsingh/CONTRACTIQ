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
    Pre-built evaluation suite for legal contract QnA.
    Contains standard test cases covering common contract questions.
    """

    STANDARD_TEST_CASES = [
        {
            "question": "Who are the parties involved in this agreement?",
            "ground_truth": "The names and roles of all parties in the contract.",
        },
        {
            "question": "What are the payment terms?",
            "ground_truth": "The payment schedule, amounts, and conditions specified.",
        },
    ]

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
        cases = test_cases or ContractEvalSuite.STANDARD_TEST_CASES
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
            
            # Rate limiting mitigation
            log.info("Sleeping for 60 seconds to respect API rate limits...")
            time.sleep(60)

        scores = {}
        
        if RAGAS_AVAILABLE and len(data_for_ragas["question"]) > 0:
            log.info("Sleeping before RAGAS to reset rate limit quota...")
            time.sleep(60)
            log.info("Starting RAGAS Scoring Phase. (This might take a while and makes additional API calls)...")
            try:
                dataset = Dataset.from_dict(data_for_ragas)
                metrics = [faithfulness, answer_relevancy] # Reduced metrics to limit API usage
                
                eval_kwargs = {
                    "dataset": dataset,
                    "metrics": metrics,
                    # limit max_workers to 1 to reduce parallel rate limit errors during RAGAS evaluations
                    "run_config": RunConfig(max_workers=1, max_retries=3)
                }
                
                # Optional: Explicitly pass your loaded LLM / Embeddings if required by Ragas 
                if rag_instance.llm:
                    eval_kwargs["llm"] = rag_instance.llm
                if hasattr(rag_instance, "model_loader"):
                    eval_kwargs["embeddings"] = rag_instance.model_loader.load_embeddings()

                evaluation = evaluate(**eval_kwargs)
                scores = dict(evaluation)
                log.info("RAGAS numerical scores calculated successfully", scores=scores)
            except Exception as e:
                log.error("RAGAS Evaluation failed. This is often due to missing OpenAI keys or Rate Limits.", error=str(e))
                scores = {"error": f"Evaluation crashed: {str(e)}"}
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
