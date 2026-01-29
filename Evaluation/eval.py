import json
import os
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config
from utils.rag_chain import InsuranceRAGChain
from utils.vector_store import VectorStoreManager


class SimpleRAGEvaluator:

    def __init__(
        self,
        rag_chain: InsuranceRAGChain,
        api_key: str,
        backup_key: str = None
    ):
        self.rag_chain = rag_chain
        self.api_key = api_key
        self.backup_key = backup_key
        self.using_backup = False

        self._init_llm()

        print("Simple Evaluator initialized")

    def _init_llm(self):
        key = self.backup_key if self.using_backup else self.api_key
        key_num = 2 if self.using_backup else 1
        print(f" Using API Key #{key_num}")

        self.eval_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=key,
            temperature=0.0,
        )

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:

        contexts_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are evaluating a RAG system's answer for FAITHFULNESS.

RETRIEVED CONTEXTS:
{contexts_text}

ANSWER TO EVALUATE:
{answer}

TASK: Determine if the answer is faithful to the retrieved contexts.
An answer is faithful if:
1. All claims in the answer are supported by the contexts
2. No information is hallucinated or made up
3. The answer doesn't contradict the contexts

Rate the faithfulness on a scale of 0.0 to 1.0:
- 1.0 = Completely faithful, all claims supported by contexts
- 0.7-0.9 = Mostly faithful, minor unsupported details
- 0.4-0.6 = Partially faithful, some claims not in contexts
- 0.0-0.3 = Not faithful, mostly hallucinated

Respond in JSON format:
{{
  "score": <float between 0 and 1>,
  "explanation": "<brief explanation>",
  "unsupported_claims": ["<claim 1>", "<claim 2>"] or []
}}"""

        try:
            response = self.eval_llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result

        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            return {"score": None, "explanation": f"Error: {str(e)}", "unsupported_claims": []}

    def evaluate_relevancy(
        self,
        question: str,
        answer: str
    ) -> Dict[str, Any]:

        prompt = f"""You are evaluating a RAG system's answer for RELEVANCY.

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

TASK: Determine if the answer is relevant and addresses the question.
An answer is relevant if:
1. It directly addresses what was asked
2. It stays on topic
3. It doesn't include excessive irrelevant information

Rate the relevancy on a scale of 0.0 to 1.0:
- 1.0 = Perfectly relevant, directly answers the question
- 0.7-0.9 = Mostly relevant, answers with minor tangents
- 0.4-0.6 = Partially relevant, misses key aspects
- 0.0-0.3 = Not relevant, doesn't address the question

Respond in JSON format:
{{
  "score": <float between 0 and 1>,
  "explanation": "<brief explanation>",
  "missing_aspects": ["<aspect 1>", "<aspect 2>"] or []
}}"""

        try:
            response = self.eval_llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result

        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            return {"score": None, "explanation": f"Error: {str(e)}", "missing_aspects": []}

    def evaluate_correctness(
        self,
        answer: str,
        ground_truth: str
    ) -> Dict[str, Any]:
 
        prompt = f"""You are evaluating a RAG system's answer for CORRECTNESS.

GROUND TRUTH (Expected Answer):
{ground_truth}

ANSWER TO EVALUATE:
{answer}

TASK: Determine if the answer is factually correct compared to the ground truth.
An answer is correct if:
1. Key facts match the ground truth
2. No incorrect information is presented
3. Core meaning aligns with ground truth (exact wording not required)

Rate the correctness on a scale of 0.0 to 1.0:
- 1.0 = Completely correct, matches ground truth
- 0.7-0.9 = Mostly correct, minor differences
- 0.4-0.6 = Partially correct, some wrong facts
- 0.0-0.3 = Incorrect, contradicts ground truth

Respond in JSON format:
{{
  "score": <float between 0 and 1>,
  "explanation": "<brief explanation>",
  "incorrect_facts": ["<fact 1>", "<fact 2>"] or []
}}"""

        try:
            response = self.eval_llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result

        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            return {"score": None, "explanation": f"Error: {str(e)}", "incorrect_facts": []}

    def evaluate_single_question(
        self,
        question: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}")

        # Get RAG response
        print("Getting RAG response...")
        response = self.rag_chain.query(question, return_sources=True)

        answer = response["answer"]
        contexts = []
        if "source_documents" in response and response["source_documents"]:
            contexts = [doc.page_content for doc in response["source_documents"]]

        if not contexts:
            contexts = [answer]

        print(f"Answer: {len(answer)} chars")
        print(f"Contexts: {len(contexts)} chunks")

        # Evaluate metrics
        print("\n Evaluating metrics...")

        print("   1. Faithfulness...", end=" ", flush=True)
        faithfulness = self.evaluate_faithfulness(answer, contexts)
        print(f"{faithfulness['score']:.3f}" if faithfulness['score'] else "No")

        print("   2. Relevancy...", end=" ", flush=True)
        relevancy = self.evaluate_relevancy(question, answer)
        print(f"{relevancy['score']:.3f}" if relevancy['score'] else "No")

        print("   3. Correctness...", end=" ", flush=True)
        correctness = self.evaluate_correctness(answer, ground_truth)
        print(f"{correctness['score']:.3f}" if correctness['score'] else "No")

        # Compile results
        result = {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts_count": len(contexts),
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "correctness": correctness,
            "timestamp": datetime.now().isoformat()
        }

        # Display summary
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        metrics = [
            ("Faithfulness", faithfulness),
            ("Relevancy", relevancy),
            ("Correctness", correctness)
        ]

        for name, metric in metrics:
            if metric['score'] is not None:
                score = metric['score']
                status = "Yes" if score >= 0.7 else "Caution" if score >= 0.5 else "No"
                print(f"\n{name}: {score:.4f} {status}")
                print(f"  → {metric['explanation']}")
            else:
                print(f"\n{name}: Failed")

        return result

    def run_batch_evaluation(
        self,
        test_cases: List[Dict[str, str]] = None,
        num_questions: int = 5
    ) -> pd.DataFrame:
        print("\n" + "="*70)
        print("BATCH EVALUATION")
        print("="*70)

        if test_cases is None:
            test_cases = [
            {
                "question": "What is IDV?",
                "ground_truth": "IDV (Insured's Declared Value) is the maximum sum insured fixed for the vehicle at the beginning of each policy period."
            },
            {
                "question": "What are the main policy exclusions?",
                "ground_truth": "Main exclusions include wear and tear, mechanical breakdown, pre-existing damage, consequential losses, and driving without valid license."
            },
            {
                "question": "Are natural disasters covered?",
                "ground_truth": "Yes, natural disasters like floods, earthquakes, storms are covered under comprehensive own damage coverage."
            },
            {
                "question": "What is comprehensive insurance?",
                "ground_truth": "Comprehensive insurance covers both own damage to the vehicle and third-party liability for injury or property damage."
            },
            {
                "question": "How does the No Claim Bonus work?",
                "ground_truth": "No Claim Bonus (NCB) is a discount on premium for claim-free years, ranging from 20% to 50% based on consecutive claim-free years."
            }
            ][:num_questions]

        print(f" Evaluating {len(test_cases)} questions\n")

        all_results = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'#'*70}")
            print(f"Question {i}/{len(test_cases)}")
            print(f"{'#'*70}")

            result = self.evaluate_single_question(
                question=test_case["question"],
                ground_truth=test_case["ground_truth"]
            )

            all_results.append(result)

            # Pause between questions
            if i < len(test_cases):
                print("\n Pausing before next question...")
                import time
                time.sleep(2)

        # Create DataFrame
        df_data = []
        for r in all_results:
            row = {
                "question": r["question"],
                "answer_preview": r["answer"][:100] + "...",
                "faithfulness": r["faithfulness"]["score"],
                "relevancy": r["relevancy"]["score"],
                "correctness": r["correctness"]["score"],
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Summary
        print("\n" + "="*70)
        print(" OVERALL SUMMARY")
        print("="*70)

        for metric in ["faithfulness", "relevancy", "correctness"]:
            valid = df[metric].dropna()
            if len(valid) > 0:
                avg = valid.mean()
                status = "Yes" if avg >= 0.7 else "Caution" if avg >= 0.5 else "No"
                print(f"{metric.capitalize():20s}: {avg:.4f} {status} ({len(valid)}/{len(df)} valid)")

        # Overall score
        metric_cols = ["faithfulness", "relevancy", "correctness"]
        overall = df[metric_cols].mean().mean()
        print(f"\n{'Overall Score':20s}: {overall:.4f}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        with open("results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Save summary CSV
        df.to_csv("eval_summary.csv", index=False)

        print(f"\n Results saved:")
        print(f"   - eval_detailed_{timestamp}.json")
        print(f"   - eval_summary_{timestamp}.csv")

        return df


def main():

    # Get API keys
    api_key = Config.GEMINI_API_KEY

    print("\n Initializing RAG system...")
    vector_store = VectorStoreManager()
    rag_chain = InsuranceRAGChain(vector_store_manager=vector_store)

    # Initialize evaluator
    evaluator = SimpleRAGEvaluator(
        rag_chain=rag_chain,
        api_key=api_key
    )

    evaluator.run_batch_evaluation(num_questions=5)

    print("\n Evaluation complete!")


if __name__ == "__main__":
    main()