import base64
import hashlib
import json
import time
import re
import textwrap
from typing import Literal, TypedDict

import litellm
import pandas as pd
from pydantic import BaseModel
from transformers import AutoTokenizer
from datasets import load_dataset
from nltk.tokenize import NLTKWordTokenizer


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()



DSQA_GRADER_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a JSON dictionary with the following top-level keys: `"explanation"` (a string), `"correctness_details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"excessive_answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")

DSQA_GRADER_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
  "correctness_details": {{
    "Belgium": true,
    "France": true
  }},
  "excessive_answers": [ "Italy" ]
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


class CorrectnessItem(BaseModel):
    key: str
    value: bool

class DSQAExtractedResult(BaseModel):
    explanation: str
    correctness_details: list[CorrectnessItem]
    excessive_answers: list[str]
    strict: Literal[True]


# helper functions from DSQA
def _calculate_metric(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    """Calculates precision, recall, and F1."""
    precision_val = 0.0
    if (true_positives + false_positives) > 0:
        precision_val = true_positives / (true_positives + false_positives)

    recall_val = 0.0
    if (true_positives + false_negatives) > 0:
        recall_val = true_positives / (true_positives + false_negatives)

    f1_score_val = 0.0
    if (precision_val + recall_val) > 0:
        f1_score_val = (
            2 * (precision_val * recall_val) / (precision_val + recall_val)
        )

    return {
        'precision': precision_val,
        'recall': recall_val,
        'f1_score': f1_score_val,
    }


def calculate_dsqa_metrics(extracted_result: DSQAExtractedResult) -> dict:
    # Extract correctness details
    details = extracted_result.correctness_details
    expected_correct_answer_list = [item.key for item in details]
    ratings = [item.value for item in details]
    
    num_correct = sum(ratings)
    true_positive = num_correct
    false_negative = len(ratings) - num_correct
    
    has_expected_answers = bool(ratings)

    all_expected_answers_correct = False
    fully_incorrect = 0
    if has_expected_answers:
        all_expected_answers_correct = num_correct == len(ratings)
        if num_correct == 0:
            fully_incorrect = 1

    # Extract excessive answers
    excessive_answers = extracted_result.excessive_answers

    has_excessive_answers = bool(excessive_answers)
    false_positives = 0
    correct_with_excessive_answers = 0
    if has_excessive_answers:
        false_positives = len(excessive_answers)
        if (all_expected_answers_correct or not has_expected_answers):
            correct_with_excessive_answers = 1

    is_all_correct = (
        all_expected_answers_correct or not has_expected_answers
    ) and not has_excessive_answers

    per_item_metric = _calculate_metric(true_positive, false_positives, false_negative)

    return {
        "all_correct": is_all_correct,
        "correct_with_excessive_answers": correct_with_excessive_answers,
        "fully_incorrect": fully_incorrect,
        **per_item_metric,
    }


class SeedDatum(TypedDict):
    document: str
    url: str

def load_seed_dataset(split: Literal['fineweb', 'bcplus'], num_samples: int = 10000) -> list[SeedDatum]:
    if split == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split='train')
    elif split == "bcplus":
        ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    else:
        raise ValueError(f"Unknown dataset: {split}")

    # truncate documents to at most 4k words
    tokenizer = NLTKWordTokenizer()
    def truncate(x):
        tokens = list(tokenizer.span_tokenize(x['text']))
        if len(tokens) > 4096:
            return {"text": x['text'][tokens[4096][0]] + "... [truncated]"}
        return x
    ds = ds.map(truncate)
    # from manual inspection, there are a few documents (<10) that are too long by tokens. those look pretty noisy anyway so let's just filter them out
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    # not too short, not too long
    ds = ds.filter(lambda x: 128 <= len(tokenizer(x['text']).input_ids) <= 8192)
    sample_dataset = ds.shuffle(seed=42).select(range(num_samples))
    return [{"document": item["text"], "url": item["url"]} for item in sample_dataset]


class QADatum(TypedDict):
    question: str
    answer: str


def _derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def load_qa_dataset(split: str) -> list[QADatum]:
    if split == "browsecomp":
        path = "/home/hyen/project/simple-evals/data/browse_comp_test_set_heldout.csv"
        df = pd.read_csv(path)
        qa = []
        for _, row in df.iterrows():
            problem = _decrypt(row.get("problem", ""), row.get("canary", ""))
            answer = _decrypt(row.get("answer", ""), row.get("canary", ""))
            qa.append({"question": problem, "answer": answer})
    
    elif "browsecomp_plus" in  split:
        path = f"/home/hyen/project/simple-evals/data/{split}.jsonl"
        with open(path, "r") as f:
            bc_plus = [json.loads(line.strip()) for line in f]
        qa = [{"question": item["query"], "answer": item["answer"]} for item in bc_plus]

    elif split == "dsqa":
        path = "/home/hyen/project/simple-evals/data/DSQA-full.csv"
        df = pd.read_csv(path)
        # rename problem to question
        df = df.rename(columns={"problem": "question"})
        qa = [row.to_dict() for _, row in df.iterrows()]

    else:
        raise ValueError(f"Unknown dataset: {split}")

    return qa


def api_call_with_retry(model: str, messages: list[dict], response_format: BaseModel | None = None):
    models = ['azure/gpt-4.1', 'azure/gpt-5.2', 'azure/gpt-5', 'azure/gpt-4o', 'azure/o3', 'azure/o4-mini', 'azure/gpt-4.1-mini', 'openai/gpt-4.1-2025-04-14']
    for m in models:
        for attempt in range(3):
            try:
                response = litellm.completion(model=m, messages=messages, response_format=response_format)
                return response
            except Exception:
                time.sleep((attempt + 1) * 2)

    raise Exception(f"Failed to call API after trying all models {models}")


def grade_answer(problem: dict, response: str) -> tuple[float, dict]:
    # return (primary_score, dict of secondary scores)
    # grades the response to the problem, support rubric-based grading like for DSQA
    answer = problem["answer"]

    if problem.get("answer_type") is not None:
        # rubric-based grading for DSQA
        grading_response = api_call_with_retry(
            # model="azure/gpt-4.1", 
            model="openai/gpt-4.1-2025-04-14", 
            messages=[{"role": "user", "content": DSQA_GRADER_PROMPT + DSQA_GRADER_OUTPUT_EXAMPLE.format(
                    prompt=problem["question"], prompt_type=problem["answer_type"], answer=answer, response=response, 
            )}],
            response_format=DSQAExtractedResult,
        )

        grading_response = grading_response['choices'][0]['message']['content']
        extracted_result = DSQAExtractedResult.model_validate_json(grading_response)
        metrics = calculate_dsqa_metrics(extracted_result)
        # primary score is f1 score
        return metrics['f1_score'], metrics

    else:
        # single answer grading
        grading_response = api_call_with_retry(
            # model="azure/gpt-4.1", 
            model="openai/gpt-4.1-2025-04-14", 
            messages=[{"role": "user", "content": GRADER_TEMPLATE.format(
                question=problem["question"], response=response, correct_answer=answer,
            )}],
            response_format=None,
        )
        # primary score is correctness score
        grading_response = grading_response['choices'][0]['message']['content']
        correct_match = re.search(r"correct: (yes|no)", grading_response)
        score = 1.0 if correct_match and correct_match.group(1) == "yes" else 0.0
        confidence_match = re.search(r"confidence: (\d+)", grading_response)
        confidence = int(confidence_match.group(1)) if confidence_match else 100
        return score, {"correctness": score, "confidence": confidence}
