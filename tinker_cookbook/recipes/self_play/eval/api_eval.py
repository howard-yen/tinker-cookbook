import asyncio
import json
import logging
import os
import random
import re
from typing import TypedDict

import chz
import litellm
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook.utils import logtree
from tinker_cookbook.recipes.self_play.search_env import SOLVER_SYSTEM_PROMPT, CHALLENGER_SYSTEM_PROMPT
from tinker_cookbook.recipes.self_play.search_utils import WebSearchTool, WebSearchToolConfig
from tinker_cookbook.recipes.self_play.utils import QADatum, load_qa_dataset, grade_answer, SeedDatum, load_seed_dataset
from tinker_cookbook.recipes.self_play.eval.config import APIEvalConfig


# Suppress LiteLLM INFO messages
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

logging.basicConfig(
    level=int(os.getenv("LOG_LEVEL", logging.INFO)),
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def tool_spec_to_openai_tool(spec: dict) -> dict:
    """Convert an internal ToolSpec to OpenAI function-calling format."""
    function_def = {
        "name": spec["name"],
        "description": spec["description"],
        "parameters": spec["parameters"],
    }
    return {"type": "function", "function": function_def}


async def execute_tool_call(tool_call, web_tool: WebSearchTool, conn_semaphore: asyncio.Semaphore) -> str:
    """Dispatch a litellm tool call to the appropriate WebSearchTool method."""
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    async with conn_semaphore:
        if name == "search":
            query_list, error = web_tool.validate_and_extract_search_args(arguments)
            if error:
                return error
            return await web_tool.batch_search(query_list)
        elif name == "visit":
            visit_args, error = web_tool.validate_and_extract_visit_args(arguments)
            if error:
                return error
            url_list, query_list = visit_args
            return await web_tool.batch_browse(url_list, query_list)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})


def extract_answer(text: str) -> str | None:
    """Extract the answer field from the last JSON block in text."""
    matches = re.findall(r"\{.*\}", text, re.DOTALL)
    if matches:
        try:
            parsed = json.loads(matches[-1])
            return parsed.get("answer")
        except json.JSONDecodeError:
            return None
    return None


def extract_json_output(text: str) -> dict | None:
    """Extract the full JSON dict from the last JSON block in text."""
    matches = re.findall(r"\{.*\}", text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            return None
    return None


class EvaluationResult(TypedDict):
    question: str
    correct_score: float
    model_answer: str | None
    num_tool_calls: int


async def run_agentic_loop(
    system_prompt: str,
    user_message: str,
    web_tool: WebSearchTool,
    config: APIEvalConfig,
    conn_semaphore: asyncio.Semaphore,
) -> tuple[str | None, int, list[dict]]:
    """Run a litellm agentic tool-calling loop.

    Returns (final_text_response, num_tool_calls, messages).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    tool_schemas = web_tool.get_tool_schemas()
    tools = [tool_spec_to_openai_tool(spec) for spec in tool_schemas]

    num_calls = 0
    final_text = None

    for turn in range(config.max_num_calls + 1):
        send_tools = tools if num_calls < config.max_num_calls else None

        response = await litellm.acompletion(
            model=config.model,
            messages=messages,
            tools=send_tools,
            max_tokens=config.max_tokens,
        )

        choice = response.choices[0]
        assistant_message = choice.message
        messages.append(assistant_message.model_dump(exclude_none=True))

        # Track token usage from the response
        total_tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0

        if assistant_message.tool_calls:
            tool_results = []
            for tc in assistant_message.tool_calls:
                num_calls += 1
                with logtree.scope_details(f"Tool call [{num_calls}]: {tc.function.name}"):
                    logtree.log_text(f"Arguments: {tc.function.arguments}")
                    result = await execute_tool_call(tc, web_tool, conn_semaphore)
                    logtree.log_text(f"Result: {str(result)[:500]}")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
            messages.extend(tool_results)

            # If we're approaching the trajectory limit, force a final text response on the next turn
            if total_tokens + config.max_tokens > config.max_trajectory_tokens:
                logtree.log_text(f"Approaching trajectory limit ({total_tokens}/{config.max_trajectory_tokens} tokens), forcing final response")
                response = await litellm.acompletion(
                    model=config.model,
                    messages=messages,
                    tools=None,
                    max_tokens=config.max_tokens,
                )
                final_text = response.choices[0].message.content or ""
                logtree.log_text(f"Model response (forced): {final_text[:500]}")
                break
        else:
            final_text = assistant_message.content or ""
            logtree.log_text(f"Model response: {final_text[:500]}")
            break

    # If we exhausted turns without a text response, try to extract from last message
    if final_text is None:
        last_content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        final_text = last_content or None

    return final_text, num_calls, messages


async def evaluate_single_item(
    item: QADatum,
    idx: int,
    total: int,
    web_tool: WebSearchTool,
    config: APIEvalConfig,
    semaphore: asyncio.Semaphore,
    conn_semaphore: asyncio.Semaphore,
) -> dict:
    """Run the agentic loop for a single eval item using litellm."""
    async with semaphore:
        with logtree.scope_details(f"[{idx}/{total}] Eval: {item['question'][:80]}"):
            logtree.log_text(f"Question: {item['question']}")

            try:
                final_text, num_calls, messages = await run_agentic_loop(
                    system_prompt=SOLVER_SYSTEM_PROMPT,
                    user_message=item["question"],
                    web_tool=web_tool,
                    config=config,
                    conn_semaphore=conn_semaphore,
                )
            except Exception as e:
                logger.warning(f"Error evaluating item: {e}")
                logtree.log_text(f"Error: {e}")
                return {
                    "idx": idx,
                    "question": item["question"],
                    "correct_score": 0.0,
                    "model_answer": None,
                    "num_tool_calls": 0,
                    "messages": [],
                }

            model_answer = extract_answer(final_text) if final_text else None

            # Grade the answer
            answer_str = model_answer or ""
            logtree.log_text(f"Extracted answer: {answer_str}")
            logtree.log_text(f"Gold answer: {item['answer']}")

            correctness_reward, grading_metrics = grade_answer(item, answer_str)
            logtree.log_text(f"Correctness: {correctness_reward}")
            logtree.log_text(f"Tool calls: {num_calls}")

            return {
                "idx": idx,
                "question": item["question"],
                "correct_score": correctness_reward,
                "model_answer": model_answer,
                "num_tool_calls": num_calls,
                "messages": messages,
                **grading_metrics,
            }


async def evaluate_one_dataset(
    split: str,
    data: list[QADatum],
    web_tool: WebSearchTool,
    config: APIEvalConfig,
) -> dict:
    """Evaluate all items in a dataset split."""
    log_path = config.log_path_value
    log_path += f"/{split}_samples{config.max_eval_samples}_tools{config.max_num_calls}"
    os.makedirs(log_path, exist_ok=True)
    logtree_path = os.path.join(log_path, "eval.html")

    semaphore = asyncio.Semaphore(config.concurrency)
    conn_semaphore = asyncio.Semaphore(128)

    tasks = [
        evaluate_single_item(item, i + 1, len(data), web_tool, config, semaphore, conn_semaphore)
        for i, item in enumerate(data)
    ]

    logger.info(f"Evaluating {len(tasks)} items for split '{split}'")
    with logtree.init_trace(f"API Evaluation — {split}", path=logtree_path):
        logtree.log_text(f"API evaluation results for split '{split}' using model '{config.model}'")
        results = await tqdm_asyncio.gather(*tasks, desc=f"Eval {split}")

        if not results:
            logger.info(f"Evaluation for '{split}' complete. HTML log saved to {logtree_path}")
            return {"split": split, "total_samples": 0, "accuracy": 0.0}

        skip_keys = {"idx", "question", "model_answer", "messages"}
        numeric_keys = [
            k for k, v in results[0].items()
            if k not in skip_keys and isinstance(v, (int, float))
        ]

        # Log per-instance results table
        QUESTION_SNIPPET_LEN = 80
        table_rows = []
        for r in results:
            row: dict[str, str | float] = {
                "idx": r["idx"],
                "question": r["question"][:QUESTION_SNIPPET_LEN] + ("..." if len(r["question"]) > QUESTION_SNIPPET_LEN else ""),
            }
            for k in numeric_keys:
                val = r.get(k, 0.0)
                row[k] = round(val, 4) if isinstance(val, float) else val
            table_rows.append(row)

        # Append an averages row
        avg_row: dict[str, str | float] = {"idx": "", "question": "AVERAGE"}
        for k in numeric_keys:
            values = [r.get(k, 0.0) for r in results]
            avg_row[k] = round(sum(values) / len(values), 4)
        table_rows.append(avg_row)

        logtree.table(table_rows, caption=f"Results — {split} ({len(results)} samples)")

    # Save per-instance trajectories as JSONL
    jsonl_path = os.path.join(log_path, "trajectories.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info(f"Trajectories saved to {jsonl_path}")

    logger.info(f"Evaluation for '{split}' complete. HTML log saved to {logtree_path}")

    aggregated: dict[str, object] = {"split": split, "total_samples": len(results)}
    for key in numeric_keys:
        values = [r.get(key, 0.0) for r in results]
        aggregated[f"avg_{key}"] = sum(values) / len(values)

    return aggregated


async def generate_single_question(
    item: SeedDatum,
    idx: int,
    total: int,
    web_tool: WebSearchTool,
    config: APIEvalConfig,
    semaphore: asyncio.Semaphore,
    conn_semaphore: asyncio.Semaphore,
) -> dict:
    """Run the challenger agentic loop for a single corpus item to generate a question."""
    async with semaphore:
        with logtree.scope_details(f"[{idx}/{total}] Challenger: {item['url'][:80]}"):
            logtree.log_text(f"URL: {item['url']}")
            logtree.log_text(f"Document: {item['document'][:300]}...")

            try:
                final_text, num_calls, messages = await run_agentic_loop(
                    system_prompt=CHALLENGER_SYSTEM_PROMPT,
                    user_message=f"Document: {item['document']}\nUrl: {item['url']}",
                    web_tool=web_tool,
                    config=config,
                    conn_semaphore=conn_semaphore,
                )
            except Exception as e:
                logger.warning(f"Error generating question: {e}")
                logtree.log_text(f"Error: {e}")
                return {
                    "idx": idx,
                    "url": item["url"],
                    "document_snippet": item["document"][:200],
                    "question": None,
                    "answer": None,
                    "explanation": None,
                    "num_tool_calls": 0,
                    "valid_format": 0.0,
                    "messages": [],
                }

            generated_output = extract_json_output(final_text) if final_text else None
            question = generated_output.get("question") if generated_output else None
            answer = generated_output.get("answer") if generated_output else None
            explanation = generated_output.get("explanation") if generated_output else None
            valid = generated_output is not None and all(
                k in generated_output for k in ("question", "answer", "explanation")
            )

            logtree.log_text(f"Generated question: {question}")
            logtree.log_text(f"Generated answer: {answer}")
            logtree.log_text(f"Valid format: {valid}, Tool calls: {num_calls}")

            return {
                "idx": idx,
                "url": item["url"],
                "document_snippet": item["document"][:200],
                "question": question,
                "answer": answer,
                "explanation": explanation,
                "num_tool_calls": num_calls,
                "valid_format": float(valid),
                "messages": messages,
            }


async def generate_questions_dataset(
    split: str,
    data: list[SeedDatum],
    web_tool: WebSearchTool,
    config: APIEvalConfig,
) -> dict:
    """Generate questions from all corpus items in a split."""
    log_path = config.log_path_value
    log_path += f"/challenger_{split}_samples{config.max_eval_samples}_tools{config.max_num_calls}"
    os.makedirs(log_path, exist_ok=True)
    logtree_path = os.path.join(log_path, "challenger.html")

    semaphore = asyncio.Semaphore(config.concurrency)
    conn_semaphore = asyncio.Semaphore(128)

    tasks = [
        generate_single_question(item, i + 1, len(data), web_tool, config, semaphore, conn_semaphore)
        for i, item in enumerate(data)
    ]

    logger.info(f"Generating questions from {len(tasks)} corpus items for split '{split}'")
    with logtree.init_trace(f"API Challenger — {split}", path=logtree_path):
        logtree.log_text(f"Question generation for corpus split '{split}' using model '{config.model}'")
        results = await tqdm_asyncio.gather(*tasks, desc=f"Challenger {split}")

        if not results:
            logger.info(f"Generation for '{split}' complete. HTML log saved to {logtree_path}")
            return {"split": split, "total_samples": 0, "generated_questions": []}

        # Log per-instance results table
        table_rows = []
        for r in results:
            table_rows.append({
                "idx": r["idx"],
                "url": (r["url"] or "")[:60],
                "question": (r["question"] or "N/A")[:80],
                "answer": (r["answer"] or "N/A")[:40],
                "num_tool_calls": r["num_tool_calls"],
                "valid_format": r["valid_format"],
            })

        valid_count = sum(1 for r in results if r["valid_format"])
        table_rows.append({
            "idx": "",
            "url": "SUMMARY",
            "question": f"{valid_count}/{len(results)} valid",
            "answer": "",
            "num_tool_calls": round(sum(r["num_tool_calls"] for r in results) / len(results), 2),
            "valid_format": round(valid_count / len(results), 4),
        })

        logtree.table(table_rows, caption=f"Generated Questions — {split} ({len(results)} samples)")

    # Save per-instance trajectories as JSONL
    jsonl_path = os.path.join(log_path, "trajectories.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info(f"Trajectories saved to {jsonl_path}")

    logger.info(f"Generation for '{split}' complete. HTML log saved to {logtree_path}")

    valid_count = sum(1 for r in results if r["valid_format"])
    return {
        "split": split,
        "total_samples": len(results),
        "valid_questions": valid_count,
        "valid_rate": valid_count / len(results),
        "avg_num_tool_calls": sum(r["num_tool_calls"] for r in results) / len(results),
        "generated_questions": [
            {"question": r["question"], "answer": r["answer"], "explanation": r["explanation"], "url": r["url"]}
            for r in results if r["valid_format"]
        ],
    }


async def cli_main(config: APIEvalConfig):
    web_tool_config = WebSearchToolConfig(
        port=config.web_tool_port,
        topk=config.web_tool_topk,
        content_length=config.web_tool_content_length,
        scoring_func=config.web_tool_scoring_func,
        chunking_func=config.web_tool_chunking_func,
        timeout=config.web_tool_timeout,
        search_mode=config.search_mode,
    )
    web_tool = WebSearchTool(web_tool_config)

    if config.mode == "challenger":
        all_results = []
        for split in config.corpus_splits:
            data = load_seed_dataset(split, num_samples=max(config.max_eval_samples * 2, 1000))

            random.seed(config.seed)
            sampled_data = random.sample(data, min(config.max_eval_samples, len(data)))

            results = await generate_questions_dataset(split, sampled_data, web_tool, config)
            all_results.append(results)

        # Print results table
        print("\nCHALLENGER RESULTS")
        col_width = 20
        table_keys = ["total_samples", "valid_questions", "valid_rate", "avg_num_tool_calls"]
        header = f"{'Corpus':<{col_width}}" + "".join(f"{k:>{col_width}}" for k in table_keys)
        print(header)
        print("-" * len(header))
        for results in all_results:
            row = f"{results['split']:<{col_width}}"
            for k in table_keys:
                v = results.get(k, 0)
                row += f"{v:>{col_width}.3f}" if isinstance(v, float) else f"{v:>{col_width}}"
            print(row)
        print("-" * len(header))

        # Print generated questions
        for results in all_results:
            print(f"\nGenerated Questions ({results['split']}):")
            for i, q in enumerate(results.get("generated_questions", []), 1):
                print(f"  {i}. Q: {q['question']}")
                print(f"     A: {q['answer']}")
                print()

        # Save results to file
        os.makedirs(os.path.dirname(config.output_file_value), exist_ok=True)
        with open(config.output_file_value, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to {config.output_file_value}")

    else:
        all_results = []
        for split in config.splits:
            data = load_qa_dataset(split)

            random.seed(config.seed)
            sampled_data = random.sample(data, min(config.max_eval_samples, len(data)))

            results = await evaluate_one_dataset(split, sampled_data, web_tool, config)
            all_results.append(results)

        # Print results table
        print("\nEVALUATION RESULTS")
        col_width = 20
        table_keys = ["total_samples", "avg_correct_score"]
        header = f"{'Task':<{col_width}}" + "".join(f"{k:>{col_width}}" for k in table_keys)
        print(header)
        print("-" * len(header))
        for results in all_results:
            row = f"{results['split']:<{col_width}}"
            for k in table_keys:
                v = results.get(k, 0)
                row += f"{v:>{col_width}.3f}" if isinstance(v, float) else f"{v:>{col_width}}"
            print(row)
        print("-" * len(header))

        # Save results to file
        os.makedirs(os.path.dirname(config.output_file_value), exist_ok=True)
        with open(config.output_file_value, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {config.output_file_value}")


if __name__ == "__main__":
    config = chz.entrypoint(APIEvalConfig)
    asyncio.run(cli_main(config))
