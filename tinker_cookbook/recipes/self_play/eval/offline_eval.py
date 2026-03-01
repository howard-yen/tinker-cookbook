import asyncio
import json
import logging
import os
import random
from typing import TypedDict

import chz
import tinker
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter

from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.utils import logtree

from tinker_cookbook.recipes.self_play.search_env import SPEnv, load_qa_dataset, QADatum, SOLVER_SYSTEM_PROMPT, CHALLENGER_SYSTEM_PROMPT
from tinker_cookbook.recipes.self_play.search_utils import WebSearchToolConfig, WebSearchTool
from tinker_cookbook.recipes.self_play.utils import SeedDatum, load_seed_dataset
from tinker_cookbook.recipes.self_play.eval.config import OfflineEvalConfig

ROLLOUT_CONCURRENCY = 1024
rollout_semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)


logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationResult(TypedDict):
    question: str
    correct_score: float
    trajectory: object


def _serialize_messages(messages: list) -> list[dict]:
    """Convert renderer messages (which may contain pydantic ToolCall objects) to plain dicts."""
    serialized = []
    for msg in messages:
        if isinstance(msg, dict):
            entry = {}
            for k, v in msg.items():
                if k == "tool_calls" and isinstance(v, list):
                    entry[k] = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in v]
                else:
                    entry[k] = v
            serialized.append(entry)
        elif hasattr(msg, "model_dump"):
            serialized.append(msg.model_dump())
        else:
            serialized.append(str(msg))
    return serialized


async def generate_single_question(
    item: SeedDatum,
    idx: int,
    total: int,
    web_tool: WebSearchTool,
    policy: TinkerTokenCompleter,
    renderer: Renderer,
    config: OfflineEvalConfig,
) -> dict:
    """Run the challenger rollout for a single corpus item to generate a question."""
    tool_schemas = web_tool.get_tool_schemas()
    initial_messages = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=CHALLENGER_SYSTEM_PROMPT,
    )

    env = SPEnv(
        document=item["document"],
        url=item["url"],
        renderer=renderer,
        search_tool=web_tool,
        coordinator=None,
        player_id=0,
        convo_prefix=initial_messages,
        max_num_calls=config.max_num_calls,
        max_tokens=config.max_tokens,
        handling_mode=config.handling_mode,
    )

    async with rollout_semaphore:
        with logtree.scope_details(f"[{idx}/{total}] Challenger: {item['url'][:80]}"):
            logtree.log_text(f"URL: {item['url']}")
            logtree.log_text(f"Document: {item['document'][:300]}...")
            trajectory = await do_single_rollout(policy, env)

            num_calls = 0
            valid_format = 0.0
            if trajectory.transitions:
                final_metrics = trajectory.transitions[-1].metrics
                num_calls = final_metrics.get("challenger_num_calls", 0)
                valid_format = float(env.generated_output is not None)

            output = env.generated_output
            logtree.log_text(f"Generated question: {output.get('question') if output else None}")
            logtree.log_text(f"Generated answer: {output.get('answer') if output else None}")
            logtree.log_text(f"Valid format: {valid_format}, Tool calls: {num_calls}")

    return {
        "idx": idx,
        "url": item["url"],
        "document_snippet": item["document"][:200],
        "question": output.get("question") if output else None,
        "answer": output.get("answer") if output else None,
        "explanation": output.get("explanation") if output else None,
        "num_tool_calls": num_calls,
        "valid_format": valid_format,
        "messages": _serialize_messages(env.past_messages),
    }


async def generate_questions_dataset(
    split: str,
    data: list[SeedDatum],
    policy: TinkerTokenCompleter,
    renderer: Renderer,
    web_tool: WebSearchTool,
    config: OfflineEvalConfig,
) -> dict:
    """Generate questions from all corpus items in a split."""
    log_path = config.log_path_value
    log_path += f"/challenger_{split}_samples{config.max_eval_samples}_tools{config.max_num_calls}"
    os.makedirs(log_path, exist_ok=True)
    logtree_path = os.path.join(log_path, "challenger.html")

    tasks = [generate_single_question(item, i + 1, len(data), web_tool, policy, renderer, config) for i, item in enumerate(data)]

    logger.info(f"Generating questions from {len(tasks)} corpus items for split '{split}'")
    with logtree.init_trace(f"Offline Challenger — {split}", path=logtree_path):
        logtree.log_text(f"Question generation for corpus split '{split}'")
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


async def evaluate_single_item(
    item: QADatum,
    idx: int,
    total: int,
    web_tool: WebSearchTool,
    policy: TinkerTokenCompleter,
    renderer: Renderer,
    config: OfflineEvalConfig,
) -> EvaluationResult:
    tool_schemas = web_tool.get_tool_schemas()
    initial_messages = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=SOLVER_SYSTEM_PROMPT,
    )

    env = SPEnv(
        document="",
        url="",
        renderer=renderer,
        search_tool=web_tool,
        coordinator=None,
        player_id=1,
        convo_prefix=initial_messages,
        max_num_calls=config.max_num_calls,
        max_tokens=config.max_tokens,
        handling_mode=config.handling_mode,
        problem=item,
    )
    async with rollout_semaphore:
        with logtree.scope_details(f"[{idx}/{total}] Rollout: {item['question'][:80]}"):
            trajectory = await do_single_rollout(policy, env)

            # Extract correct metric from the last transition
            correct_score = 0.0
            final_metrics = {}
            if trajectory.transitions:
                final_metrics = trajectory.transitions[-1].metrics
                correct_score = final_metrics.get("solver_correctness_reward", 0.0)

            logtree.log_text(f"Gold answer: {item['answer']}")
            logtree.log_text(f"Correct score: {correct_score}")
            logtree.log_text(f"Tool calls: {final_metrics.get('solver_num_calls', 0)}")

    return {
        "idx": idx,
        "question": item["question"],
        "correct_score": correct_score,
        "trajectory": trajectory,
        "messages": _serialize_messages(env.past_messages),
        **final_metrics,
    }


async def evaluate_one_dataset(
    split: str,
    data: list[QADatum],
    policy: TinkerTokenCompleter,
    renderer: Renderer,
    web_tool: WebSearchTool,
    config: OfflineEvalConfig,
) -> dict:
    # Set up logtree logging
    log_path = config.log_path_value
    log_path += f"/{split}_samples{config.max_eval_samples}_tools{config.max_num_calls}"

    os.makedirs(log_path, exist_ok=True)
    logtree_path = os.path.join(log_path, "eval.html")

    # Run evaluations in parallel using asyncio.gather
    tasks = [evaluate_single_item(item, i + 1, len(data), web_tool, policy, renderer, config) for i, item in enumerate(data)]

    logger.info(f"Evaluating {len(tasks)} items for split '{split}'")
    with logtree.init_trace(f"Offline Evaluation — {split}", path=logtree_path):
        logtree.log_text(
            f"Offline evaluation results for split '{split}'"
        )
        results = await tqdm_asyncio.gather(*tasks, desc=f"Eval {split}")

        # Aggregate results — average all numeric metrics across items
        if not results:
            logger.info(f"Evaluation for '{split}' complete. HTML log saved to {logtree_path}")
            return {"split": split, "total_samples": 0, "accuracy": 0.0}

        skip_keys = {"idx", "question", "trajectory", "messages"}
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

    # Save per-instance trajectories as JSONL (exclude non-serializable trajectory object)
    jsonl_path = os.path.join(log_path, "trajectories.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k != "trajectory"}
            f.write(json.dumps(row, default=str) + "\n")
    logger.info(f"Trajectories saved to {jsonl_path}")

    logger.info(f"Evaluation for '{split}' complete. HTML log saved to {logtree_path}")

    aggregated: dict[str, object] = {"split": split, "total_samples": len(results)}
    for key in numeric_keys:
        values = [r.get(key, 0.0) for r in results]
        aggregated[f"avg_{key}"] = sum(values) / len(values)

    return aggregated


async def cli_main(config: OfflineEvalConfig):
    # Set up shared resources
    service_client = tinker.ServiceClient()
    if config.tinker_checkpoint_url:
        sampling_client = service_client.create_sampling_client(model_path=config.tinker_checkpoint_url)
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.base_model)

    policy = TinkerTokenCompleter(sampling_client, max_tokens=config.max_tokens)

    tokenizer = tokenizer_utils.get_tokenizer(config.base_model)
    renderer_name = model_info.get_recommended_renderer_name(config.base_model)
    renderer = get_renderer(renderer_name, tokenizer)

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

            results = await generate_questions_dataset(split, sampled_data, policy, renderer, web_tool, config)
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
        # Evaluate each split
        all_results = []
        for split in config.splits:
            data = load_qa_dataset(split)

            random.seed(config.seed)
            sampled_data = random.sample(data, min(config.max_eval_samples, len(data)))

            results = await evaluate_one_dataset(split, sampled_data, policy, renderer, web_tool, config)
            all_results.append(results)

        # Print results table (fixed columns only)
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
    config = chz.entrypoint(OfflineEvalConfig)
    asyncio.run(cli_main(config))
