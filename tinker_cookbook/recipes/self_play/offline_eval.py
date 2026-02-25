import asyncio
import logging
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Literal, TypedDict

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter

from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.utils import logtree

from tinker_cookbook.recipes.self_play.search_env import SPEnv, load_qa_dataset, QADatum, SOLVER_SYSTEM_PROMPT
from tinker_cookbook.recipes.self_play.search_utils import WebSearchToolConfig, WebSearchTool

ROLLOUT_CONCURRENCY = 1024
rollout_semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)


logging.basicConfig(level=os.getenv("LOG_LEVEL", logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@chz.chz
class CLIConfig:
    # Evaluation parameters
    max_eval_samples: int = chz.field(
        default=100, doc="Maximum number of samples to evaluate per data source"
    )
    seed: int = chz.field(default=42, doc="Random seed for sampling")
    splits: tuple[str, ...] = chz.field(default=("browsecomp",), doc="Dataset splits to evaluate (e.g. browsecomp, dsqa)")

    # Model parameters
    base_model: str = chz.field(default="Qwen/Qwen3-4B-Instruct-2507", doc="Base model to use")
    tinker_checkpoint_url: str | None = chz.field(default=None, doc="Tinker checkpoint URL (optional)")
    max_tokens: int = chz.field(default=1024, doc="Maximum number of tokens to generate")

    # Environment parameters
    web_tool_port: int = 8000
    web_tool_topk: int = 5
    web_tool_content_length: int = 10000
    web_tool_scoring_func: str = "rouge"
    web_tool_chunking_func: str = "newline"
    web_tool_timeout: float = 300.0
    vector_search: bool = False

    handling_mode: Literal["raise", "return", "continue"] = "continue"

    max_num_calls: int = 4

    # Logging
    log_path: str | None = chz.field(default=None, doc="Directory to save logtree HTML files")

    # Output
    output_file: str | None = chz.field(default=None, doc="Path to save results as JSON")

    @chz.init_property
    def log_path_value(self) -> str:
        if self.log_path is None:
            log_path = f"/tmp/tinker/self_play/eval/{os.path.basename(self.base_model)}"
            if self.tinker_checkpoint_url:
                log_path += f"/{self.tinker_checkpoint_url.replace('tinker://', '').replace('/', '-')}"
            return log_path
        return self.log_path

    @chz.init_property
    def output_file_value(self) -> str:
        if self.output_file is None:
            output_file = self.log_path_value + "/results.json"
            return output_file
        return self.output_file

class EvaluationResult(TypedDict):
    question: str
    correct_score: float
    trajectory: object


async def evaluate_single_item(
    item: QADatum,
    web_tool: WebSearchTool,
    policy: TinkerTokenCompleter,
    renderer: Renderer,
    config: CLIConfig,
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
        with logtree.scope_details(f"Rollout: {item['question'][:80]}"):
            trajectory = await do_single_rollout(policy, env)

    # Extract correct metric from the last transition
    correct_score = 0.0
    if trajectory.transitions:
        final_metrics = trajectory.transitions[-1].metrics
        correct_score = final_metrics.get("correctness_reward", 0.0)

    return {"question": item["question"], "correct_score": correct_score, "trajectory": trajectory, **final_metrics}


async def evaluate_one_dataset(
    split: str,
    data: list[QADatum],
    policy: TinkerTokenCompleter,
    renderer: Renderer,
    web_tool: WebSearchTool,
    config: CLIConfig,
) -> dict:
    # Set up logtree logging
    log_path = config.log_path_value
    log_path += f"/{split}_samples{config.max_eval_samples}_tools{config.max_num_calls}"

    os.makedirs(log_path, exist_ok=True)
    logtree_path = os.path.join(log_path, "eval.html")

    # Run evaluations in parallel using asyncio.gather
    tasks = [evaluate_single_item(item, web_tool, policy, renderer, config) for item in data]

    logger.info(f"Evaluating {len(tasks)} items for split '{split}'")
    with logtree.init_trace(f"Offline Evaluation — {split}", path=logtree_path):
        logtree.log_text(
            "This HTML log was generated by logtree during offline evaluation. "
            "It shows rollouts and rewards for evaluation items."
        )
        results = await asyncio.gather(*tasks)

    logger.info(f"Evaluation for '{split}' complete. HTML log saved to {logtree_path}")
    # print(f"Evaluation for '{split}' complete. HTML log saved to {logtree_path}")

    # Aggregate results — average all numeric metrics across items
    if not results:
        return {"split": split, "total_samples": 0, "accuracy": 0.0}

    # Collect all numeric fields from the first result (skip non-numeric like question, trajectory)
    skip_keys = {"question", "trajectory"}
    numeric_keys = [
        k for k, v in results[0].items()
        if k not in skip_keys and isinstance(v, (int, float))
    ]

    aggregated = {"split": split, "total_samples": len(results)}
    for key in numeric_keys:
        values = [r.get(key, 0.0) for r in results]
        aggregated[f"avg_{key}"] = sum(values) / len(values)

    return aggregated


async def cli_main(config: CLIConfig):
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
        vector_search=config.vector_search,
    )
    web_tool = WebSearchTool(web_tool_config)

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
    with open(config.output_file_value, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {config.output_file_value}")


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(config))
