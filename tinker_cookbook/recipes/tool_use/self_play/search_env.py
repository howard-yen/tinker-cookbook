import asyncio
import logging
import os
import random
import re
import string
import json
from functools import partial, reduce
from pathlib import Path
from typing import Literal, Sequence, TypedDict, cast

import chz
import numpy as np
import pandas as pd
import litellm
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.recipes.tool_use.self_play.search_utils import WebSearchTool, WebSearchToolConfig


logger = logging.getLogger(__name__)

_CONNECTION_SEMAPHORE = asyncio.Semaphore(128)

# Tool calling. Execute the tool by wrapping calls in <tool_call>...</tool_call>
CHALLENGER_SYSTEM_PROMPT = """
You are a expert teacher who writes challenging questions for a student to answer.
You are given a document and a url as a starting point, and you will have access to tools to help you to collect more information to write a challenging problem.
Use the tools effectively to write a challenging, clear, motivated, and solvable problem.

When using tools, make sure to write your arguments in the correct format, you should output a json string with the following format:
{
    "name": "the name of the tool you are calling",
    "args": "the arguments of the tool in a json dictionary",
}

The search tool you are given has the following schema:
```
{
    "name": "search",
    "title": "Search the web",
    "description": "Search the web for relevant information with the queries. This tool will return a list of urls with a snippet of the content in the url for each query.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of fully-formed semantic queries. This tool will return search results for each query.",
            }
        },
        "required": ["query_list"],
    },
    "outputSchema": {
        "type": "string",
        "description": "The search results in JSON format",
    },
}
```

The browse tool you are given has the following schema:
```
{
    "name": "browse",
    "title": "Browse the web",
    "description": "Browse the urls. This tool will return a snippet of the content in each url. Optionally, you can search for a specific query in each url, and the tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "url_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of urls to browse. The tool will return a snippet of the content in each url.",
            },
            "query_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of queries to search for in each url. The tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query. If given an empty query, the tool will return the beginning of the page.",
            },
        },
        "required": ["url_list"],
    },
    "outputSchema": {
        "type": "string",
        "description": "The browse results in JSON format",
    },
}
```

Use the tools to collect information to write a challenging, clear, motivated, and solvable problem. Use the given document and url as a starting point. 
For example, you may use the tools to collect more information about the topic of the document.
Then, after you have collected enough information, write a problem with both the question statement and the answer. 
The question should require reasoning over many documents to answer, so you should use the tools at least once to collect more documents.
Both the question statement and the answer should be clear, unambiguous, and concise, the answer should be easily verifiable and does not exceed more than 20 words. 

Write your final output in the json format:
```json
{
    "question": "The question statement of the problem",
    "answer": "The answer to the problem",
    "explanation": "The explanation of the problem and what document sources were used to answer the question",
}
```
""".strip()


# for gpt oss
GPT_OSS_SYSTEM_PROMPT = """
You are an expert teaching who writes challenging questions for a student to solve.
You are given a document and a url as the starting point, and you will have access to tools to help you collect more informaiton to write a challenging question.

Use the tools to collect information to write a challenging, clear, motivated, and solvable problem. Use the given document and url as a starting point. 
For example, you may use the tools to collect more information about the topic of the document.
Then, after you have collected enough information, write a problem with both the question statement and the answer. 
The question should require reasoning over many documents to answer, so you should use the tools at least once to collect more documents.
Both the question statement and the answer should be clear, unambiguous, and concise, the answer should be easily verifiable and does not exceed more than 20 words. 

Write your final output in the json format:
```json
{
    "question": "The question statement of the problem",
    "answer": "The answer to the problem",
    "explanation": "The explanation of the problem and what document sources were used to answer the question",
}
```

# Tools

## functions

namespace functions {

// Search the web for relevant information with the queries. This tool will return a list of ids with a snippet from each document for each query.
type search = (_: {
// A list of queries to search the web for.
query_list: string[],
}) => any;

// Visit the documents by their ids. This tool will return a snippet of the content in each document. Optionally, you can search for a specific query in each document, and the tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query.",
type visit = (_: {
// A list of document ids to browse. The tool will return the content in each document.
id_list: string[],
// A list of queries to search for in each document. The tool will perform fuzzy matching to find the part of the document that contains the highest textual similarity to the query.
query_list?: string[] | null,
}) => any;

} // namespace functions

""".strip()


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





# TODO: should add some example problems 
DEMONSTRATION=""""""


def normalize_answer(s: str) -> str:
    """Normalize answer by lowercasing, removing punctuation, articles, and fixing whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    # Apply transformations in order using reduce
    transformations = [lower, remove_punc, remove_articles, white_space_fix]
    return reduce(lambda text, func: func(text), transformations, s)


class SPEnv(Env):
    def __init__(
        self,
        document: str,
        url: str,
        search_tool: WebSearchTool,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        timeout: float = 1.0,
        max_num_calls: int = 4,
    ):
        self.renderer: renderers.Renderer = renderer
        self.convo_prefix: list[renderers.Message] | None = convo_prefix
        self.document: str = document
        self.url: str = url
        self.timeout: float = timeout
        self.search_tool: WebSearchTool = search_tool
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.past_messages: list[renderers.Message] = convo_prefix.copy() if convo_prefix else []
        self.current_num_calls: int = 0
        self.max_num_calls: int = max_num_calls

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.convo_prefix + [
            {"role": "user", "content": f"Document: {self.document}\nUrl: {self.url}"},
        ]
        self.past_messages = convo.copy()
        return self.renderer.build_generation_prompt(convo), self.stop_condition


    def _extract_json(self, sample_str: str) -> dict | None:
        matches = re.findall(r"\{.*\}", sample_str, re.DOTALL)
        if len(matches) > 0:
            try:
                return json.loads(matches[-1])
            except json.JSONDecodeError:
                return None
        return None


    def check_format(self, sample_str: str) -> bool:
        return self._extract_json(sample_str) is not None


    async def call_search_tool(self, tool_call: renderers.ToolCall) -> list[renderers.Message]:
        query_list = json.loads(tool_call.function.arguments)["query_list"]
        async with _CONNECTION_SEMAPHORE:
            search_results = await self.search_tool.batch_search(query_list)
            return [renderers.Message(role="tool", name="search", content=search_results)]

    
    async def call_browse_tool(self, tool_call: renderers.ToolCall) -> list[renderers.Message]:
        args = json.loads(tool_call.function.arguments)
        id_list = args["id_list"]
        query_list = args.get("query_list", None)
        async with _CONNECTION_SEMAPHORE:
            browse_results = await self.search_tool.batch_browse(id_list, query_list)
            return [renderers.Message(role="tool", name="visit", content=browse_results)]


    def solve(self, question: str, answer: str) -> bool:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        response = litellm.completion(model="openai/o3", messages=messages, n=8)
        responses = [r["message"]["content"] for r in response['choices']]

        correctness = []
        for response in responses:
            grading_response = litellm.completion(model="openai/gpt-4.1-2025-04-14", messages=[{"role": "user", "content": GRADER_TEMPLATE.format(question=question, response=response, correct_answer=answer)}])
            grading_response = grading_response['choices'][0]['message']['content']
            correct_match = re.search(r"correct: (yes|no)", grading_response)
            correctness_bool = correct_match.group(1) == "yes" if correct_match else False
            correctness.append(correctness_bool)
        return correctness

            
    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        self.past_messages.append(message)
        
        if "tool_calls" in message:
            # log the tool calls
            tool_calls = message["tool_calls"]
            failure_result = StepResult(
                reward=-1.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
            )
            if not tool_calls:
                logtree.log_text("Empty tool calls found.")
                raise ValueError("Empty tool calls found.")
                return failure_result
            
            logtree.log_text(f"Tool calls invoked.")
            if tool_calls[0].function.name == "search":
                logtree.log_text(f"Search tool called: {tool_calls[0].function.arguments}")
                self.current_num_calls += 1
                if self.current_num_calls > self.max_num_calls:
                    tool_return_message = [renderers.Message(role="tool", name="search", content="Error calling search tool: Max number of calls reached, please complete the task without using any more tools.")]
                    self.past_messages.extend(tool_return_message)

                elif "query_list" not in tool_calls[0].function.arguments:
                    logtree.log_text(f"Query list not found in tool calls: {tool_calls[0].function.arguments}")
                    raise ValueError(f"Query list not found in tool calls: {tool_calls[0].function.arguments}")
                    return failure_result

                else:
                    try:
                        tool_return_message = await self.call_search_tool(tool_calls[0])
                        self.past_messages.extend(tool_return_message)
                    except Exception as e:
                        # logger.error(f"Error calling search tool: {repr(e)}")
                        raise e
                        logtree.log_text(f"Error calling search tool: {repr(e)}")
                        return failure_result
                
                next_observation = self.renderer.build_generation_prompt(self.past_messages)
                if next_observation.length > self.max_trajectory_tokens:
                    logtree.log_text(f"Next observation is too long: {next_observation.length}")
                    raise ValueError(f"Next observation is too long: {next_observation.length}")
                    return failure_result
            
            elif tool_calls[0].function.name == "visit":
                logtree.log_text(f"Browse tool called: {tool_calls[0].function.arguments}")
                self.current_num_calls += 1
                if self.current_num_calls > self.max_num_calls:
                    tool_return_message = [renderers.Message(role="tool", name="visit", content="Error calling browse tool: Max number of calls reached, please complete the task without using any more tools.")]
                    self.past_messages.extend(tool_return_message)

                elif "id_list" not in tool_calls[0].function.arguments:
                    logtree.log_text(f"Id list not found in tool calls: {tool_calls[0].function.arguments}")
                    raise ValueError(f"Id list not found in tool calls: {tool_calls[0].function.arguments}")
                    return failure_result

                else:
                    try:
                        tool_return_message = await self.call_browse_tool(tool_calls[0])
                        self.past_messages.extend(tool_return_message)
                    except Exception as e:
                        # logger.error(f"Error calling browse tool: {repr(e)}")
                        raise e
                        logtree.log_text(f"Error calling browse tool: {repr(e)}")
                        return failure_result
                
                next_observation = self.renderer.build_generation_prompt(self.past_messages)
                if next_observation.length > self.max_trajectory_tokens:
                    logtree.log_text(f"Next observation is too long: {next_observation.length}")
                    raise ValueError(f"Next observation is too long: {next_observation.length}")
                    return failure_result
            
            else:
                logtree.log_text(f"Invalid tool name: {tool_calls[0].function.name}")
                # raise ValueError(f"Invalid tool name: {tool_calls[0].function.name}, {message}")
                return failure_result
                
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=self.renderer.build_generation_prompt(self.past_messages),
                next_stop_condition=self.stop_condition,
            )

        else:
            # currently, just check the format, we will add some quality metrics later
            correct_format = float(parse_success) and float(self.check_format(message["content"]))
            correctness_reward = 0.0
            penalty = 0.0
            if correct_format:
                output = self._extract_json(message["content"])
                if "question" not in output:
                    penalty -= 0.2
                if "answer" not in output:
                    penalty -= 0.2
                elif len(output["answer"].split()) > 20:
                        penalty -= 0.1
                if "explanation" not in output:
                    penalty -= 0.1
                # add length penalty

                if "question" in output and "answer" in output:
                    correctness = self.solve(output["question"], output["answer"])
                    # from SPICE
                    correctness_reward = np.exp(-(np.var(correctness) - 0.25) ** 2 / 0.02)
                    correctness_reward = float(correctness_reward)
            
            if correct_format and self.current_num_calls < 1:
                penalty -= 0.5
                
            # TODO: implement the variance reward later, can use like 8 runs of gpt-5 or something
            total_reward = correct_format + penalty + correctness_reward
            # log the response
            logtree.log_text(f"==========Final Output==========")
            logtree.log_text(f"Initial document: {self.document}")
            logtree.log_text(f"Response: {message['content']}")
            logtree.log_text(f"Format valid: {correct_format}")
            logtree.log_text(f"Correctness reward: {correctness_reward}")
            logtree.log_text(f"Total reward: {total_reward}")
            logtree.log_text(f"==========End of Output==========")
            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format": correct_format,
                    "num_calls": self.current_num_calls,
                    "penalty": penalty,
                    "correctness_reward": correctness_reward,
                }
            )


    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            # {
            #     "role": "system",
            #     "content": CHALLENGER_SYSTEM_PROMPT,
            # },

            # for gpt oss, the tool definitions are in the developer message
            {
                "role": "developer",
                "content": GPT_OSS_SYSTEM_PROMPT,
            }
        ]



class FinewebDatum(TypedDict):
    document: str
    url: str


def load_fineweb_dataset(split: Literal["train", "test"]) -> list[FinewebDatum]:
    with open(f"tinker_cookbook/recipes/tool_use/self_play/{split}.jsonl", "r") as f:
        fw = [json.loads(line.strip()) for line in f]

    return [{"document": item["text"], "url": item["url"]} for item in fw]


class SPDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        search_tool: WebSearchTool,
        # optional args
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        split: Literal["train", "test"] = "train",
        subset_size: int | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        max_num_calls: int = 4,
    ):
        self.batch_size: int = batch_size
        self.group_size: int = group_size
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.max_num_calls: int = max_num_calls
        self.renderer: renderers.Renderer = renderer
        self.convo_prefix: list[renderers.Message] | None = convo_prefix
        self.search_tool: WebSearchTool = search_tool
        self.seed: int = seed
        self.split: Literal["train", "test"] = split
        self.ds: list[FinewebDatum] = load_fineweb_dataset(split)
        # shuffle with seed
        rng = random.Random(self.seed)
        rng.shuffle(self.ds)
        # Limit dataset size if subset_size is specified
        if subset_size is not None:
            self.ds = self.ds[:subset_size]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            self._make_env_group_builder(row, self.group_size)
            for row in self.ds[index * self.batch_size : (index + 1) * self.batch_size]
        ]

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size
    
    def _make_env_group_builder(self, row: FinewebDatum, group_size: int) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                SPEnv,
                row["document"],
                row["url"],
                self.search_tool,
                self.renderer,
                convo_prefix=self.convo_prefix,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_num_calls=self.max_num_calls,
            ),
            num_envs=group_size,
        )


@chz.chz
class SPDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    search_tool_config: WebSearchToolConfig
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_eval_size: int = 1024
    max_trajectory_tokens: int = 32 * 1024
    max_num_calls: int = 4
    n_batches: int | None = None  # If set, limits the number of training batches

    async def __call__(self) -> tuple[SPDataset, None]:
        if self.convo_prefix == "standard":
            convo_prefix = SPEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        search_tool = WebSearchTool(self.search_tool_config)

        # Compute subset_size from n_batches if specified
        subset_size = None
        if self.n_batches is not None:
            subset_size = self.n_batches * self.batch_size

        train_dataset = SPDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            search_tool=search_tool,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_num_calls=self.max_num_calls,
            subset_size=subset_size,
        )
        return (train_dataset, None)
