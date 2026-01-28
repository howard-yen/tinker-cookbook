import asyncio
import logging
import os
import random
import re
import string
import json
import hashlib
from functools import partial, reduce
from pathlib import Path
from typing import Literal, Sequence, TypedDict, cast, Callable, List
from dataclasses import dataclass

import chz
import numpy as np
import pandas as pd
import litellm
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder, ProblemEnv
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


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress LiteLLM INFO messages
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

_CONNECTION_SEMAPHORE = asyncio.Semaphore(128)

FORMAT_REWARD = 0.5


GPT_OSS_TOOL_DESC = """
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

# Tool calling. Execute the tool by wrapping calls in <tool_call>...</tool_call>
CHALLENGER_SYSTEM_PROMPT = """
You are an expert teacher who writes challenging questions for a student to answer.
You are given a document and a url as the starting point, and you will have access to tools to help you collect more information to write a challenging problem.
Use the tools to write a challenging, clear, motivated, and solvable problem.

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


CHALLENGER_FALLBACK_SYSTEM_PROMPT = """
You are an expert teacher who writes challenging questions for a student to answer.
You are given a document as the starting point, write a challenging, clear, motivated, and solvable problem.

Document: {DOCUMENT}

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
GPT_OSS_CHALLENGER_SYSTEM_PROMPT = """
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
""".strip() + GPT_OSS_TOOL_DESC


GPT_OSS_SOLVER_SYSTEM_PROMPT = """
You are a helpful assistant that can search the web. You are encourage to use the search tool to best answer the user's question. Use the search tool to collect useful information.
When using the search tool, you should think carefully about the problem. Decompose and rewrite the search query if necessary. After using the search tool, you should reason about the results and summarize the relevant information to answer the problem. If the search results are not relevant, you are encouraged to refine your search query and search again. Continue to use the tools until you have collected all the information you need, this may take many iterations.
The search tool will return a list of documents, and you should visit the documents relevant to the problem.
After you have collected all the information you need, you should reason about the results and answer the problem.

Write your final output in the json format:
```json
{
    "answer": "The answer to the problem",
    "explanation": "The explanation of the problem and what document sources were used to answer the problem",
}
```
""".strip() + GPT_OSS_TOOL_DESC


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


class SPCoordinator:
    """Coordinator for the SPEnv. Allows the challenger and the solver to communicate with each other (challenger passes the problem to the solver)."""

    def __init__(self, num_solvers: int = 1, document: str = "", coordinator_id: int = 0):
        assert num_solvers > 0, f"Number of solvers must be greater than 0: {num_solvers}"
        self.condition = asyncio.Condition()
        self.problem = None
        # "challenger": challenger phase, "solver": solver phase
        self._current_phase: Literal["challenger", "solver"] = "challenger"
        # number of solvers, wait for all solvers to finish before finishing
        self.num_solvers: int = num_solvers
        self._solver_results: list[bool | None] = [None] * num_solvers
        self._solver_tools: list[int | None] = [None] * num_solvers
        self.done: bool = False # only finishes when there is an error or solver finishes the problem
        self.doc_id: str = hashlib.sha256(document.encode()).hexdigest()[:8]
        self.coordinator_id: int = coordinator_id
        self.status = ["init"] * (num_solvers + 1)

    @property
    def game_done(self) -> bool:
        return self.done

    @property
    def current_phase(self) -> Literal["challenger", "solver"]:
        return self._current_phase

    @property
    def solver_results(self) -> list[bool | None]:
        return self._solver_results

    @property
    def solver_tools(self) -> list[int | None]:
        return self._solver_tools
    
    @property
    def id(self) -> str:
        return f"{self.doc_id}-{self.coordinator_id}"

    def check_phase(self, player: int) -> bool:
        valid = (self.current_phase == "challenger" and player == 0) or (self.current_phase == "solver" and player > 0)
        return valid
    
    async def wait_across_env(self, player: int) -> None:
        """
        Player id corresponds to solver (0) and challenger (1 to num_solvers).
        This method allows the player to wait until it's their phase.
        """
        assert 0 <= player <= self.num_solvers, f"Invalid player id: {player}"
        self.status[player] = "wait env"
        logger.debug(f"{self.id}: {self.status}")
        async with self.condition:
            self.status[player] = "wait env cond"
            logger.debug(f"{self.id}: {self.status}")
            await self.condition.wait_for(lambda: self.game_done or self.check_phase(player))
            self.status[player] = "wait env done"
            logger.debug(f"{self.id}: {self.status}")
    
    async def make_move(self, player: int, move) -> None:
        # if the challenger makes a move, save the problem to solver results and notify the solver
        # if the solver makes a move, solve the problem and conclude the game
        self.status[player] = "make move"
        logger.debug(f"{self.id}: {self.status}")
        async with self.condition:
            self.status[player] = "make move cond"
            logger.debug(f"{self.id}: {self.status}")
            current_phase = self.current_phase
            if not self.game_done and not self.check_phase(player):
                raise ValueError(f"Not {player}'s turn (current phase: {current_phase}), the results are {self.solver_results}")

            if current_phase == "challenger":
                self.problem = move
                self._current_phase = "solver"
            elif current_phase == "solver":
                self.solver_results[player-1] = move[0]
                self.solver_tools[player-1] = move[1]
                if all(result is not None for result in self.solver_results):
                    self.done = True
                    self._current_phase = "challenger"
            else:
                raise ValueError(f"Invalid phase: {current_phase}")

            self.condition.notify_all()
            self.status[player] = "make move done"
            logger.debug(f"{self.id}: {self.status}")


class SPEnv(Env):
    def __init__(
        self,
        document: str,
        url: str,
        search_tool: WebSearchTool,
        renderer: renderers.Renderer,
        player_id: int,
        coordinator: SPCoordinator,
        self_play: bool = True,
        opponent_policy: TinkerMessageCompleter | None = None, # fixed policy when not doing self-play
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        max_num_calls: int = 4,
        handling_mode: Literal["raise", "return", "continue"] = "raise",
        difficulty_reward_mode: Literal["variance", "linear", "none"] = "variance",
        tool_reward_mode: Literal["max", "mean", "none"] = "max",
    ):
        self.renderer: renderers.Renderer = renderer
        self.convo_prefix: list[renderers.Message] | None = convo_prefix
        self.player_id: int = player_id
        self.coordinator: SPCoordinator = coordinator
        self.self_play: bool = self_play
        self.opponent_policy: TinkerMessageCompleter | None = opponent_policy
        assert self.self_play == (self.opponent_policy is None), (
            "If self_play is True, opponent_policy must be None"
        )
        assert 0 <= player_id <= coordinator.num_solvers, f"Invalid player id: {player_id}, expect at most {coordinator.num_solvers} solvers"

        self.document: str = document
        self.url: str = url
        self.search_tool: WebSearchTool = search_tool
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.past_messages: list[renderers.Message] = convo_prefix.copy() if convo_prefix else []
        self.current_num_calls: int = 0
        self.max_num_calls: int = max_num_calls
        self.handling_mode: Literal["raise", "return", "continue"] = handling_mode
        self.difficulty_reward_mode: Literal["variance", "linear", "none"] = difficulty_reward_mode
        self.tool_reward_mode: Literal["max", "mean", "none"] = tool_reward_mode

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()


    async def wait_for_turn(self) -> None:
        if not self.coordinator.game_done:
            if self.self_play:
                role = "Challenger" if self.player_id == 0 else f"Solver {self.player_id}"
                logger.debug(f"{self.coordinator.id} {role} waiting for turn (phase: {self.coordinator.current_phase})")
                await self.coordinator.wait_across_env(self.player_id)
                logger.debug(f"{self.coordinator.id} {role} turn acquired")
            else:
                raise ValueError("Not implemented: opponent policy not supported when not doing self-play")
                # self.opponent_policy(self.past_messages)
        else:
            logger.debug(f"{self.coordinator.id} Coordinator game is done, no need to wait for turn")


    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.player_id == 0:
            # logger.info(f"{self.coordinator.id} Challenger starting to make problem")
            logger.debug(f"{self.coordinator.id} {self.player_id} making problem")
            logtree.log_text(f"{self.coordinator.id} Challenger {self.player_id} starting to make problem")
            convo = self.convo_prefix + [
                {"role": "user", "content": f"Document: {self.document}\nUrl: {self.url}"},
            ]
        else:
            logger.debug(f"{self.coordinator.id} {self.player_id} init wait")
            await self.wait_for_turn()
            logger.debug(f"{self.coordinator.id} {self.player_id} init wait done")
            problem_summary = str(self.coordinator.problem)[:100] + "..." if self.coordinator.problem else "None"
            # logger.info(f"{self.coordinator.id} Solver {self.player_id} starting to solve problem")
            logtree.log_text(f"{self.coordinator.id} Solver {self.player_id} waiting for turn completed, starting to solve {self.coordinator.problem}")
            if self.coordinator.problem is None:
                # can replace with a oracle sampler later
                raise ValueError("No problem found in coordinator")
            convo = self.convo_prefix + [
                {"role": "user", "content": self.coordinator.problem["question"]},
            ]
        
        self.past_messages = convo.copy()
        return self.renderer.build_generation_prompt(convo), self.stop_condition


    async def opponent_step(self) -> None:
        raise NotImplementedError("Not implemented: opponent step not supported when not doing self-play")


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


    def get_failure_results(self) -> StepResult:
        return StepResult(
            reward=-1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
        )
    
    def get_invalid_results(self) -> StepResult:
        return StepResult(
            reward=-0.2,
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.past_messages),
            next_stop_condition=self.stop_condition,
        )

    def handle_error(self, message: str | None = None) -> StepResult:
        if self.handling_mode == "raise":
            raise ValueError(message)
        elif self.handling_mode == "return":
            return self.get_failure_results()
        elif self.handling_mode == "continue":
            assert message is not None, "Message is required when handling mode is continue"
            self.past_messages.append(renderers.Message(role="user", content=message))
            return self.get_invalid_results()

    async def call_tool(self, message: renderers.Message) -> StepResult:
        # TODO: maybe define constants FAILURE_REWARD, SUCCESS_REWARD, etc.
        tool_calls = message["tool_calls"]
        
        if not tool_calls:
            # three ways to handle: return failure_results, raise ValueError, or continue with next step with observation message
            return self.handle_error("No tool calls found in the previous message.")
        
        logtree.log_text(f"Tool calls invoked.")
        tool_call = tool_calls[0]
        if tool_call.function.name == "search":
            # check for tool call validity
            logtree.log_text(f"Search tool called: {tool_call.function.arguments}")
            self.current_num_calls += 1
            if self.current_num_calls > self.max_num_calls:
                tool_return_message = [renderers.Message(role="tool", name="search", content="Error calling search tool: Max number of calls reached, please complete the task without using any more tools.")]
                self.past_messages.extend(tool_return_message)
            
            elif "query_list" not in tool_call.function.arguments:
                return self.handle_error(f"Query list not found in tool calls: {tool_call.function.arguments}\nMake sure to include a query_list argument when using the search tool.")
            
            else:
                try:
                    tool_return_message = await self.call_search_tool(tool_call)
                    self.past_messages.extend(tool_return_message)
                except Exception as e:
                    return self.handle_error(f"Error calling search tool: {repr(e)}")

            next_observation = self.renderer.build_generation_prompt(self.past_messages)
            if next_observation.length > self.max_trajectory_tokens:
                return self.handle_error(f"Next observation is too long: {next_observation.length}\nMake sure to keep the observation within the maximum trajectory length.")

        elif tool_call.function.name == "visit":
            logtree.log_text(f"Browse tool called: {tool_call.function.arguments}")
            self.current_num_calls += 1
            if self.current_num_calls > self.max_num_calls:
                tool_return_message = [renderers.Message(role="tool", name="visit", content="Error calling browse tool: Max number of calls reached, please complete the task without using any more tools.")]
                self.past_messages.extend(tool_return_message)
            
            elif "id_list" not in tool_call.function.arguments:
                return self.handle_error(f"Id list not found in tool calls: {tool_call.function.arguments}\nMake sure to include an id_list argument when using the browse tool.")
            
            else:
                try:
                    tool_return_message = await self.call_browse_tool(tool_call)
                    self.past_messages.extend(tool_return_message)
                except Exception as e:
                    return self.handle_error(f"Error calling browse tool: {repr(e)}")

            next_observation = self.renderer.build_generation_prompt(self.past_messages)
            if next_observation.length > self.max_trajectory_tokens:
                return self.handle_error(f"Next observation is too long: {next_observation.length}\nMake sure to keep the observation within the maximum trajectory length.")
       
        else:
            return self.handle_error(f"Invalid tool name: {tool_call.function.name}\nMake sure to use only search or visit tools.")

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.past_messages),
            next_stop_condition=self.stop_condition,
        )


    async def challenger_final_step(self, message: renderers.Message, correct_format: bool) -> StepResult:
        # correct format = json format is valid
        format_reward = 0.0
        difficulty_reward = 0.0
        tool_reward = 0.0
        correctness = None
        tool_use = None
        
        if correct_format:
            output = self._extract_json(message["content"])
            if [x in output for x in ["question", "answer", "explanation"]]:
                format_reward = FORMAT_REWARD
                await self.coordinator.make_move(self.player_id, output)
                await self.wait_for_turn()

                # only calculate difficulty and tool reward if the solver uses problem generated by the real challenger
                correctness = self.coordinator.solver_results
                logtree.log_text(f"Solver correctness: {correctness}")
                if self.difficulty_reward_mode == "variance":
                    difficulty_reward = np.exp(-(np.var(correctness) - 0.25) ** 2 / 0.02)
                    difficulty_reward = float(difficulty_reward)
                elif self.difficulty_reward_mode == "linear":
                    difficulty_reward = 1.1 - np.mean(correctness)
                elif self.difficulty_reward_mode == "none":
                    difficulty_reward = 0.0
                else:
                    raise ValueError(f"Invalid difficulty reward mode: {self.difficulty_reward_mode}")
                
                tool_use = self.coordinator.solver_tools
                logtree.log_text(f"Solver tool use: {tool_use}")
                correct_tool_use = [t for (c, t) in zip(correctness, tool_use) if c]
                # note that the max num tool call may not be the same between challenger and solver, TODO!
                if self.tool_reward_mode == "max":
                    # take the maximum number of tool calls among correct trajectories
                    if len(correct_tool_use) > 0:
                        tool_reward = float(np.max(correct_tool_use) / self.max_num_calls)
                elif self.tool_reward_mode == "mean":
                    # take the mean number of tool calls among correct trajectories
                    if len(correct_tool_use) > 0:
                        tool_reward = float(np.mean(correct_tool_use) / self.max_num_calls)
                elif self.tool_reward_mode == "none":
                    tool_reward = 0.0
                else:
                    raise ValueError(f"Invalid tool reward mode: {self.tool_reward_mode}")

            else:
                # fall back to using oracle challenger, don't need to wait for the solvers
                response = litellm.completion(
                    model="openai/gpt-4.1-2025-04-14", 
                    messages=[{
                        "role": "user", "content": CHALLENGER_FALLBACK_SYSTEM_PROMPT.format(DOCUMENT=self.document)
                    }]
                )
                response = response['choices'][0]['message']['content']
                output = self._extract_json(response)
                if output is None:
                    raise ValueError(f"Invalid output: {response}")
                await self.coordinator.make_move(self.player_id, output)
            
            total_reward = format_reward + difficulty_reward + tool_reward
            # log the response
            logtree.log_text(f"==========Challenger Final Output==========")
            logtree.log_text(f"Initial document: {self.document}")
            logtree.log_text(f"Response: {message['content']}")
            logtree.log_text(f"Format reward: {format_reward}")
            logtree.log_text(f"Difficulty reward: {difficulty_reward}")
            logtree.log_text(f"Tool reward: {tool_reward}")
            logtree.log_text(f"Total reward: {total_reward}")
            logtree.log_text(f"==========Challenger End of Output==========")
            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "num_calls": self.current_num_calls,
                    "format_reward": format_reward,
                    "difficulty_reward": difficulty_reward,
                    "tool_reward": tool_reward,
                }
            )
            
        else:
            # the solver could be waiting forever... should probably just raise the error here
            raise ValueError(f"Invalid output: {message['content']}\nMake sure to output the final problem in the json format.")
            return self.handle_error(f"Invalid output: {message['content']}\nMake sure to output the final problem in the json format.")


    async def solver_final_step(self, message: renderers.Message, correct_format: bool) -> StepResult:
        format_reward = 0.0
        correctness_reward = 0.0
        tool_reward = 0.0

        if correct_format:
            output = self._extract_json(message["content"])
            if "answer" in output and "explanation" in output:
                format_reward = FORMAT_REWARD
                grading_response = litellm.completion(
                    model="openai/gpt-4.1-2025-04-14", 
                    messages=[{
                        "role": "user", "content": GRADER_TEMPLATE.format(
                            question=self.coordinator.problem["question"], 
                            response=output["answer"], 
                            correct_answer=self.coordinator.problem["answer"]
                        )
                    }]
                )
                grading_response = grading_response['choices'][0]['message']['content']
                correct_match = re.search(r"correct: (yes|no)", grading_response)
                correct = correct_match.group(1) == "yes" if correct_match else False
                if correct:
                    correctness_reward = 1.0
                    tool_reward = (self.max_num_calls - self.current_num_calls) / self.max_num_calls
                # need to give the tool usage too
                await self.coordinator.make_move(self.player_id, (correct, self.current_num_calls))

            else:
                # still need to make a move to signal the solver is done
                await self.coordinator.make_move(self.player_id, (False, self.current_num_calls))
            
            total_reward = format_reward + correctness_reward + tool_reward
            
            # log the response
            logtree.log_text(f"==========Solver {self.player_id} Final Output==========")
            logtree.log_text(f"Problem: {self.coordinator.problem}")
            logtree.log_text(f"Response: {message['content']}")
            logtree.log_text(f"Format reward: {format_reward}")
            logtree.log_text(f"Correctness reward: {correctness_reward}")
            logtree.log_text(f"Tool reward: {tool_reward}")
            logtree.log_text(f"Total reward: {total_reward}")
            logtree.log_text(f"==========Solver {self.player_id} End of Output==========")
            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format_reward": format_reward,
                    "correctness_reward": correctness_reward,
                    "tool_reward": tool_reward,
                }
            )

        else:
            return self.handle_error(f"Invalid output: {message['content']}\nMake sure to output the final answer and explanation in the json format.")


    async def step(self, action: Action) -> StepResult:
        # if this is the challenger's environment, we go ahead and take a step
        # if this is the solver's environment, we wait for the challenger to make a move first
        # but that is already handled in the initial observation function
        
        message, parse_success = self.renderer.parse_response(action)
        self.past_messages.append(message)
        
        if "tool_calls" in message:
            return await self.call_tool(message)

        else:
            # challenger and solve share different logic here
            correct_format = float(parse_success) and float(self.check_format(message["content"]))
            logger.debug(f"{self.coordinator.id} {self.player_id} final step")

            if self.player_id == 0:            
                return await self.challenger_final_step(message, correct_format)
            else:
                return await self.solver_final_step(message, correct_format)
                

    @staticmethod
    def standard_fewshot_prefix(player_id: int) -> list[renderers.Message]:
        if player_id == 0:
            return [{"role": "developer", "content": GPT_OSS_CHALLENGER_SYSTEM_PROMPT},]
        else:
            return [{"role": "developer", "content": GPT_OSS_SOLVER_SYSTEM_PROMPT},]
            # {
            #     "role": "system",
            #     "content": CHALLENGER_SYSTEM_PROMPT,
            # },

            # for gpt oss, the tool definitions are in the developer message



class FinewebDatum(TypedDict):
    document: str
    url: str


def load_fineweb_dataset(split: Literal["train", "test"]) -> list[FinewebDatum]:
    with open(f"tinker_cookbook/recipes/tool_use/self_play/{split}.jsonl", "r") as f:
        fw = [json.loads(line.strip()) for line in f]

    return [{"document": item["text"], "url": item["url"]} for item in fw]


@dataclass(frozen=True)
class SPGroupBuilder(ProblemGroupBuilder):
    env_thunk: Callable[[], ProblemEnv]
    num_envs: int
    dataset_name: str = "self-play"
    coordinator: List[SPCoordinator] | SPCoordinator | None = None
    phase: Literal["challenger", "solver"] = "challenger"

    async def make_envs(self) -> Sequence[Env]:
        if self.phase == "challenger":
            assert isinstance(self.coordinator, list), "Challenger environments expect a list of different coordinators"
            return [self.env_thunk(player_id=0, coordinator=self.coordinator[i]) for i in range(self.num_envs)]
        else:
            assert isinstance(self.coordinator, SPCoordinator), "Solver environments expect a single coordinator"
            return [self.env_thunk(player_id=i+1, coordinator=self.coordinator) for i in range(self.num_envs)]


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
        self_play: bool = True,
        handling_mode: Literal["raise", "return", "continue"] = "raise",
        difficulty_reward_mode: Literal["variance", "linear", "none"] = "variance",
        tool_reward_mode: Literal["max", "mean", "none"] = "max",
    ):
        self.batch_size: int = batch_size
        self.group_size: int = group_size
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.max_num_calls: int = max_num_calls
        self.renderer: renderers.Renderer = renderer
        # self.convo_prefix: list[renderers.Message] | None = convo_prefix
        self.search_tool: WebSearchTool = search_tool
        self.seed: int = seed
        self.split: Literal["train", "test"] = split
        self.ds: list[FinewebDatum] = load_fineweb_dataset(split)
        self.self_play: bool = self_play
        self.handling_mode: Literal["raise", "return", "continue"] = handling_mode
        self.difficulty_reward_mode: Literal["variance", "linear", "none"] = difficulty_reward_mode
        self.tool_reward_mode: Literal["max", "mean", "none"] = tool_reward_mode
        # shuffle with seed
        rng = random.Random(self.seed)
        rng.shuffle(self.ds)
        # Limit dataset size if subset_size is specified
        if subset_size is not None:
            self.ds = self.ds[:subset_size]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        # HY: this might be the only thing that we really need to change.
        # If self-play, then we also have to make the solver environments
        # Each challenger environment will have group_size solver environments
        # Thus, each row will result in group_size challenger environments and group_size*group_size solver environments

        if self.self_play:
            # each challenger and its set of group_size solver environments will share the same coordinator
            batches = []
            for idx, row in enumerate(self.ds[index * self.batch_size : (index + 1) * self.batch_size]):
                coordinator = [SPCoordinator(num_solvers=self.group_size, document=row["document"], coordinator_id=i) for i in range(self.group_size)]
                challenger_env = self._make_challenger_env_group_builder(row, self.group_size, coordinator=coordinator)
                batches.append(challenger_env)
                solver_envs = [self._make_solver_env_group_builder(row, self.group_size, coordinator=coordinator[i]) for i in range(self.group_size)]
                batches.extend(solver_envs)
            return batches
        
        else:        
            # just training the challenger
            return [
                self._make_env_group_builder(row, self.group_size)
                for row in self.ds[index * self.batch_size : (index + 1) * self.batch_size]
            ]

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size
    
    def _make_challenger_env_group_builder(self, row: FinewebDatum, group_size: int, coordinator: SPCoordinator | None = None) -> SPGroupBuilder:
        return SPGroupBuilder(
            phase="challenger",
            coordinator=coordinator,
            env_thunk=partial(
                SPEnv,
                row["document"],
                row["url"],
                self.search_tool,
                self.renderer,
                convo_prefix=SPEnv.standard_fewshot_prefix(0),
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_num_calls=self.max_num_calls,
                self_play=self.self_play,
                handling_mode=self.handling_mode,
                difficulty_reward_mode=self.difficulty_reward_mode,
                tool_reward_mode=self.tool_reward_mode,
            ),
            num_envs=group_size,
        )

    def _make_solver_env_group_builder(self, row: FinewebDatum, group_size: int, coordinator: SPCoordinator | None = None) -> SPGroupBuilder:
        # solver gets a special group builder because it also needs to pass in the player id
        return SPGroupBuilder(
            phase="solver",
            coordinator=coordinator,
            env_thunk=partial(
                SPEnv,
                row["document"],
                row["url"],
                self.search_tool,
                self.renderer,
                convo_prefix=SPEnv.standard_fewshot_prefix(1),
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_num_calls=self.max_num_calls,
                self_play=self.self_play,
                handling_mode=self.handling_mode,
                difficulty_reward_mode=self.difficulty_reward_mode,
                tool_reward_mode=self.tool_reward_mode,
            ),
            num_envs=group_size,
        )


@chz.chz
class SPDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    handling_mode: Literal["raise", "return", "continue"] = "raise"
    difficulty_reward_mode: Literal["variance", "linear", "none"] = "variance"
    tool_reward_mode: Literal["max", "mean", "none"] = "max"
    model_name_for_tokenizer: str
    renderer_name: str
    search_tool_config: WebSearchToolConfig
    # convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_eval_size: int = 1024
    max_trajectory_tokens: int = 32 * 1024
    max_num_calls: int = 4
    n_batches: int | None = None  # If set, limits the number of training batches

    async def __call__(self) -> tuple[SPDataset, None]:
        # if self.convo_prefix == "standard":
        #     convo_prefix = SPEnv.standard_fewshot_prefix()
        # else:
        #     convo_prefix = self.convo_prefix
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
            # convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_num_calls=self.max_num_calls,
            subset_size=subset_size,
            handling_mode=self.handling_mode,
            difficulty_reward_mode=self.difficulty_reward_mode,
            tool_reward_mode=self.tool_reward_mode,
        )
        return (train_dataset, None)
