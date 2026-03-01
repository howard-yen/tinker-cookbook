import asyncio
import logging
import os
import random
import re
import json
import hashlib
from functools import partial
from typing import Literal, Sequence, Callable, List
from dataclasses import dataclass

import chz
import numpy as np
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
from tinker_cookbook.renderers.base import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.recipes.self_play.search_utils import WebSearchTool, WebSearchToolConfig
from tinker_cookbook.recipes.self_play.utils import grade_answer, SeedDatum, load_seed_dataset, QADatum, load_qa_dataset, api_call_with_retry


# Suppress LiteLLM INFO messages
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Use force=True so this takes effect even if another library already called basicConfig
logging.basicConfig(level=int(os.getenv("LOG_LEVEL", logging.INFO)), format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

_CONNECTION_SEMAPHORE = asyncio.Semaphore(128)

FORMAT_REWARD = 0.5


CHALLENGER_FALLBACK_SYSTEM_PROMPT = """
You are an expert teacher who writes challenging questions for a student to answer.
You are given a document as the starting point, write a challenging, clear, motivated, and solvable problem.

Document: {DOCUMENT}

Write your final output in the json format:
```json
{{
    "question": "The question statement of the problem",
    "answer": "The answer to the problem",
    "explanation": "The explanation of the problem and what document sources were used to answer the question",
}}
```
""".strip()


CHALLENGER_SYSTEM_PROMPT = """
You are an expert teaching who writes challenging questions for a student to solve.
You are given a document and a url as the starting point, and you will have access to tools to help you collect more informaiton to write a challenging question.
The student will not have access to the given document, and will need to use the tools to collect information to answer the question. Thus, do not include any explicit references the document you are given.
Importantly, your question and answer MUST BE grounded in the documents you are given or collected from the tools. You should explicitly state the document sources you used to answer the question in your explanation.

Use the tools to collect information to write a challenging, clear, motivated, and solvable problem. Use the given document and url as a starting point. 
For example, you may use the tools to collect more information about the topic of the document.
Then, after you have collected enough information, write a problem with both the question statement and the answer. 
The question should require reasoning over many documents to answer, so you should use the tools at least once to collect more documents.
Both the question statement and the answer should be clear, unambiguous, and concise, the answer should be easily verifiable and does not exceed more than 20 words. 
The explanation should explicitly cite the document sources you used to answer the question.

Write your final output in the json format:
```json
{
    "question": "The question statement of the problem",
    "answer": "The answer to the problem",
    "explanation": "The explanation of the problem and what document sources were used to answer the question",
}
```
""".strip()


SOLVER_SYSTEM_PROMPT = """
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
""".strip()




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
        # challenger's move is a dict with keys "question", "answer", "explanation"
        # special case: False means the challenger failed to make a move, the game is done, and the solver should not attempt
        # if the solver makes a move, solve the problem and conclude the game
        # the solver's move is a tuple of (correctness bool, num tool calls)
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
        problem: dict | None = None,
        opponent_policy: TinkerMessageCompleter | None = None, # fixed policy when not doing self-play
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 1024,
        max_num_calls: int = 4,
        handling_mode: Literal["raise", "return", "continue"] = "raise",
        difficulty_reward_mode: Literal["variance", "linear", "none"] = "variance",
        tool_reward_mode: Literal["max", "mean", "none"] = "max",
        do_fallback_challenger: bool = True,
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
        if self.coordinator is not None:
            assert 0 <= player_id <= coordinator.num_solvers, f"Invalid player id: {player_id}, expect at most {coordinator.num_solvers} solvers"
            self.cid = coordinator.id
        elif player_id == 0:
            # Standalone challenger mode: no coordinator, no problem needed
            self.cid = hashlib.sha256(document[:100].encode()).hexdigest()[:8]
        else:
            assert problem is not None, "Problem is required when coordinator is None"
            self.cid = hashlib.sha256(problem["question"].encode()).hexdigest()[:8]

        self.document: str = document
        self.url: str = url
        self.problem: dict | None = problem
        self.generated_output: dict | None = None
        self.search_tool: WebSearchTool = search_tool
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.past_messages: list[renderers.Message] = convo_prefix.copy() if convo_prefix else []
        self.current_num_calls: int = 0
        self.max_num_calls: int = max_num_calls
        self.max_tokens: int = max_tokens
        self.handling_mode: Literal["raise", "return", "continue"] = handling_mode
        self.difficulty_reward_mode: Literal["variance", "linear", "none"] = difficulty_reward_mode
        self.tool_reward_mode: Literal["max", "mean", "none"] = tool_reward_mode
        self.do_fallback_challenger: bool = do_fallback_challenger


    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()


    async def wait_for_turn(self) -> None:
        if self.coordinator is not None and not self.coordinator.game_done:
            if self.self_play:
                role = "Challenger" if self.player_id == 0 else f"Solver {self.player_id}"
                logger.debug(f"{self.cid} {role} waiting for turn (phase: {self.coordinator.current_phase})")
                await self.coordinator.wait_across_env(self.player_id)
                logger.debug(f"{self.cid} {role} turn acquired")
            else:
                raise ValueError("Not implemented: opponent policy not supported when not doing self-play")
                # self.opponent_policy(self.past_messages)
        else:
            logger.debug(f"{self.cid} Coordinator game is done, no need to wait for turn")


    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.player_id == 0:
            logger.debug(f"{self.cid} {self.player_id} making problem")
            logtree.log_text(f"{self.cid} Challenger {self.player_id} starting to make problem")
            convo = self.convo_prefix + [
                {"role": "user", "content": f"Document: {self.document}\nUrl: {self.url}"},
            ]
        else:
            await self.wait_for_turn()

            logger.debug(f"{self.cid} Solver {self.player_id} waiting for turn")
            if self.coordinator is not None:
                problem = self.coordinator.problem
                self.problem = problem
            else:
                problem = self.problem
            
            if problem is False:
                # return special value that indicate this rollout should be dropped
                return None, self.stop_condition

            logtree.log_text(f"{self.cid} Solver {self.player_id} waiting for turn completed, starting to solve {problem}")
            logger.debug(f"{self.cid} Solver {self.player_id} waiting for turn completed, starting to solve {problem}")
            if problem is None:
                # can replace with a oracle sampler later
                raise ValueError("No problem found in coordinator or problem")

            convo = self.convo_prefix + [
                {"role": "user", "content": problem["question"]},
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
        args = json.loads(tool_call.function.arguments)
        query_list, error = self.search_tool.validate_and_extract_search_args(args)
        if error:
            return [renderers.Message(role="tool", name="search", content=error)]
        async with _CONNECTION_SEMAPHORE:
            search_results = await self.search_tool.batch_search(query_list)
            logtree.log_text(f"Search results: {search_results}")
            return [renderers.Message(role="tool", name="search", content=search_results)]

    async def call_browse_tool(self, tool_call: renderers.ToolCall) -> list[renderers.Message]:
        args = json.loads(tool_call.function.arguments)
        visit_args, error = self.search_tool.validate_and_extract_visit_args(args)
        if error:
            return [renderers.Message(role="tool", name="visit", content=error)]
        url_list, query_list = visit_args
        async with _CONNECTION_SEMAPHORE:
            browse_results = await self.search_tool.batch_browse(url_list, query_list)
            logtree.log_text(f"Browse results: {browse_results}")
            return [renderers.Message(role="tool", name="visit", content=browse_results)]


    def get_failure_results(self) -> StepResult:
        return StepResult(
            reward=-1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={("challenger" if self.player_id == 0 else "solver") + "_num_calls": self.current_num_calls},
        )
    
    def get_invalid_results(self) -> StepResult:
        return StepResult(
            reward=-0.2,
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.past_messages),
            next_stop_condition=self.stop_condition,
        )

    def handle_error(self, message: str | None = None, override: Literal["raise", "return", "continue"] | None = None) -> StepResult:
        logtree.log_text(f"Handling error: {message}, override: {override}")
        mode = self.handling_mode if override is None else override
        if mode == "raise":
            raise ValueError(message)
        elif mode == "return":
            return self.get_failure_results()
        elif mode == "continue":
            assert message is not None, "Message is required when handling mode is continue"
            self.past_messages.append(renderers.Message(role="user", content=message))
            return self.get_invalid_results()

    async def call_tool(self, message: renderers.Message) -> StepResult:
        # TODO: maybe define constants FAILURE_REWARD, SUCCESS_REWARD, etc.
        tool_calls = message["tool_calls"]
        
        if not tool_calls:
            # three ways to handle: return failure_results, raise ValueError, or continue with next step with observation message
            return self.handle_error("No tool calls found in the previous message.")
        
        tool_call = tool_calls[0]
        if tool_call.function.name not in ("search", "visit"):
            return self.handle_error(f"Invalid tool name: {tool_call.function.name}\nMake sure to use only search or visit tools.")

        tool_label = "Search" if tool_call.function.name == "search" else "Browse"

        if self.current_num_calls >= self.max_num_calls:
            tool_return_message = [renderers.Message(role="tool", name=tool_call.function.name, content=f"Error calling {tool_call.function.name} tool: Max number of calls reached, please complete the task without using any more tools.")]
            self.past_messages.extend(tool_return_message)
        else:
            self.current_num_calls += 1
            try:
                with logtree.scope_details(f"{tool_label} tool call {self.current_num_calls}"):
                    logtree.log_text(f"{tool_label} tool arguments: {tool_call.function.arguments}")
                    if tool_call.function.name == "search":
                        tool_return_message = await self.call_search_tool(tool_call)
                    else:
                        tool_return_message = await self.call_browse_tool(tool_call)
                self.past_messages.extend(tool_return_message)
            except Exception as e:
                return self.handle_error(f"Error calling {tool_call.function.name} tool: {repr(e)}")

        next_observation = self.renderer.build_generation_prompt(self.past_messages)
        if next_observation.length + self.max_tokens > self.max_trajectory_tokens:
            logger.error(f"{self.cid} {self.player_id} Next observation is too long: {next_observation.length} + {self.max_tokens} > {self.max_trajectory_tokens}\nMake sure to keep the observation within the maximum trajectory length.")
            return self.handle_error(f"{self.cid} {self.player_id} Next observation is too long: {next_observation.length} + {self.max_tokens} > {self.max_trajectory_tokens}\nMake sure to keep the observation within the maximum trajectory length.", override="return")

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(self.past_messages),
            next_stop_condition=self.stop_condition,
        )

    async def _fallback_challenger(self) -> None:
        logtree.log_text(f"{self.cid} Challenger {self.player_id} falling back to using oracle challenger")
        response = api_call_with_retry(
            model="azure/gpt-4.1", 
            messages=[{"role": "user", "content": CHALLENGER_FALLBACK_SYSTEM_PROMPT.format(DOCUMENT=self.document)}],
            response_format=None,
        )

        response = response['choices'][0]['message']['content']
        output = self._extract_json(response)
        if output is None:
            raise ValueError(f"Invalid output: {response}")
        await self.coordinator.make_move(self.player_id, output)

    async def challenger_final_step(self, content: str, correct_format: bool) -> StepResult:
        # correct format = json format is valid
        format_reward = 0.0
        difficulty_reward = 0.0
        tool_reward = 0.0
        correctness = None
        tool_use = None
        
        if correct_format:
            output = self._extract_json(content)
            if all(x in output for x in ["question", "answer", "explanation"]):
                format_reward = FORMAT_REWARD
                self.generated_output = output
                logger.debug(f"{self.cid} Challenger {self.player_id} making move with output: {output}")

                if self.coordinator is not None:
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
                    elif self.tool_reward_mode == "min":
                        if len(correct_tool_use) > 0:
                            tool_reward = float(np.min(correct_tool_use) / self.max_num_calls)
                    elif self.tool_reward_mode == "none":
                        tool_reward = 0.0
                    else:
                        raise ValueError(f"Invalid tool reward mode: {self.tool_reward_mode}")

            else:
                # fall back to using oracle challenger, don't need to wait for the solvers
                if self.coordinator is not None:
                    if self.do_fallback_challenger:
                        await self._fallback_challenger()
                    else:
                        # just return failure to coordinator
                        await self.coordinator.make_move(self.player_id, False)
            
            total_reward = format_reward + difficulty_reward + tool_reward
            # log the response
            logtree.log_text(f"==========Challenger Final Output==========")
            logtree.log_text(f"Initial document: {self.document[:500]}...")
            with logtree.scope_details("Challenger Initial Document"):
                logtree.log_text(self.document)
            logtree.log_text(f"Response: {content}")
            logtree.log_text(f"Format reward: {format_reward}")
            logtree.log_text(f"Difficulty reward: {difficulty_reward}")
            logtree.log_text(f"Tool reward: {tool_reward}; num calls: {self.current_num_calls}")
            logtree.log_text(f"Total reward: {total_reward}")
            logtree.log_text(f"==========Challenger End of Output==========")
            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "challenger_num_calls": self.current_num_calls,
                    "challenger_format_reward": format_reward,
                    "challenger_difficulty_reward": difficulty_reward,
                    "challenger_tool_reward": tool_reward,
                }
            )
            
        else:
            # the solver could be waiting forever... should probably just raise the error here
            # raise ValueError(f"Invalid output: {content}\nMake sure to output the final problem in the json format.")
            return self.handle_error(f"Invalid output: {content}\nMake sure to output the final problem in the json format.")


    async def solver_final_step(self, content: str, correct_format: bool) -> StepResult:
        format_reward = 0.0
        correctness_reward = 0.0
        tool_reward = 0.0

        if correct_format:
            output = self._extract_json(content)
            extras = {}
            if "answer" in output and "explanation" in output:
                format_reward = FORMAT_REWARD
                correct, extras = grade_answer(self.problem, output["answer"])
                correctness_reward = correct
                if correct >= 0.5:
                    # TODO: this should be configurable
                    tool_reward = (self.max_num_calls - self.current_num_calls) / self.max_num_calls
                # need to give the tool usage too
                if self.coordinator is not None:
                    await self.coordinator.make_move(self.player_id, (correct, self.current_num_calls))

            elif self.coordinator is not None:
                # still need to make a move to signal the solver is done
                await self.coordinator.make_move(self.player_id, (False, self.current_num_calls))
            
            total_reward = format_reward + correctness_reward + tool_reward
            
            # log the response
            logtree.log_text(f"==========Solver {self.player_id} Final Output==========")
            logtree.log_text(f"Problem: {self.problem}")
            logtree.log_text(f"Response: {content}")
            logtree.log_text(f"Format reward: {format_reward}")
            logtree.log_text(f"Correctness reward: {correctness_reward}")
            logtree.log_text(f"Tool reward: {tool_reward}; num calls: {self.current_num_calls}")
            logtree.log_text(f"Total reward: {total_reward}")
            logtree.log_text(f"==========Solver {self.player_id} End of Output==========")
            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "solver_num_calls": self.current_num_calls,
                    "solver_format_reward": format_reward,
                    "solver_correctness_reward": correctness_reward,
                    "solver_tool_reward": tool_reward,
                    **extras,
                }
            )

        else:
            return self.handle_error(f"Invalid output: {content}\nMake sure to output the final answer and explanation in the json format.")


    async def _ensure_coordinator_notified(self) -> None:
        """Safety net: ensure coordinator is notified when an episode ends.

        Without this, a solver/challenger that exits early (e.g. due to format
        errors or trajectory length limits) can leave the other side waiting
        forever on the coordinator condition, causing a deadlock.
        """
        if self.coordinator is None or self.coordinator.game_done:
            return
        if self.player_id > 0 and self.coordinator.solver_results[self.player_id - 1] is None:
            logger.warning(f"{self.cid} Solver {self.player_id} episode ending without make_move, signaling failure to coordinator")
            await self.coordinator.make_move(self.player_id, (False, self.current_num_calls))
        elif self.player_id == 0 and self.coordinator.current_phase == "challenger":
            if self.do_fallback_challenger:
                logger.warning(f"{self.cid} Challenger episode ending without make_move, using fallback")
                try:
                    await self._fallback_challenger()
                except Exception as e:
                    logger.error(f"{self.cid} Fallback challenger failed: {e}, marking game as done")
                    async with self.coordinator.condition:
                        self.coordinator.done = True
                        self.coordinator.condition.notify_all()
            else:
                logger.warning(f"{self.cid} Challenger episode ending without make_move, not using fallback, just returning failure to coordinator")
                await self.coordinator.make_move(self.player_id, False)


    async def step(self, action: Action) -> StepResult:
        # if this is the challenger's environment, we go ahead and take a step
        # if this is the solver's environment, we wait for the challenger to make a move first
        # but that is already handled in the initial observation function

        message, parse_success = self.renderer.parse_response(action)
        self.past_messages.append(message)

        try:
            if "tool_calls" in message:
                result = await self.call_tool(message)
            else:
                # challenger and solve share different logic here
                # the message is a list with different types (thinking vs. text), we only care about the text message
                content = get_text_content(message)
                correct_format = float(parse_success) and float(self.check_format(content))
                logger.debug(f"{self.cid} {self.player_id} final step")

                if self.player_id == 0:
                    result = await self.challenger_final_step(content, correct_format)
                else:
                    result = await self.solver_final_step(content, correct_format)
        except Exception:
            await self._ensure_coordinator_notified()
            raise

        if result.episode_done:
            await self._ensure_coordinator_notified()

        return result
                

    @staticmethod
    def standard_fewshot_prefix(renderer: renderers.Renderer, search_tool: WebSearchTool, player_id: int) -> list[renderers.Message]:
        return renderer.create_conversation_prefix_with_tools(
            tools=search_tool.get_tool_schemas(),
            system_prompt=CHALLENGER_SYSTEM_PROMPT if player_id == 0 else SOLVER_SYSTEM_PROMPT,
        )


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
            # assert isinstance(self.coordinator, SPCoordinator), "Solver environments expect a single coordinator"
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
        split: Literal["fineweb", "bcplus", "browsecomp", "browsecomp_plus", "dsqa"] = "fineweb",
        subset_size: int | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 1024,
        max_num_calls: int = 4,
        self_play: bool = True,
        handling_mode: Literal["raise", "return", "continue"] = "raise",
        difficulty_reward_mode: Literal["variance", "linear", "none"] = "variance",
        tool_reward_mode: Literal["max", "mean", "none"] = "max",
        do_fallback_challenger: bool = True,
    ):
        self.batch_size: int = batch_size
        self.group_size: int = group_size
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.max_tokens: int = max_tokens
        self.max_num_calls: int = max_num_calls
        self.renderer: renderers.Renderer = renderer
        self.search_tool: WebSearchTool = search_tool
        self.seed: int = seed
        self.split: Literal["fineweb", "bcplus", "browsecomp", "browsecomp_plus", "dsqa"] = split
        self.do_fallback_challenger: bool = do_fallback_challenger

        if split in ["fineweb", "bcplus"]:
            self.ds: list[SeedDatum] = load_seed_dataset(split, num_samples=subset_size)
        elif split in ["browsecomp", "browsecomp_plus", "dsqa", "browsecomp_plus_train", "browsecomp_plus_test"]:
            self.ds: list[QADatum] = load_qa_dataset(split)
        else:
            raise ValueError(f"Unknown dataset: {split}")

        self.self_play: bool = self_play
        self.handling_mode: Literal["raise", "return", "continue"] = handling_mode
        self.difficulty_reward_mode: Literal["variance", "linear", "none"] = difficulty_reward_mode
        self.tool_reward_mode: Literal["max", "mean", "none"] = tool_reward_mode
        # shuffle with seed
        self.rng = random.Random(self.seed)
        self.rng.shuffle(self.ds)
        # Limit dataset size if subset_size is specified
        self.subset_size: int | None = subset_size
        if subset_size is not None:
            self.ds = self.ds[:subset_size]

    def _get_batch_rows(self, index: int) -> list:
        """Get rows for a batch, wrapping around the dataset for multi-epoch support."""
        n = len(self.ds)
        start = (index * self.batch_size) % n
        if start + self.batch_size > n:
            data = self.ds[start:]
            # for the next epoch, we need to shuffle the data
            self.rng.shuffle(self.ds)
            return data
        else:
            return self.ds[start:start + self.batch_size]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        # HY: this might be the only thing that we really need to change.
        # If self-play, then we also have to make the solver environments
        # Each challenger environment will have group_size solver environments
        # Thus, each row will result in group_size challenger environments and group_size*group_size solver environments

        rows = self._get_batch_rows(index)

        if self.self_play and self.split in ["fineweb", "bcplus"]:
            # each challenger and its set of group_size solver environments will share the same coordinator
            batches = []
            for idx, row in enumerate(rows):
                coordinator = [SPCoordinator(num_solvers=self.group_size, document=row["document"], coordinator_id=i) for i in range(self.group_size)]
                challenger_env = self._make_challenger_env_group_builder(row, self.group_size, coordinator=coordinator)
                batches.append(challenger_env)
                solver_envs = [self._make_solver_env_group_builder(row, self.group_size, coordinator=coordinator[i]) for i in range(self.group_size)]
                batches.extend(solver_envs)
            return batches

        elif self.split in ["browsecomp", "browsecomp_plus", "browsecomp_plus_train", "browsecomp_plus_test", "dsqa"]:
            # just testing the challenger
            # need some special logic for the solver groups so it doesn't wait for the challenger to make a move
            return [self._make_solver_env_group_builder(row=None, group_size=self.group_size, coordinator=None, problem=row) for row in rows]

        else:
            # just training the challenger
            return [
                self._make_env_group_builder(row, self.group_size)
                for row in rows
            ]

    def __len__(self) -> int:
        return self.subset_size // self.batch_size
    
    def _make_challenger_env_group_builder(self, row: SeedDatum, group_size: int, coordinator: SPCoordinator | None = None) -> SPGroupBuilder:
        return SPGroupBuilder(
            phase="challenger",
            coordinator=coordinator,
            env_thunk=partial(
                SPEnv,
                row["document"],
                row["url"],
                self.search_tool,
                self.renderer,
                convo_prefix=SPEnv.standard_fewshot_prefix(self.renderer, self.search_tool, 0),
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_tokens=self.max_tokens,
                max_num_calls=self.max_num_calls,
                self_play=self.self_play,
                handling_mode=self.handling_mode,
                difficulty_reward_mode=self.difficulty_reward_mode,
                tool_reward_mode=self.tool_reward_mode,
                do_fallback_challenger=self.do_fallback_challenger,
            ),
            num_envs=group_size,
        )

    def _make_solver_env_group_builder(self, row: SeedDatum | QADatum, group_size: int, coordinator: SPCoordinator | None = None, problem: dict | None = None) -> SPGroupBuilder:
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
                convo_prefix=SPEnv.standard_fewshot_prefix(self.renderer, self.search_tool, 1),
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_tokens=self.max_tokens,
                max_num_calls=self.max_num_calls,
                self_play=self.self_play,
                handling_mode=self.handling_mode,
                difficulty_reward_mode=self.difficulty_reward_mode,
                tool_reward_mode=self.tool_reward_mode,
            ),
            num_envs=group_size,
        ) if problem is None else SPGroupBuilder(
            phase="solver",
            coordinator=coordinator,
            env_thunk=partial(
                SPEnv,
                None,
                None,
                self.search_tool,
                self.renderer,
                problem=problem,
                convo_prefix=SPEnv.standard_fewshot_prefix(self.renderer, self.search_tool, 1),
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_tokens=self.max_tokens,
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
    eval_group_size: int
    eval_max_num_calls: int = 25
    handling_mode: Literal["raise", "return", "continue"] = "raise"
    difficulty_reward_mode: Literal["variance", "linear", "none"] = "variance"
    tool_reward_mode: Literal["max", "mean", "none"] = "max"

    train_split: Literal["fineweb", "bcplus"] = "fineweb"
    eval_split: Literal["browsecomp", "browsecomp_plus", "dsqa"] = "browsecomp"
    n_batches: int | None = None  # If set, limits the number of training batches

    model_name_for_tokenizer: str
    renderer_name: str
    search_tool_config: WebSearchToolConfig
    do_fallback_challenger: bool = True

    max_eval_size: int = 1024
    max_trajectory_tokens: int = 32 * 1024
    max_tokens: int = 1024
    max_num_calls: int = 4
    seed: int = 0

    async def __call__(self) -> tuple[SPDataset, None]:
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
            split=self.train_split,
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_tokens=self.max_tokens,
            max_num_calls=self.max_num_calls,
            subset_size=subset_size,
            handling_mode=self.handling_mode,
            difficulty_reward_mode=self.difficulty_reward_mode,
            tool_reward_mode=self.tool_reward_mode,
            do_fallback_challenger=self.do_fallback_challenger,
        )

        # now we make the eval dataset, but we are only going evaluate the solver
        # instead of using questions generated by the challenger, we are just gonna load them from another dataset with qa pairs already
        # this also means that we need to change the some of the logic in the SPEnv
        eval_dataset = SPDataset(
            batch_size=self.batch_size,
            group_size=self.eval_group_size,
            renderer=renderer,
            search_tool=search_tool,
            split=self.eval_split,
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_tokens=self.max_tokens,
            max_num_calls=self.eval_max_num_calls,
            subset_size=self.max_eval_size,
            handling_mode=self.handling_mode,
            difficulty_reward_mode=self.difficulty_reward_mode,
            tool_reward_mode=self.tool_reward_mode,
        )
        
        return (train_dataset, eval_dataset)
