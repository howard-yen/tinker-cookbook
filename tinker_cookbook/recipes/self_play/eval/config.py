import os
from typing import Literal

import chz


@chz.chz
class EvalConfig:
    # Mode
    mode: Literal["solver", "challenger"] = chz.field(default="solver", doc="Run mode: 'solver' for QA evaluation, 'challenger' for question generation")

    # Evaluation parameters
    max_eval_samples: int = chz.field(default=100, doc="Maximum number of samples to evaluate per data source")
    seed: int = chz.field(default=42, doc="Random seed for sampling")
    splits: tuple[str, ...] = chz.field(default=("browsecomp",), doc="Dataset splits to evaluate (e.g. browsecomp, dsqa)")
    corpus_splits: tuple[str, ...] = chz.field(default=("bcplus",), doc="Corpus splits for challenger mode (e.g. bcplus, fineweb)")
    concurrency: int = chz.field(default=32, doc="Max concurrent eval items")
    max_tokens: int = chz.field(default=1024, doc="Maximum tokens per completion")
    max_num_calls: int = 4
    max_trajectory_tokens: int = chz.field(default=128_000, doc="Maximum total tokens (prompt + completion) allowed in the conversation before forcing a final response")
    handling_mode: Literal["raise", "return", "continue"] = "continue"

    # Web tool
    web_tool_port: int = 8000
    web_tool_topk: int = 5
    web_tool_content_length: int = 10000
    web_tool_scoring_func: str = "rouge"
    web_tool_chunking_func: str = "newline"
    web_tool_timeout: float = 300.0
    search_mode: Literal["default", "vector", "single"] = "default"

    # Logging
    log_path: str | None = chz.field(default=None, doc="Directory to save logtree HTML files")

    # Output
    output_file: str | None = chz.field(default=None, doc="Path to save results as JSON")

    def _log_path_suffix(self) -> str:
        return (
            f"/samples{self.max_eval_samples}_tools{self.max_num_calls}"
            f"_topk{self.web_tool_topk}_cl{self.web_tool_content_length}"
            f"_{self.search_mode}_seed{self.seed}"
        )

    @chz.init_property
    def output_file_value(self) -> str:
        if self.output_file is None:
            if self.mode == "challenger":
                return self.log_path_value + f"/results_{'_'.join(self.corpus_splits)}.json"
            else:
                return self.log_path_value + f"/results_{'_'.join(self.splits)}.json"
        return self.output_file


@chz.chz
class APIEvalConfig(EvalConfig):
    model: str = chz.field(default="openai/gpt-4o", doc="LiteLLM model string (e.g. 'openai/gpt-4o', 'anthropic/claude-sonnet-4-20250514')")

    @chz.init_property
    def log_path_value(self) -> str:
        if self.log_path is None:
            return f"/tmp/tinker/self_play/eval/{self.model.replace('/', '-')}/{self.mode}" + self._log_path_suffix()
        return self.log_path


@chz.chz
class OfflineEvalConfig(EvalConfig):
    base_model: str = chz.field(default="Qwen/Qwen3-4B-Instruct-2507", doc="Base model to use")
    tinker_checkpoint_url: str | None = chz.field(default=None, doc="Tinker checkpoint URL (optional)")

    @chz.init_property
    def log_path_value(self) -> str:
        if self.log_path is None:
            log_path = f"/tmp/tinker/self_play/eval/{os.path.basename(self.base_model)}/{self.mode}"
            if self.tinker_checkpoint_url:
                log_path += f"/{self.tinker_checkpoint_url.replace('tinker://', '').replace('/', '-')}"
            log_path += self._log_path_suffix()
            return log_path
        return self.log_path
