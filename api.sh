#!/usr/bin/env bash
# API-based evaluation using litellm
# Override any parameter by setting env var: MODEL=anthropic/claude-sonnet-4-20250514 ./api.sh
#
# Modes:
#   MODE=solver (default) — evaluate on QA datasets (browsecomp, dsqa, etc.)
#   MODE=challenger       — generate questions from corpus data (bcplus, fineweb)

# Mode: "solver" or "challenger"
MODE=${MODE:-'solver'}

# Model configuration
MODEL=${MODEL:-'azure/o3'}
MAX_TOKENS=${MAX_TOKENS:-8192}

# Evaluation parameters
N=${N:-100}
SEED=${SEED:-42}
SPLITS=${SPLITS:-'browsecomp,dsqa,browsecomp_plus'}
CORPUS_SPLITS=${CORPUS_SPLITS:-'bcplus'}
CONCURRENCY=${CONCURRENCY:-16}
MAX_NUM_CALLS=${MAX_NUM_CALLS:-100}
MAX_TRAJECTORY_TOKENS=${MAX_TRAJECTORY_TOKENS:-131_072}

# Web search tool settings
SEARCH_MODE=${SEARCH_MODE:-default}
PORT=${PORT:-8000}
WEB_TOPK=${WEB_TOPK:-5}
WEB_CONTENT_LENGTH=${WEB_CONTENT_LENGTH:-10000}
WEB_SCORING_FUNC=${WEB_SCORING_FUNC:-'rouge'}
WEB_CHUNKING_FUNC=${WEB_CHUNKING_FUNC:-'newline'}
WEB_TIMEOUT=${WEB_TIMEOUT:-300.0}

CMD="uv run python -m tinker_cookbook.recipes.self_play.eval.api_eval \
  mode=$MODE \
  model=$MODEL \
  max_tokens=$MAX_TOKENS \
  max_eval_samples=$N \
  seed=$SEED \
  splits=$SPLITS \
  corpus_splits=$CORPUS_SPLITS \
  concurrency=$CONCURRENCY \
  max_num_calls=$MAX_NUM_CALLS \
  max_trajectory_tokens=$MAX_TRAJECTORY_TOKENS \
  web_tool_port=$PORT \
  web_tool_topk=$WEB_TOPK \
  web_tool_content_length=$WEB_CONTENT_LENGTH \
  web_tool_scoring_func=$WEB_SCORING_FUNC \
  web_tool_chunking_func=$WEB_CHUNKING_FUNC \
  web_tool_timeout=$WEB_TIMEOUT \
  search_mode=$SEARCH_MODE"

eval $CMD
