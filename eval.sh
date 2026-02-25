#!/usr/bin/env bash
# Self-play offline evaluation
# Override any parameter by setting env var: LR=1e-4 ./run.sh

# Model configuration
BASE_MODEL=${BASE_MODEL:-'openai/gpt-oss-120b'}
TINKER_CHECKPOINT_URL=${TINKER_CHECKPOINT_URL:-''}
MAX_TOKENS=${MAX_TOKENS:-2048}

# Evaluation parameters
N=${N:-100}
SEED=${SEED:-42}
SPLITS=${SPLITS:-'browsecomp,dsqa,browsecomp_plus'}

# RL-specific parameters
MAX_NUM_CALLS=${MAX_NUM_CALLS:-25}
HANDLING_MODE=${HANDLING_MODE:-'continue'}

# Web search tool settings
VECTOR_SEARCH=${VECTOR_SEARCH:-true}
PORT=${PORT:-8000}
WEB_TOPK=${WEB_TOPK:-10}
WEB_CONTENT_LENGTH=${WEB_CONTENT_LENGTH:-10000}
WEB_SCORING_FUNC=${WEB_SCORING_FUNC:-'rouge'}
WEB_CHUNKING_FUNC=${WEB_CHUNKING_FUNC:-'newline'}
WEB_TIMEOUT=${WEB_TIMEOUT:-300.0}

# Build command with required parameters
CMD="uv run python -m tinker_cookbook.recipes.self_play.offline_eval \
  base_model=$BASE_MODEL \
  max_eval_samples=$N \
  seed=$SEED \
  splits=$SPLITS \
  max_tokens=$MAX_TOKENS \
  max_num_calls=$MAX_NUM_CALLS \
  handling_mode=$HANDLING_MODE \
  web_tool_port=$PORT \
  web_tool_topk=$WEB_TOPK \
  web_tool_content_length=$WEB_CONTENT_LENGTH \
  web_tool_scoring_func=$WEB_SCORING_FUNC \
  web_tool_chunking_func=$WEB_CHUNKING_FUNC \
  web_tool_timeout=$WEB_TIMEOUT \
  vector_search=$VECTOR_SEARCH"

# Add optional parameters if set
[[ -n "$TINKER_CHECKPOINT_URL" ]] && CMD="$CMD tinker_checkpoint_url=$TINKER_CHECKPOINT_URL"

# Execute command
eval $CMD

