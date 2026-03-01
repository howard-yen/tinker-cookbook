#!/usr/bin/env bash
# Self-play training
# Override any parameter by setting env var: LR=1e-4 ./run.sh

# Model configuration
MODEL=${MODEL:-'openai/gpt-oss-120b'}
RANK=${RANK:-16}
RENDERER=${RENDERER:-''}  # Auto-detect if empty

# Training hyperparameters
LOSS_FN=${LOSS_FN:-'ppo'}
LR=${LR:-1e-5}
LR_SCHEDULE=${LR_SCHEDULE:-'cosine'}  # constant, linear, or cosine
WARMUP_STEPS=${WARMUP_STEPS:-1}
BZ=${BZ:-2}
TOKENS=${TOKENS:-4096} # max generation tokens
SEED=${SEED:-42}
EVAL_EVERY=${EVAL_EVERY:-0}
N_BATCHES=${N_BATCHES:-'5'}  # Unlimited if empty
EVAL_N_BATCHES=${EVAL_N_BATCHES:-'50'}  # Unlimited if empty

# RL-specific parameters
GS=${GS:-4}
EGS=${EGS:-4}
NS=${NS:-10}
ENS=${ENS:-25}
TRAJ_TOKENS=${TRAJ_TOKENS:-32768} # max trajectory tokens

TRAIN_SPLIT=${TRAIN_SPLIT:-'bcplus'}
EVAL_SPLIT=${EVAL_SPLIT:-'browsecomp_plus'}

# Self-play behavior
SELF_PLAY=${SELF_PLAY:-true}
HANDLING_MODE=${HANDLING_MODE:-'continue'}
# linear, variance
DIFFICULTY_REWARD_MODE=${DIFFICULTY_REWARD_MODE:-'variance'}
TOOL_REWARD_MODE=${TOOL_REWARD_MODE:-'min'}
DO_FALLBACK_CHALLENGER=${DO_FALLBACK_CHALLENGER:-true}

# Streaming configuration
STREAM_MINIBATCH=${STREAM_MINIBATCH:-false}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-4}

# Web search tool settings
SEARCH_MODE=${SEARCH_MODE:-default}
PORT=${PORT:-8000}
WEB_TOPK=${WEB_TOPK:-10}
WEB_CONTENT_LENGTH=${WEB_CONTENT_LENGTH:-5000}
WEB_SCORING_FUNC=${WEB_SCORING_FUNC:-'rouge'}
WEB_CHUNKING_FUNC=${WEB_CHUNKING_FUNC:-'newline'}
WEB_TIMEOUT=${WEB_TIMEOUT:-300.0}

# Logging and checkpointing
NUM_GROUPS_TO_LOG=${NUM_GROUPS_TO_LOG:-4}
LOG_PATH=${LOG_PATH:-''}  # Auto-generated if empty
TAG=${TAG:-''}
WANDB_PROJECT=${WANDB_PROJECT:-'hyen-self-play'}
WANDB_NAME=${WANDB_NAME:-''}  # Auto-generated if empty
SAVE_EVERY=${SAVE_EVERY:-5}

# Build command with required parameters
CMD="uv run python -m tinker_cookbook.recipes.self_play.train \
  model_name=$MODEL \
  batch_size=$BZ \
  loss_fn=$LOSS_FN \
  learning_rate=$LR \
  lr_schedule=$LR_SCHEDULE \
  warmup_steps=$WARMUP_STEPS \
  max_tokens=$TOKENS \
  seed=$SEED \
  eval_every=$EVAL_EVERY \
  group_size=$GS \
  eval_group_size=$EGS \
  eval_n_batches=$EVAL_N_BATCHES \
  max_trajectory_tokens=$TRAJ_TOKENS \
  max_num_calls=$NS \
  eval_max_num_calls=$ENS \
  lora_rank=$RANK \
  self_play=$SELF_PLAY \
  handling_mode=$HANDLING_MODE \
  difficulty_reward_mode=$DIFFICULTY_REWARD_MODE \
  tool_reward_mode=$TOOL_REWARD_MODE \
  do_fallback_challenger=$DO_FALLBACK_CHALLENGER \
  stream_minibatch=$STREAM_MINIBATCH \
  num_minibatches=$NUM_MINIBATCHES \
  search_mode=$SEARCH_MODE \
  web_tool_port=$PORT \
  web_tool_topk=$WEB_TOPK \
  web_tool_content_length=$WEB_CONTENT_LENGTH \
  web_tool_scoring_func=$WEB_SCORING_FUNC \
  web_tool_chunking_func=$WEB_CHUNKING_FUNC \
  web_tool_timeout=$WEB_TIMEOUT \
  num_groups_to_log=$NUM_GROUPS_TO_LOG \
  run_tag=$TAG \
  wandb_project=$WANDB_PROJECT \
  save_every=$SAVE_EVERY \
  train_split=$TRAIN_SPLIT \
  eval_split=$EVAL_SPLIT"

# Add optional parameters if set
[[ -n "$RENDERER" ]] && CMD="$CMD renderer_name=$RENDERER"
[[ -n "$N_BATCHES" ]] && CMD="$CMD n_batches=$N_BATCHES"
[[ -n "$LOG_PATH" ]] && CMD="$CMD log_path=$LOG_PATH"
[[ -n "$WANDB_NAME" ]] && CMD="$CMD wandb_name=$WANDB_NAME"

# Execute command
eval $CMD
