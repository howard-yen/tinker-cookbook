# Self-play training

MODEL='openai/gpt-oss-120b'
LR=1e-5
BZ=2
TOKENS=8192
GS=8
TRAJ_TOKENS=$((GS*TOKENS))
NS=5
RANK=16

HANDLING_MODE="continue"
DIFFICULTY_REWARD_MODE="linear"
TOOL_REWARD_MODE="max"

# LOGGING_DIR=
PORT=8000

uv run python -m tinker_cookbook.recipes.tool_use.self_play.train \
  model_name=$MODEL \
  batch_size=$BZ \
  loss_fn="ppo" \
  learning_rate=$LR \
  max_tokens=$TOKENS \
  group_size=$GS \
  max_trajectory_tokens=$TRAJ_TOKENS \
  max_num_calls=$NS \
  lora_rank=$RANK \
  web_tool_port=$PORT \
  handling_mode=$HANDLING_MODE \
  difficulty_reward_mode=$DIFFICULTY_REWARD_MODE \
  tool_reward_mode=$TOOL_REWARD_MODE \
  self_play=true
