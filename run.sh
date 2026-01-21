MODEL='openai/gpt-oss-120b'
LR=1e-5
BZ=4
TOKENS=2048
GS=8
TRAJ_TOKENS=$((GS*TOKENS))
NS=4
RANK=16

# LOGGING_DIR=
PORT=8000

uv run python -m tinker_cookbook.recipes.tool_use.self_play.train \
  model_name=$MODEL \
  batch_size=$BZ \
  learning_rate=$LR \
  max_tokens=$TOKENS \
  group_size=$GS \
  max_trajectory_tokens=$TRAJ_TOKENS \
  max_num_calls=$NS \
  lora_rank=$RANK \
  web_tool_port=$PORT 
