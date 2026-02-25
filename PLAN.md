# Plan: Value Function with GAE for Search Environment

## Overview

Add a learned value function (critic) to the RL training loop. After rollouts complete, trajectories are run through a separate locally-running language model with a feedforward value head to produce per-step value estimates V(s_t). These replace the GRPO group-mean-centering with GAE (Generalized Advantage Estimation) advantages. The value model is trained on-the-fly each batch with MSE loss between predicted values and actual returns.

**Key constraint:** Tinker does not expose hidden states, so the value model must be a separate local HuggingFace model. The policy continues to train via Tinker as before.

## Files to Change

1. **NEW** `tinker_cookbook/rl/value_function.py` — Value model, GAE computation, training
2. **MODIFY** `tinker_cookbook/rl/data_processing.py` — Per-step advantage support
3. **MODIFY** `tinker_cookbook/rl/train.py` — Thread value model through training loop
4. **MODIFY** `tinker_cookbook/recipes/self_play/train.py` — CLI config for value function

## Step 1: Create `tinker_cookbook/rl/value_function.py`

### 1a. `ValueFunctionConfig` (chz dataclass)

```python
@chz.chz
class ValueFunctionConfig:
    model_name: str                          # HuggingFace model for backbone
    learning_rate: float = 1e-4
    gae_lambda: float = 0.95
    gamma: float = 1.0                       # 1.0 for finite horizon (no discounting)
    device: str = "cuda"
    value_head_intermediate_size: int = 256
    num_value_epochs: int = 1                # MSE training epochs per batch
    freeze_backbone: bool = False            # If True, only train the value head
    lora_rank: int | None = 16              # Apply LoRA to backbone (None = full fine-tune)
    gradient_checkpointing: bool = True      # Save memory for long sequences
```

### 1b. `ValueHead` (nn.Module)

Simple feedforward: `Linear(hidden_size, intermediate) -> ReLU -> Linear(intermediate, 1)`.

### 1c. `ValueModel` class

- Loads a HuggingFace `AutoModelForCausalLM` as backbone (bf16, `output_hidden_states=True`)
- **Entire model is trainable** (backbone + value head)
- When `lora_rank` is set (default 16): applies PEFT LoRA to the backbone, so only LoRA adapters + value head are trainable. This keeps memory manageable.
- When `lora_rank` is None: full fine-tune of the entire backbone + value head (expensive).
- When `freeze_backbone` is True: only the value head is trainable (cheapest, but less expressive).
- Enables `gradient_checkpointing` by default to manage memory with long sequences.
- Creates AdamW optimizer targeting all trainable parameters (LoRA params + value head, or all params, depending on config).

Key methods:

**`compute_trajectory_values(traj: Trajectory) -> list[float]`**
1. Flatten the trajectory's observations and actions into one token sequence (reusing the same prefix-extension logic as `trajectory_to_data` in `data_processing.py`)
2. Track "step boundary positions" — the last token of each observation, just before the action starts
3. Forward pass through backbone (`torch.no_grad()`) to get hidden states
4. Extract hidden states at boundary positions
5. Pass through ValueHead → scalar V(s_t) per step
6. Also returns V(s_terminal) = 0 (episode always terminates)

**`train_on_batch(trajectory_groups_P, gamma) -> dict[str, float]`**
For each trajectory:
1. Compute returns at each step: `G_t = sum(rewards[t:]) + final_reward` (with gamma=1.0)
2. Forward pass through backbone WITH gradient tracking (needed to update backbone/LoRA params)
3. Extract hidden states at boundary positions
4. Pass through ValueHead → predicted values
5. MSE loss between predicted values and actual returns
6. Backward + optimizer step (updates all trainable params: LoRA adapters + value head, or full model)

Returns metrics: `value_loss`, `explained_variance`, `mean_value`, `mean_return`.

**Memory approach:**
- `gradient_checkpointing=True` by default — trades compute for memory during backward pass
- LoRA (default rank 16) keeps trainable param count low (~0.5% of backbone)
- bf16 throughout
- Process trajectories one at a time (sequences can be up to 32K tokens)
- For value inference (computing advantages), uses `torch.no_grad()` — no activation storage needed

### 1d. `compute_gae_advantages` function

```python
def compute_gae_advantages(
    rewards: list[float],      # per-transition rewards, length T
    final_reward: float,       # from TrajectoryGroup
    values: list[float],       # V(s_t) for t=0..T-1
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> list[float]:
    # V(s_T) = 0 (terminal)
    # delta_t = (r_t + gamma * V(s_{t+1})) - V(s_t)
    #   where r for last step includes final_reward
    # A_t = sum_{l=0}^{T-1-t} (gamma * lambda)^l * delta_{t+l}
```

### 1e. `compute_gae_advantages_for_batch` function

Iterates over all trajectory groups → all trajectories within each group:
1. Compute per-step values using `value_model.compute_trajectory_values(traj)`
2. Compute per-step GAE advantages using `compute_gae_advantages(...)`
3. **Group centering**: subtract the mean total advantage across trajectories in each group (preserves GRPO-style zero-centering semantics while adding temporal credit assignment)

Returns `list[list[list[float]]]` — indexed as `[group_p][traj_g][step_t]`.

## Step 2: Modify `tinker_cookbook/rl/data_processing.py`

### 2a. Modify `trajectory_to_data` signature

Current: `trajectory_to_data(traj: Trajectory, traj_advantage: float)`

New: Add optional `per_transition_advantages: list[float] | None = None` parameter.

When `per_transition_advantages` is provided (length == number of transitions), each transition's action tokens get that transition's advantage value instead of the uniform scalar. The change is localized to the advantage assignment inside the loop (lines 163-165):

```python
# Current:
[traj_advantage] * len(ac_with_logprobs.tokens)

# With per_transition_advantages:
[per_transition_advantages[transition_idx]] * len(ac_with_logprobs.tokens)
```

Observation tokens still get advantage 0.

### 2b. Modify `assemble_training_data` signature

Current: `assemble_training_data(trajectory_groups_P, advantages_P: List[torch.Tensor])`

New: Add optional `per_step_advantages_P: list[list[list[float]]] | None = None` parameter.

When `per_step_advantages_P` is provided, call `trajectory_to_data(traj, per_transition_advantages=per_step_adv)` instead of `trajectory_to_data(traj, float(traj_advantage))`.

Exactly one of `advantages_P` or `per_step_advantages_P` should be non-None.

## Step 3: Modify `tinker_cookbook/rl/train.py`

### 3a. Add `value_function_config` to `Config` (line ~323)

```python
value_function_config: ValueFunctionConfig | None = None  # None = use GRPO
```

### 3b. Modify `prepare_minibatch` (lines 770-805)

Add optional `value_model: ValueModel | None = None` and GAE params. Replace the advantage computation block:

```python
if value_model is not None:
    per_step_advantages_P = compute_gae_advantages_for_batch(
        trajectory_groups_P, value_model, gamma, gae_lambda,
    )
    data_D, _metadata_D = assemble_training_data(
        trajectory_groups_P, per_step_advantages_P=per_step_advantages_P,
    )
else:
    advantages_P = compute_advantages(trajectory_groups_P)
    data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)
```

### 3c. Modify `do_train_step_and_get_sampling_client` (lines 978-1024)

Add `value_model` parameter, pass through to `prepare_minibatch`. After the policy training step, train the value model:

```python
if value_model is not None:
    vf_metrics = value_model.train_on_batch(
        trajectory_groups_P, gamma=cfg.value_function_config.gamma,
    )
    metrics.update({f"value_fn/{k}": v for k, v in vf_metrics.items()})
```

### 3d. Modify `do_sync_training` (lines 1028-1107)

Add `value_model` parameter, pass through to `do_train_step_and_get_sampling_client`.

### 3e. Modify `main` (lines 1110-1221)

After creating the training client (line ~1161), initialize the value model if configured:

```python
value_model = None
if cfg.value_function_config is not None:
    from tinker_cookbook.rl.value_function import ValueModel
    value_model = ValueModel(cfg.value_function_config)
```

Pass `value_model` to the training function call (line ~1192). The streaming and async training paths will NOT get value function support initially — only `do_sync_training`.

## Step 4: Modify `tinker_cookbook/recipes/self_play/train.py`

### 4a. Add value function fields to `CLIConfig`

```python
use_value_function: bool = False
value_model_name: str | None = None   # defaults to model_name if not set
value_lr: float = 1e-4
gae_lambda: float = 0.95
value_head_intermediate_size: int = 256
num_value_epochs: int = 1
value_lora_rank: int | None = 16     # LoRA rank for value backbone (None = full fine-tune)
freeze_value_backbone: bool = False   # If True, only train value head
```

### 4b. Construct `ValueFunctionConfig` in `cli_main`

When `use_value_function=True`, create a `ValueFunctionConfig` and pass it into `train.Config`. Append `_gae` to the run name for experiment tracking.

## Implementation Order

1. `value_function.py` — standalone, no dependencies on other changes
2. `data_processing.py` — small, backward-compatible signature change
3. `train.py` — thread value model through; backward compatible (value_model=None by default)
4. `recipes/self_play/train.py` — CLI wiring

## Not in Scope (for now)

- Value model checkpointing (save/load value head weights on resume)
- Streaming minibatch or async training paths (only sync training gets value function support)
- Image chunk handling in value model (search env is text-only)
- Normalizing advantages by std (start with centering only, matching GRPO)
