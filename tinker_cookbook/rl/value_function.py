"""
Value function with GAE (Generalized Advantage Estimation) for RL training.

Provides a learned value function that runs locally on CPU/GPU, producing
per-step value estimates V(s_t) at transition boundaries. These are used to
compute GAE advantages, giving temporal credit assignment within trajectories.
"""

import logging
from typing import List

import chz
import numpy as np
import tinker
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinker_cookbook.rl.data_processing import _flatten_chunks, _is_prefix, FlatOb, FlatObElem
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


@chz.chz
class ValueFunctionConfig:
    """Configuration for the local value function model."""

    model_name: str
    """HuggingFace model name for the backbone."""
    learning_rate: float = 1e-4
    gae_lambda: float = 0.95
    gamma: float = 1.0
    """Discount factor. 1.0 = no discounting (finite horizon)."""
    device: str = "cuda"
    """Device for single-GPU mode. Ignored when gpu_ids is set."""
    gpu_ids: list[int] | None = None
    """Specific GPU ids to place the model on, e.g. [6, 7].
    When set, uses HuggingFace device_map='auto' constrained to these GPUs.
    The value head is placed on the first GPU in the list."""
    value_head_intermediate_size: int = 256
    num_value_epochs: int = 1
    freeze_backbone: bool = False
    """If True, only train the value head (not the backbone)."""
    lora_rank: int | None = 16
    """LoRA rank for backbone. None = full fine-tune (when freeze_backbone=False)."""
    gradient_checkpointing: bool = True


class ValueHead(nn.Module):
    """MLP head that maps hidden states to scalar value estimates."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (*, hidden_size)
        Returns:
            values: (*,) scalar value per position
        """
        x = F.relu(self.fc1(hidden_states))
        return self.fc2(x).squeeze(-1)


class ValueModel:
    """
    Local value function model for computing per-step value estimates.

    Uses a pretrained LM backbone (optionally with LoRA) and a ValueHead
    to produce V(s_t) at transition boundaries within trajectories.
    """

    def __init__(self, config: ValueFunctionConfig):
        self.config = config
        self._multi_gpu = config.gpu_ids is not None and len(config.gpu_ids) > 0

        if self._multi_gpu:
            # Multi-GPU: use device_map constrained to specific GPUs
            # Value head goes on the first listed GPU
            self.device = torch.device(f"cuda:{config.gpu_ids[0]}")
            max_memory = {i: "100GiB" for i in config.gpu_ids}
            logger.info(f"Loading value backbone across GPUs {config.gpu_ids}")
            self.backbone = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                output_hidden_states=True,
                device_map="auto",
                max_memory=max_memory,
            )
        else:
            self.device = torch.device(config.device)
            self.backbone = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                output_hidden_states=True,
            )

        # Apply LoRA if configured
        if config.lora_rank is not None and not config.freeze_backbone:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_rank * 2,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            logger.info(f"Applied LoRA (rank={config.lora_rank}) to value backbone")

        # Freeze backbone if configured
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
            logger.info("Value backbone frozen — only training value head")

        # Gradient checkpointing
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # Create value head
        hidden_size = self.backbone.config.hidden_size
        self.value_head = ValueHead(hidden_size, config.value_head_intermediate_size)
        self.value_head = self.value_head.to(dtype=torch.bfloat16)

        # Move to device (for single-GPU; multi-GPU backbone is already placed)
        if not self._multi_gpu:
            self.backbone = self.backbone.to(self.device)
        self.value_head = self.value_head.to(self.device)

        # Optimizer over all trainable params
        trainable_params = []
        trainable_params.extend(p for p in self.backbone.parameters() if p.requires_grad)
        trainable_params.extend(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

        n_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Value model initialized: {n_trainable:,} trainable parameters")

    def _build_sequence_and_boundaries(
        self, traj: Trajectory
    ) -> tuple[list[int], list[int]]:
        """
        Build a flat token sequence from a trajectory and identify step boundary positions.

        Step boundaries are the positions of the last token of each observation
        (before the action starts). These are the positions where we extract V(s_t).

        Returns:
            tokens: flat list of token ids
            boundary_positions: list of token indices where value estimates are extracted
        """
        full_sequence: FlatOb = []
        boundary_positions: list[int] = []

        for transition in traj.transitions:
            ob_flat = _flatten_chunks(transition.ob.chunks)

            if len(full_sequence) == 0:
                delta_ob_flat = ob_flat
            elif _is_prefix(full_sequence, ob_flat):
                delta_ob_flat = ob_flat[len(full_sequence) :]
            else:
                # Non-prefix observation: reset sequence
                # This shouldn't normally happen in search env, but handle gracefully
                full_sequence = []
                delta_ob_flat = ob_flat

            full_sequence.extend(delta_ob_flat)

            # Boundary is at the end of the observation (before action tokens)
            current_len = self._flat_ob_token_len(full_sequence)
            if current_len > 0:
                boundary_positions.append(current_len - 1)

            # Add action tokens
            full_sequence.extend(transition.ac.tokens)

        # Convert FlatOb to plain token list (dropping any non-int chunks)
        tokens: list[int] = []
        for elem in full_sequence:
            if isinstance(elem, int):
                tokens.append(elem)
            else:
                # ModelInputChunk — skip for value model (text-only)
                pass

        return tokens, boundary_positions

    @staticmethod
    def _flat_ob_token_len(flat_ob: FlatOb) -> int:
        """Count tokens in a flat observation sequence."""
        out = 0
        for elem in flat_ob:
            if isinstance(elem, int):
                out += 1
            else:
                out += elem.length
        return out

    @torch.no_grad()
    def compute_trajectory_values(self, traj: Trajectory) -> list[float]:
        """
        Compute value estimates V(s_t) at each step boundary in the trajectory.

        Returns one value per transition. Terminal value V(s_T) = 0 (not included).
        """
        self.backbone.eval()
        tokens, boundary_positions = self._build_sequence_and_boundaries(traj)

        if not tokens or not boundary_positions:
            return [0.0] * len(traj.transitions)

        input_ids = torch.tensor([tokens], device=self.device)
        outputs = self.backbone(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_size)

        # Clamp boundary positions to valid range
        max_pos = hidden_states.shape[1] - 1
        clamped_positions = [min(p, max_pos) for p in boundary_positions]

        boundary_hidden = hidden_states[0, clamped_positions, :]  # (num_steps, hidden_size)
        values = self.value_head(boundary_hidden)  # (num_steps,)

        return values.float().cpu().tolist()

    def train_on_batch(
        self,
        trajectory_groups_P: List[TrajectoryGroup],
        gamma: float,
    ) -> dict[str, float]:
        """
        Train the value model on a batch of trajectory groups using MSE loss.

        Accumulates gradients across all trajectories, then does a single
        optimizer step.
        """
        self.backbone.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        for traj_group in trajectory_groups_P:
            for traj, final_reward in zip(
                traj_group.trajectories_G, traj_group.final_rewards_G
            ):
                tokens, boundary_positions = self._build_sequence_and_boundaries(traj)
                if not tokens or not boundary_positions:
                    continue

                # Compute per-step rewards and returns
                step_rewards = [t.reward for t in traj.transitions]
                returns = compute_returns(step_rewards, final_reward, gamma)
                targets = torch.tensor(returns, device=self.device, dtype=torch.bfloat16)

                input_ids = torch.tensor([tokens], device=self.device)
                outputs = self.backbone(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                max_pos = hidden_states.shape[1] - 1
                clamped_positions = [min(p, max_pos) for p in boundary_positions]

                boundary_hidden = hidden_states[0, clamped_positions, :]
                values = self.value_head(boundary_hidden)

                # Align lengths (boundary_positions should match transitions)
                min_len = min(len(values), len(targets))
                loss = F.mse_loss(values[:min_len], targets[:min_len])

                loss.backward()
                total_loss += loss.item()
                n += 1

        if n > 0:
            # Average gradients
            for param in self.backbone.parameters():
                if param.grad is not None:
                    param.grad /= n
            for param in self.value_head.parameters():
                if param.grad is not None:
                    param.grad /= n
            self.optimizer.step()

        self.optimizer.zero_grad()

        avg_loss = total_loss / max(n, 1)
        return {
            "value_loss": avg_loss,
            "num_trajectories": n,
        }


def compute_returns(
    rewards: list[float], final_reward: float, gamma: float
) -> list[float]:
    """
    Compute discounted returns G_t for each step.

    G_t = sum_{k=t}^{T-1} gamma^{k-t} * r_k + gamma^{T-t} * final_reward

    With gamma=1.0: G_t = sum(rewards[t:]) + final_reward
    """
    T = len(rewards)
    returns: list[float] = []
    for t in range(T):
        G = 0.0
        for k in range(t, T):
            G += (gamma ** (k - t)) * rewards[k]
        G += (gamma ** (T - t)) * final_reward
        returns.append(G)
    return returns


def compute_gae_advantages(
    rewards: list[float],
    final_reward: float,
    values: list[float],
    gamma: float,
    gae_lambda: float,
) -> list[float]:
    """
    Compute GAE advantages for a single trajectory.

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}

    Terminal V(s_T) = 0. The final_reward is added to the last step's reward.
    """
    T = len(rewards)
    if T == 0:
        return []

    # Combine rewards: add final_reward to last step
    effective_rewards = list(rewards)
    effective_rewards[-1] += final_reward

    # Values with terminal value = 0
    values_extended = list(values) + [0.0]

    # Compute deltas
    deltas: list[float] = []
    for t in range(T):
        delta = effective_rewards[t] + gamma * values_extended[t + 1] - values_extended[t]
        deltas.append(delta)

    # Compute GAE (backward pass)
    advantages: list[float] = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae

    return advantages


def compute_gae_advantages_for_batch(
    trajectory_groups_P: List[TrajectoryGroup],
    value_model: ValueModel,
    gamma: float,
    gae_lambda: float,
) -> list[list[list[float]]]:
    """
    Compute GAE advantages for a full batch of trajectory groups.

    For each group, compute per-trajectory GAE advantages, then center
    advantages across trajectories within each group (matching GRPO's
    group-centering semantics).

    Returns:
        advantages_P_G_T[p][g][t]: per-step advantage for problem p,
        trajectory g, transition t.
    """
    advantages_P_G_T: list[list[list[float]]] = []

    for traj_group in trajectory_groups_P:
        group_advantages: list[list[float]] = []
        group_total_advantages: list[float] = []

        for traj, final_reward in zip(
            traj_group.trajectories_G, traj_group.final_rewards_G
        ):
            # Compute values at step boundaries
            values = value_model.compute_trajectory_values(traj)

            # Compute per-step rewards
            step_rewards = [t.reward for t in traj.transitions]

            # Compute GAE
            adv = compute_gae_advantages(
                step_rewards, final_reward, values, gamma, gae_lambda
            )
            group_advantages.append(adv)

            # Track total advantage for centering
            total_adv = sum(adv) if adv else 0.0
            group_total_advantages.append(total_adv)

        # Center advantages within the group
        if group_total_advantages:
            mean_total = sum(group_total_advantages) / len(group_total_advantages)
            n_trajs = len(group_advantages)
            if n_trajs > 0:
                # Distribute the centering offset evenly across steps
                for g_idx in range(n_trajs):
                    n_steps = len(group_advantages[g_idx])
                    if n_steps > 0:
                        offset = mean_total / n_steps
                        group_advantages[g_idx] = [
                            a - offset for a in group_advantages[g_idx]
                        ]

        advantages_P_G_T.append(group_advantages)

    return advantages_P_G_T
