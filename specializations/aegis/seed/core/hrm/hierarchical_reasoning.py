"""
Hierarchical Reasoning Model (HRM) Implementation
Based on: Wang et al. 2025 "Hierarchical Reasoning Model" (arXiv:2506.21734)

Features:
- Dual-timescale processing (slow abstract planning, fast detailed execution)
- Adaptive computation time
- Recurrent state passing between modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache
        self.max_seq_len = max_seq_len
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]

        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]

        return self._cos_cached, self._sin_cached


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding"""

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE"""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rope(x)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out


class RecurrentBlock(nn.Module):
    """Recurrent transformer block with attention and FFN"""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        ffn_multiplier: int = 4
    ):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), mask)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class CrossAttention(nn.Module):
    """Cross-attention for communication between modules"""

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key_value.shape

        # Project
        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        out = self.out_proj(out)

        return out


class AdaptiveComputationTime(nn.Module):
    """Adaptive Computation Time mechanism"""

    def __init__(
        self,
        hidden_dim: int,
        halt_threshold: float = 0.99,
        max_steps: int = 16,
        penalty_weight: float = 0.001
    ):
        super().__init__()

        self.halt_threshold = halt_threshold
        self.max_steps = max_steps
        self.penalty_weight = penalty_weight

        # Halting unit
        self.halt_linear = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute halting probabilities

        Args:
            states: (batch, seq_len, hidden_dim)

        Returns:
            halt_probs: (batch, seq_len)
            ponder_cost: scalar
        """

        batch_size, seq_len, _ = states.shape

        # Compute halting probability
        halt_logits = self.halt_linear(states).squeeze(-1)  # (batch, seq_len)
        halt_probs = torch.sigmoid(halt_logits)

        # Ponder cost (average number of steps)
        ponder_cost = halt_probs.mean()

        return halt_probs, ponder_cost


class RecurrentPlanningModule(nn.Module):
    """High-level module for abstract planning"""

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
        max_cycles: int = 4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_cycles = max_cycles

        # Recurrent blocks
        self.blocks = nn.ModuleList([
            RecurrentBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        num_cycles: Optional[int] = None
    ) -> torch.Tensor:
        """
        Run multiple recurrent cycles

        Args:
            x: (batch, seq_len, hidden_dim)
            num_cycles: Number of recurrent cycles (default: max_cycles)

        Returns:
            Final state after recurrent processing
        """

        if num_cycles is None:
            num_cycles = self.max_cycles

        # Run recurrent cycles
        for cycle in range(num_cycles):
            for block in self.blocks:
                x = block(x)

        return self.norm(x)


class RecurrentExecutionModule(nn.Module):
    """Low-level module for detailed computation"""

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
        max_cycles: int = 4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_cycles = max_cycles

        # Recurrent blocks
        self.blocks = nn.ModuleList([
            RecurrentBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        high_level_guidance: Optional[torch.Tensor] = None,
        num_cycles: Optional[int] = None
    ) -> torch.Tensor:
        """
        Run multiple recurrent cycles with optional high-level guidance

        Args:
            x: (batch, seq_len, hidden_dim)
            high_level_guidance: Optional guidance from high-level module
            num_cycles: Number of recurrent cycles

        Returns:
            Final state after recurrent processing
        """

        if num_cycles is None:
            num_cycles = self.max_cycles

        # Incorporate high-level guidance if provided
        if high_level_guidance is not None:
            x = x + high_level_guidance

        # Run recurrent cycles
        for cycle in range(num_cycles):
            for block in self.blocks:
                x = block(x)

        return self.norm(x)


class HierarchicalReasoningModel(nn.Module):
    """
    Main Hierarchical Reasoning Model

    Combines high-level abstract planning with low-level detailed execution
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        high_level_layers: int = 6,
        low_level_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        h_cycles: int = 3,
        l_cycles: int = 3,
        max_act_steps: int = 16,
        use_act: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.use_act = use_act

        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # High-level planning module
        self.high_level = RecurrentPlanningModule(
            hidden_dim=d_model,
            n_layers=high_level_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_cycles=h_cycles
        )

        # Low-level execution module
        low_level_dim = d_model // 2
        self.low_level = RecurrentExecutionModule(
            hidden_dim=low_level_dim,
            n_layers=low_level_layers,
            n_heads=n_heads // 2,
            dropout=dropout,
            max_cycles=l_cycles
        )

        # Inter-module communication
        self.high_to_low = nn.Linear(d_model, low_level_dim)
        self.low_to_high = nn.Linear(low_level_dim, d_model)

        # Adaptive computation time
        if use_act:
            self.act = AdaptiveComputationTime(
                hidden_dim=d_model,
                max_steps=max_act_steps
            )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""

        nn.init.normal_(self.token_embedding.weight, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: (batch, seq_len)
            mask: Optional attention mask

        Returns:
            Dictionary containing logits and auxiliary outputs
        """

        batch_size, seq_len = input_ids.shape

        # Embed
        x = self.token_embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]

        # High-level abstract planning
        high_level_state = self.high_level(x)

        # Project to low-level dimension
        low_level_input = self.high_to_low(high_level_state)

        # Low-level detailed execution
        low_level_state = self.low_level(low_level_input)

        # Project back to high-level dimension
        low_to_high_state = self.low_to_high(low_level_state)

        # Combine states
        final_state = high_level_state + low_to_high_state

        # Adaptive computation time (if enabled)
        ponder_cost = None
        if self.use_act:
            halt_probs, ponder_cost = self.act(final_state)

        # Output projection
        logits = self.output_proj(final_state)

        return {
            'logits': logits,
            'ponder_cost': ponder_cost,
            'high_level_state': high_level_state,
            'low_level_state': low_level_state
        }
