import math
import numpy as np
from math import pi
from typing import Optional, Tuple, Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from timm.models.vision_transformer import Mlp


# =============================================================================
# Inlined Dependencies
# =============================================================================

# -----------------------------------------------------------------------------
# RMSNorm (from models/rmsnorm.py)
# -----------------------------------------------------------------------------


class RMSNorm(torch.nn.Module):
    """RMSNorm normalization layer."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# -----------------------------------------------------------------------------
# SwiGLUFFN (from models/swiglu_ffn.py)
# -----------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    @torch.compile
    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# -----------------------------------------------------------------------------
# VisionRotaryEmbeddingFast (from models/pos_embed.py)
# -----------------------------------------------------------------------------


def broadcat(tensors, dim=-1):
    """Broadcast and concatenate tensors."""
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    """Rotate half of the dimensions."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    """Fast Vision Rotary Embedding."""

    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


# =============================================================================
# Basic Components
# =============================================================================


@torch.compile
def modulate(
    x: torch.Tensor, shift: Optional[torch.Tensor], scale: torch.Tensor
) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift"""
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


@torch.compile
def pixel_modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Apply pixel-wise AdaLN modulation.
    Args:
        x: (B*L, p^2, D_pix) pixel tokens
        shift: (B*L, p^2, D_pix) per-pixel shift
        scale: (B*L, p^2, D_pix) per-pixel scale
    """
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations with dropout for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


# =============================================================================
# Patch-level DiT Block (for global semantic learning)
# =============================================================================


class DiTBlock(nn.Module):
    """Patch-level DiT Block for global semantic learning.

    Features: RoPE, QK-Norm, RMSNorm, SwiGLU, AdaLN (with optional shift removal)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        **block_kwargs,
    ):
        super().__init__()

        # Normalization layers
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Attention
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs,
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=lambda: nn.GELU(approximate="tanh"),
                drop=0,
            )

        # AdaLN modulation
        self.wo_shift = wo_shift
        num_adaln_params = 4 if wo_shift else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, num_adaln_params * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope=None) -> torch.Tensor:
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
                4, dim=1
            )
            shift_msa, shift_mlp = None, None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=1)
            )

        # unsqueeze(1) adds sequence dimension for broadcasting: [B, D] -> [B, 1, D]
        shift_msa_expand = shift_msa.unsqueeze(1) if shift_msa is not None else None
        shift_mlp_expand = shift_mlp.unsqueeze(1) if shift_mlp is not None else None
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa_expand, scale_msa.unsqueeze(1)), rope=rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp_expand, scale_mlp.unsqueeze(1))
        )
        return x


class PatchDetailerHeadUnet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        global_dim=1152,  # DiT-XL hidden dim
        base_channels=64,
    ):
        super().__init__()

        # --- Downsampling Path (Encoder) ---
        # Level 1: 16x16 -> 16x16 (Features) -> 8x8 (Pooled)
        self.enc1 = ConvBlock(in_channels, base_channels)

        # Level 2: 8x8 -> 8x8 -> 4x4
        self.enc2 = ConvBlock(base_channels, base_channels * 2)

        # Level 3: 4x4 -> 4x4 -> 2x2
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        # Level 4: 2x2 -> 2x2 -> 1x1
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        # Input features at 1x1 is 512 (from UNet) + 1152 (from DiT) = 1664
        # Output is 512
        bottleneck_in_dim = (base_channels * 8) + global_dim
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bottleneck_in_dim, base_channels * 8, kernel_size=1), nn.SiLU()
        )

        # --- Upsampling Path (Decoder) ---
        # Level 4: 1x1 -> 2x2. Skip connection adds 512 channels.
        # In: 512 (up) + 512 (skip) = 1024 -> Out: 256
        self.up4 = UpBlock(base_channels * 8 + base_channels * 8, base_channels * 4)

        # Level 3: 2x2 -> 4x4.
        # In: 256 (up) + 256 (skip) = 512 -> Out: 128
        self.up3 = UpBlock(base_channels * 4 + base_channels * 4, base_channels * 2)

        # Level 2: 4x4 -> 8x8.
        # In: 128 (up) + 128 (skip) = 256 -> Out: 64
        self.up2 = UpBlock(base_channels * 2 + base_channels * 2, base_channels)

        # Level 1: 8x8 -> 16x16.
        # In: 64 (up) + 64 (skip) = 128 -> Out: 64 (See Table 4, last upsample output is 64)
        self.up1 = UpBlock(base_channels + base_channels, base_channels)

        # --- Output Layer ---
        # 1x1 Conv to map back to pixel space (3 channels)
        # Kernel size 1 is specified in Section 4.2 for the last layer
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, global_cond):
        """
        Args:
            x: Patch tensor [Batch, 3, 16, 16]
            global_cond: Global features from DiT [Batch, Global_Dim] or [Batch, Global_Dim, 1, 1]
        """
        # --- Encoder ---
        # x: [B, 3, 16, 16]
        s1 = self.enc1(x)  # [B, 64, 16, 16] (Skip connection 1)
        p1 = self.pool(s1)  # [B, 64, 8, 8]

        s2 = self.enc2(p1)  # [B, 128, 8, 8] (Skip connection 2)
        p2 = self.pool(s2)  # [B, 128, 4, 4]

        s3 = self.enc3(p2)  # [B, 256, 4, 4] (Skip connection 3)
        p3 = self.pool(s3)  # [B, 256, 2, 2]

        s4 = self.enc4(p3)  # [B, 512, 2, 2] (Skip connection 4)
        p4 = self.pool(s4)  # [B, 512, 1, 1]

        # --- Conditioning ---
        # Ensure global condition is [B, D, 1, 1] for concatenation
        if global_cond.dim() == 2:
            global_cond = global_cond.unsqueeze(-1).unsqueeze(-1)

        # Concatenate at the spatial bottleneck (1x1 resolution)
        # p4: [B, 512, 1, 1], global_cond: [B, 1152, 1, 1] -> [B, 1664, 1, 1]
        bottleneck_in = torch.cat([p4, global_cond], dim=1)
        bottleneck_out = self.bottleneck_conv(bottleneck_in)  # -> [B, 512, 1, 1]

        # --- Decoder ---
        # Up 4: [B, 512, 1, 1] -> [B, 256, 2, 2]
        d4 = self.up4(bottleneck_out, s4)

        # Up 3: [B, 256, 2, 2] -> [B, 128, 4, 4]
        d3 = self.up3(d4, s3)

        # Up 2: [B, 128, 4, 4] -> [B, 64, 8, 8]
        d2 = self.up2(d3, s2)

        # Up 1: [B, 64, 8, 8] -> [B, 64, 16, 16]
        d1 = self.up1(d2, s1)

        # --- Output ---
        out = self.out_conv(d1)  # [B, 3, 16, 16]

        return out


# --- Helper Blocks ---


class ConvBlock(nn.Module):
    """
    Standard block: Conv3x3 -> SiLU.
    Padding is 1 to maintain spatial resolution before pooling.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            # nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block: Upsample -> Concat with Skip -> ConvBlock
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        # 1. Bilinear or Nearest upsampling (Paper doesn't specify, bilinear is standard for details)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # 2. Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # 3. Convolution
        return self.conv(x)

class PatchDetailerHeadMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        patch_size: int,
        in_channels: int,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = in_channels * patch_size * patch_size
        self.fc1 = nn.Linear(in_features, in_features * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(in_features * 4, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class DiP(nn.Module):
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1152,  # D: semantic dimension (XL default)
        patch_depth: int = 26,  # N: number of patch-level blocks (XL default)
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_checkpoint: bool = False,
        base_channels: int = 64,
        patch_detail_type: str = "unet",
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.patch_detail_type = patch_detail_type

        # Calculate dimensions
        self.num_patches = (input_size // patch_size) ** 2
        self.num_pixels = patch_size * patch_size

        # === Patch-level pathway embeddings ===
        # Patch embedding for semantic pathway
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding for patch tokens
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )

        # Timestep and label embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # RoPE for patch-level attention
        if use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.patch_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.patch_rope = None

        # === Patch-level blocks (N blocks) ===
        self.patch_blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=False,
                )
                for _ in range(patch_depth)
            ]
        )

        # patch detailer head
        if self.patch_detail_type == "unet":
            self.patch_detailer_head = PatchDetailerHeadUnet(
                in_channels, in_channels, hidden_size, base_channels
            )
        elif self.patch_detail_type == "standard_mlp":
            self.patch_detailer_head = PatchDetailerHeadMlp(
                in_features=hidden_size,
                patch_size=patch_size,
                in_channels=in_channels,
                drop=0,
            )
        else:
            self.patch_detailer_head = nn.Linear(hidden_size, in_channels * patch_size * patch_size)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""

        # Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size, int(self.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # pixel_pos_embed = get_2d_sincos_pos_embed(
        #     self.hidden_size,
        #     self.input_size
        # )
        # self.pixel_pos_embed.data.copy_(torch.from_numpy(pixel_pos_embed).float().unsqueeze(0))

        # Initialize patch embedding
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.bias, 0)

        # Initialize embedders
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out AdaLN layers (DiTBlock has single adaLN_modulation)
        for block in self.patch_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def patchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reorganize pixel tokens into patch-local sequences.

        This is the KEY operation described in the paper:
        "To align with patch-level semantic tokens, we reshape into B·L sequences
        of p² pixel tokens, i.e., X ∈ R^(B·L)×p²×D_pix"

        Args:
            x: (B, D_pix, H, W) pixel embeddings
        Returns:
            (B, L, p², D_pix) patch-local pixel tokens
            where L = (H/p) * (W/p) is number of patches
        """
        p = self.patch_size
        # (B, D, H, W) -> (B, L, p², D) where L = (H/p)*(W/p)
        return rearrange(x, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=p, p2=p)

    def unpatchify_pixels(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Reorganize patch-local pixel tokens back to image layout.

        Args:
            x: (B, L, p², C) patch-local pixel tokens
            H, W: original image height and width
        Returns:
            (B, C, H, W) image tensor
        """
        p = self.patch_size
        h, w = H // p, W // p
        # (B, L, p², C) -> (B, C, H, W)
        return rearrange(
            x, "(b h w) c p1 p2 -> b c (h p1) (w p2)", h=h, w=w, p1=p, p2=p
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of PixelDiT.

        Args:
            x: (B, C, H, W) input images in pixel space
            t: (B,) diffusion timesteps
            y: (B,) class labels
        Returns:
            (B, C, H, W) predicted noise/velocity
        """
        B, C, H, W = x.shape
        p = self.patch_size
        L = (H // p) * (W // p)  # Number of patches

        # === Embeddings ===
        t_emb = self.t_embedder(t)  # (B, D)
        y_emb = self.y_embedder(y, self.training)  # (B, D)
        c = t_emb + y_emb  # (B, D) conditioning

        # === Patch-level pathway ===
        # Embed patches: (B, D, H/p, W/p) -> (B, L, D)
        s = self.patch_embed(x)  # (B, D, H/p, W/p)
        s = s.flatten(2).transpose(1, 2)  # (B, L, D)
        s = s + self.pos_embed  # Add position embedding

        # Apply patch-level blocks
        for block in self.patch_blocks:
            if self.use_checkpoint:
                s = checkpoint(block, s, c, self.patch_rope, use_reentrant=True)
            else:
                s = block(s, c, self.patch_rope)

        # Semantic conditioning: s_cond = s_N + t
        if self.patch_detail_type == "unet":
            s_cond = s + t_emb.unsqueeze(1)  # (B, L, D)
            s_cond = s_cond.reshape(-1, self.hidden_size)

            unet_input = self.patchify_pixels(x)
            unet_output = self.patch_detailer_head(unet_input, s_cond)
            output = self.unpatchify_pixels(unet_output, H, W)
        else:
            output = self.patch_detailer_head(s) # B L D -> B L P²C
            output = rearrange(output, "b l (p1 p2) c -> b c (h p1) (w p2)", h=H//p, w=W//p, p1=p, p2=p)

        return output

    def forward_with_cfg(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        """Forward with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        half_out = uncond_out + cfg_scale * (cond_out - uncond_out)

        return torch.cat([half_out, half_out], dim=0)


# =============================================================================
# Positional Embedding Utilities
# =============================================================================


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> np.ndarray:
    """Generate 2D sinusoidal positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D sinusoidal positional embedding from grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embedding from positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def test_patch_detailer_head():
    model = PatchDetailerHeadUnet()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    dummy_patch = torch.randn(2, 3, 16, 16)
    dummy_global = torch.randn(2, 1152)

    output = model(dummy_patch, dummy_global)

    print(f"Input shape: {dummy_patch.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == dummy_patch.shape, "Output shape mismatch!"
    print("Test Passed.")


def test_dip():
    model = DiP()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # x
    x = torch.randn(2, 3, 256, 256)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 1000, (2,))

    output = model(x, t, y)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


# --- Test Case ---
if __name__ == "__main__":
    test_patch_detailer_head()
    test_dip()
