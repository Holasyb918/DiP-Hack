import functools
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.init import zeros_
from torch.nn.modules.module import T

# from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange

# code based on DDT

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Embed(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(
        self,
        labels,
    ):
        embeddings = self.embedding_table(labels)
        return embeddings


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def precompute_freqs_cis_2d(
    dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0
):
    # assert  H * H == end
    # flat_patch_pos = torch.linspace(-1, 1, end) # N = end
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)
    )  # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float()  # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float()  # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat(
        [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
    )  # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 1, 3, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B N H Hc
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # B, H, N, Hc
        k = (
            k.view(B, -1, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
            .contiguous()
        )  # B, H, N, Hc
        v = (
            v.view(B, -1, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
            .contiguous()
        )

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RAttention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), pos, mask=mask
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_heads=12,
        hidden_size=1152,
        num_blocks=18,
        patch_size=2,
        num_classes=1000,
        learn_sigma=True,
        deep_supervision=0,
        weight_path=None,
        load_ema=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.x_embedder = Embed(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size)

        self.final_layer = FinalLayer(hidden_size, in_channels * patch_size**2)

        self.weight_path = weight_path

        self.load_ema = load_ema
        self.blocks = nn.ModuleList(
            [DiTBlock(self.hidden_size, self.num_heads) for _ in range(self.num_blocks)]
        )
        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(
                self.hidden_size // self.num_heads, height, width
            ).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, masks=None):
        if masks is None:
            masks = [
                None,
            ] * self.num_blocks
        if isinstance(masks, torch.Tensor):
            masks = masks.unbind(0)
        if isinstance(masks, (tuple, list)) and len(masks) < self.num_blocks:
            masks = masks + [None] * (self.num_blocks - len(masks))

        B, _, H, W = x.shape
        x = torch.nn.functional.unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)
        x = self.x_embedder(x)
        pos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        B, L, C = x.shape
        t = self.t_embedder(t.view(-1)).view(B, -1, C)
        y = self.y_embedder(y).view(B, 1, C)
        condition = nn.functional.silu(t + y)
        for i, block in enumerate(self.blocks):
            x = block(x, condition, pos, masks[i])
        x = self.final_layer(x, condition)
        x = torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(),
            (H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size,
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
        hidden_size: int,
        patch_size: int,
        in_channels: int,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = in_channels * patch_size * patch_size
        self.fc1 = nn.Linear(in_features + hidden_size, in_features * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(in_features * 4, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, condition], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DiP(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_heads=12,
        hidden_size=1152,
        num_blocks=18,
        patch_size=2,
        num_classes=1000,
        learn_sigma=True,
        deep_supervision=0,
        weight_path=None,
        load_ema=False,
        base_channels: int = 64,
        patch_detail_type: str = "unet",
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.x_embedder = Embed(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes + 1, hidden_size)

        self.weight_path = weight_path

        self.load_ema = load_ema
        self.blocks = nn.ModuleList(
            [DiTBlock(self.hidden_size, self.num_heads) for _ in range(self.num_blocks)]
        )

        # patch detailer head
        self.patch_detail_type = patch_detail_type
        if self.patch_detail_type == "unet":
            self.patch_detailer_head = PatchDetailerHeadUnet(
                in_channels, in_channels, hidden_size, base_channels
            )
        elif self.patch_detail_type == "standard_mlp":
            self.patch_detailer_head = PatchDetailerHeadMlp(
                in_features=in_channels * patch_size**2,
                hidden_size=hidden_size,
                patch_size=patch_size,
                in_channels=in_channels,
                drop=0,
            )
        else:
            self.patch_detailer_head = FinalLayer(
                hidden_size, in_channels * patch_size**2
            )

        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(
                self.hidden_size // self.num_heads, height, width
            ).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        if self.patch_detail_type == "unet":
            pass
        elif self.patch_detail_type == "standard_mlp":
            pass
        else:
            nn.init.constant_(self.patch_detailer_head.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.patch_detailer_head.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.patch_detailer_head.linear.weight, 0)
            nn.init.constant_(self.patch_detailer_head.linear.bias, 0)

    def patchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        return rearrange(x, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=p, p2=p)

    def unpatchify_pixels(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        p = self.patch_size
        h, w = H // p, W // p
        # (B, L, p², C) -> (B, C, H, W)
        return rearrange(
            x, "(b h w) c p1 p2 -> b c (h p1) (w p2)", h=h, w=w, p1=p, p2=p
        )

    def forward(self, x, t, y, masks=None):
        if masks is None:
            masks = [
                None,
            ] * self.num_blocks
        if isinstance(masks, torch.Tensor):
            masks = masks.unbind(0)
        if isinstance(masks, (tuple, list)) and len(masks) < self.num_blocks:
            masks = masks + [None] * (self.num_blocks - len(masks))

        B, _, H, W = x.shape
        x0 = x
        _x = torch.nn.functional.unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)
        x = self.x_embedder(_x)
        pos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        B, L, C = x.shape
        t = self.t_embedder(t.view(-1)).view(B, -1, C)
        y = self.y_embedder(y).view(B, 1, C)
        condition = nn.functional.silu(t + y)
        for i, block in enumerate(self.blocks):
            x = block(x, condition, pos, masks[i])

        p = self.patch_size
        if self.patch_detail_type == "unet":
            unet_input = self.patchify_pixels(x0)
            unet_output = self.patch_detailer_head(
                unet_input, x.reshape(-1, x.shape[-1], 1, 1)
            )
            x = self.unpatchify_pixels(unet_output, H, W)
        elif self.patch_detail_type == "standard_mlp":
            x = self.patch_detailer_head(_x, x)  # B L D -> B L P²C
            x = rearrange(
                x,
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                h=H // p,
                w=W // p,
                p1=p,
                p2=p,
            )
        else:
            x = self.patch_detailer_head(x, condition)  # B L D -> B L P²C
            x = rearrange(
                x,
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                h=H // p,
                w=W // p,
                p1=p,
                p2=p,
            )
            # x1 = torch.nn.functional.fold(
            #     x.transpose(1, 2).contiguous(),
            #     (H, W),
            #     kernel_size=self.patch_size,
            #     stride=self.patch_size,
            # )
        return x


if __name__ == "__main__":
    model = DiP(
        in_channels=3,
        num_heads=16,
        hidden_size=1152,
        num_blocks=26,
        patch_size=16,
        num_classes=1000,
        learn_sigma=True,
        deep_supervision=0,
        weight_path=None,
        load_ema=False,
        base_channels=64,
        patch_detail_type="unet",
    )
    print("num parameters", sum(p.numel() for p in model.parameters()))
    x = torch.randn(2, 3, 256, 256)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 1000, (2,))
    output = model(x, t, y)
    print(f"Output shape: {output.shape}")
