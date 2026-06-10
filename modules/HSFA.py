
import torch.nn as nn
from manifolds.lorentz import Lorentz
from manifolds.lorentz_functions import *

class LorentzLinear(nn.Module):
    """
    Lorentz-linear projection that keeps output on the hyperboloid.
    Input/Output: [..., d+1]
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.1, nonlin=None, merge=False, manifold=None):
        super().__init__()
        self.nonlin = nonlin
        self.manifold = manifold if manifold is not None else Lorentz()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.bias = bias
        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * 2.3)

    def forward(self, x, bias=None):
        if self.nonlin is not None:
            x = self.nonlin(x)
        if not self.merge:
            x = self.weight(self.dropout(x))
        else:
            x = self.weight(self.dropout(x.flatten(-2)))
        if bias is not None:
            x = x + bias

        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        denom = (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-9)
        scale = (time * time - self.manifold.k) / denom
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return self.manifold.projx(x)

    def reset_parameters(self):
        stdv = 0.02
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        step = self.in_features
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class MultiHeadAttention(nn.Module):
    """
    Stable MHA wrapper with residual + layer norm.
    Input/Output: [B, L, D]
    """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.mha(input_tensor, input_tensor, input_tensor, need_weights=False)
        hidden_states = self.out_dropout(attn_output)
        return self.layer_norm(hidden_states + input_tensor)


class FeedForward(nn.Module):
    """
    Point-wise FFN with residual + layer norm.
    """

    def __init__(self, hidden_size: int, forward_expansion: int, dropout: float):
        super().__init__()
        intermediate_size = forward_expansion * hidden_size
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class Spectral_Calibrator(nn.Module):
    """
    Local adaptive filter module for a single band branch.
    """

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.filter = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        # x_complex: [B, F, D]
        magnitude = x_complex.abs().permute(0, 2, 1)  # [B, D, F]
        return torch.sigmoid(self.filter(magnitude))  # [B, D, F]


class High_Low_SC(nn.Module):
    """
    MUFFIN-LFM style filter bank.
    Every band keeps its own LocalAdaptive filter and returns independent band outputs.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        num_bands: int = 6,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bands = num_bands
        self.complex_weight = nn.Parameter(torch.randn(1, hidden_size, 1, 2, dtype=torch.float32) * 0.02)
        self.local_filters = nn.ModuleList(
            [Spectral_Calibrator(hidden_size, kernel_size) for _ in range(num_bands)]
        )
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # input_tensor: [B, L, D]
        _, seq_len, _ = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")  # [B, F, D]
        freq_len = x.size(1)
        weight = torch.view_as_complex(self.complex_weight)  # [1, D, 1]

        frequency_bands = []
        for band_idx in range(self.num_bands):
            local_filter = self.local_filters[band_idx](x)  # [B, D, F]
            filtered_weight = torch.complex(
                local_filter * weight.real, local_filter * weight.imag
            ).permute(0, 2, 1)  # [B, F, D]
            x_filtered = x * filtered_weight

            band_start = band_idx * freq_len // self.num_bands
            band_end = (band_idx + 1) * freq_len // self.num_bands
            masked_x = torch.zeros_like(x_filtered)
            masked_x[:, band_start:band_end, :] = x_filtered[:, band_start:band_end, :]

            sequence_emb_fft = torch.fft.irfft(masked_x, n=seq_len, dim=1, norm="ortho")
            band_output = self.out_dropout(sequence_emb_fft)
            frequency_bands.append(self.layer_norm(band_output + input_tensor))

        # [B, num_bands, L, D]
        return torch.stack(frequency_bands, dim=1)


class TransformerFrequency(nn.Module):
    """
    LFM-only frequency module that exposes all/low/high band outputs.
    """

    def __init__(
        self,
        model_dim: int,
        euclidean_dim: int,
        dt_min: float,
        dt_max: float,
        state_dim: int = None,
        dropout: float = 0.1,
        forward_expansion: int = 2,
        num_bands: int = 6,
        low_band_count: int = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim if state_dim is not None else model_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.num_bands = max(2, int(num_bands))
        if low_band_count is None:
            self.low_band_count = self.num_bands // 2
        else:
            self.low_band_count = int(low_band_count)
        if not (1 <= self.low_band_count < self.num_bands):
            raise ValueError(
                f"low_band_count must be in [1, {self.num_bands - 1}], got {self.low_band_count}"
            )
        _ = forward_expansion

        self.filterlayer = High_Low_SC(
            hidden_size=model_dim,
            dropout=dropout,
            num_bands=self.num_bands,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        euclidean_features: torch.Tensor,
    ) -> tuple:
        _ = euclidean_features
        all_bands = self.filterlayer(inputs)  # [B, num_bands, L, D]
        low_count = self.low_band_count
        low_bands = all_bands[:, :low_count, :, :]
        high_bands = all_bands[:, low_count:, :, :]
        return all_bands, low_bands, high_bands


class SoftMoEGate(nn.Module):
    """
    Soft gate for MoE experts (weights sum to 1 per sample).
    """

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
       
        self.gate = nn.Sequential(
            nn.Linear(2*input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_experts),
        )


    def forward(self, pooled_context: torch.Tensor) -> torch.Tensor:
        pooled_context = torch.fft.rfft(pooled_context, dim=1, norm="ortho")  # [B, F, D]
        magnitude = pooled_context.abs()
        phase = torch.angle(pooled_context)

        mag_features = torch.mean(magnitude, dim=1)
        phase_features = torch.mean(phase, dim=1)
        freq_features = torch.cat([mag_features, phase_features], dim=-1)
        logits = self.gate(freq_features)
        probs = torch.softmax(logits, dim=-1)
        return probs / probs.sum(dim=-1, keepdim=True)  # [B, num_bands]


class SoftMoEGate_low(nn.Module):
    """
    Soft gate for MoE experts (weights sum to 1 per sample).
    """

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_experts),
        )

    def forward(self, pooled_context: torch.Tensor) -> torch.Tensor:
        logits = self.gate(pooled_context)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        return torch.softmax(logits, dim=-1)

class TransformerParallel(nn.Module):
    """
    Requested architecture:
    1) low-frequency branch: gsp residual + low-band aggregation
    2) high-frequency branch: high-band aggregation
    3) if a branch has only one band, skip its Soft-MoE
    """

    def __init__(
        self,
        model_dim: int,
        euclidean_dim: int,
        dt_min: float,
        dt_max: float,
        state_dim: int = None,
        dropout: float = 0.1,
        num_heads: int = 4,
        forward_expansion: int = 2,
        alpha: float = 0.5,
        low_freq_ratio: float = 0.5,
        num_bands: int = 6,
        low_band_count: int = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.lorentz_manifold = Lorentz(1)
        self.model_dim = model_dim
        self.state_dim = state_dim if state_dim is not None else model_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        _ = alpha
        _ = low_freq_ratio

        heads = self._resolve_num_heads(model_dim, num_heads)
        self.num_bands = max(2, int(num_bands))
        if low_band_count is None:
            self.low_band_count = self.num_bands // 2
        else:
            self.low_band_count = int(low_band_count)
        if not (1 <= self.low_band_count < self.num_bands):
            raise ValueError(
                f"low_band_count must be in [1, {self.num_bands - 1}], got {self.low_band_count}"
            )
        self.high_band_count = self.num_bands - self.low_band_count
        self.in_proj = nn.Linear(self.model_dim, self.state_dim)
        self.out_proj = nn.Linear(self.state_dim, self.model_dim)

        self.attention = MultiHeadAttention(model_dim, heads, dropout)  # gsp
        self.frequency_encoder = TransformerFrequency(
            model_dim=model_dim,
            euclidean_dim=euclidean_dim,
            dt_min=dt_min,
            dt_max=dt_max,
            state_dim=state_dim,
            dropout=dropout,
            forward_expansion=forward_expansion,
            num_bands=self.num_bands,
            low_band_count=self.low_band_count,
            kernel_size=kernel_size,
        )

        self.use_low_moe = self.low_band_count > 1
        self.use_high_moe = self.high_band_count > 1
        self.low_moe_gate = (
            SoftMoEGate_low(model_dim, self.low_band_count) if self.use_low_moe else None
        )
        self.high_moe_gate = (
            SoftMoEGate(model_dim, self.high_band_count) if self.use_high_moe else None
        )
        self.mix_norm = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForward(model_dim, forward_expansion, dropout)

    @staticmethod
    def _resolve_num_heads(embed_size: int, target_heads: int) -> int:
        for heads in range(min(target_heads, embed_size), 0, -1):
            if embed_size % heads == 0:
                return heads
        return 1

    @staticmethod
    def _apply_moe(experts: list, weights: torch.Tensor) -> torch.Tensor:
        # experts: list of [B, L, D], weights: [B, E]
        stacked = torch.stack(experts, dim=1)  # [B, E, L, D]
        return torch.sum(weights[:, :, None, None] * stacked, dim=1)  # [B, L, D]

    def forward(
        self,
        inputs: torch.Tensor,
        euclidean_features: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = inputs.shape
        device = inputs.device

        x = self.lorentz_manifold.projx(inputs)
        x_tan = self.lorentz_manifold.logmap0(x)  # [B,L,d+1]
        x_spatial = x_tan[..., 1:]  # [B,L,d]

        h = self.in_proj(x_spatial)  # [B,L,C]
        inputs = h

        gsp = self.attention(inputs)  # [B, L, D]
        all_bands, low_bands, high_bands = self.frequency_encoder(inputs, euclidean_features)
        _ = all_bands  # kept for explicit module visibility

        # low-frequency branch
        low_experts = [low_bands[:, idx, :, :] for idx in range(low_bands.size(1))]
        if self.use_low_moe:
            low_gate_context = torch.cat(
                [
                    torch.mean(gsp, dim=1),
                    torch.mean(low_bands, dim=(1, 2)),
                ],
                dim=-1,
            )
            low_weights = self.low_moe_gate(low_gate_context)  # [B, low_band_count]
            low_freq_out = self._apply_moe(low_experts, low_weights)
        else:
            low_freq_out = low_experts[0]
        low_moe_out = gsp + low_freq_out

        # high-frequency branch
        high_experts = [high_bands[:, idx, :, :] for idx in range(high_bands.size(1))]
        if self.use_high_moe:
            high_weights = self.high_moe_gate(inputs)  # [B, high_band_count]
            high_moe_out = self._apply_moe(high_experts, high_weights)
        else:
            high_moe_out = high_experts[0]

        if (not self.use_low_moe) or (not self.use_high_moe):
            # When low or high side has only one band, skip soft-MoE fusion and use direct sum.
            fused = low_moe_out + high_moe_out
        else:
            fused = 0.7 * low_moe_out + 0.3 * high_moe_out

        out_spatial = self.out_proj(fused)  # [B,L,d]
        out_spatial = self.mix_norm(out_spatial)  

        zero_time = torch.zeros(B, L, 1, device=device, dtype=out_spatial.dtype)
        out_tan_full = torch.cat([zero_time, out_spatial], dim=-1)  # [B,L,d+1]
        out_lorentz = self.lorentz_manifold.projx(self.lorentz_manifold.expmap0(out_tan_full))
        return out_lorentz


class HSFALayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        euclidean_dim: int,
        dt_min: float,
        dt_max: float,
        state_dim: int = None,
        dropout: float = 0.1,
        num_heads: int = 4,
        forward_expansion: int = 2,
        alpha: float = 0.5,
        num_bands: int = 6,
        low_band_count: int = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.lorentz_dim = model_dim
        self.lorentz_manifold = Lorentz(1)
        self.parallel_block = TransformerParallel(
            model_dim=model_dim,
            euclidean_dim=euclidean_dim,
            dt_min=dt_min,
            dt_max=dt_max,
            state_dim=state_dim,
            dropout=dropout,
            num_heads=num_heads,
            forward_expansion=forward_expansion,
            alpha=alpha,
            num_bands=num_bands,
            low_band_count=low_band_count,
            kernel_size=kernel_size,
        )
        self.output_projection = LorentzLinear(model_dim + 1, model_dim + 1, bias=True, dropout=dropout)

    def _tangent_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        tan = self.lorentz_manifold.logmap0(x)  # [..., d+1]
        spatial = tan[..., 1:]  # [..., d]
        spatial = nn.functional.layer_norm(spatial, normalized_shape=(self.lorentz_dim,), eps=1e-5)
        u_full = torch.cat([torch.zeros_like(spatial[..., :1]), spatial], dim=-1)  # [..., d+1]
        out = self.lorentz_manifold.expmap0(u_full)
        return self.lorentz_manifold.projx(out)

    def forward(
        self,
        inputs: torch.Tensor,
        euclidean_features: torch.Tensor,
    ) -> torch.Tensor:
        mixed_output = self.parallel_block(inputs , euclidean_features)            # [B, L, d]
        projected_output = self.output_projection(mixed_output)                            # [B, L, d]
        output = self.lorentz_manifold.mobius_add(inputs, projected_output)
        output = self._tangent_layer_norm(output)
        return output


class MultiLayerHSFA(nn.Module):
    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        euclidean_dim: int,
        dt_min: float,
        dt_max: float,
        state_dim: int = None,
        dropout: float = 0.1,
        num_heads: int = 4,
        forward_expansion: int = 2,
        alpha: float = 0.5,
        num_bands: int = 6,
        low_band_count: int = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HSFALayer(
                    model_dim=model_dim,
                    euclidean_dim=euclidean_dim,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    state_dim=state_dim,
                    dropout=dropout,
                    num_heads=num_heads,
                    forward_expansion=forward_expansion,
                    alpha=alpha,
                    num_bands=num_bands,
                    low_band_count=low_band_count,
                    kernel_size=kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        euclidean_features: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x, euclidean_features)
        return x
