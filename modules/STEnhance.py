import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.lorentz import Lorentz
from typing import Tuple
import math

class Spatial_Encoder(nn.Module):
    def __init__(self, embed_dim: int, lat_range: tuple, lon_range: tuple, num_rbf_centers: int,grid_size: int = 100):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.R_EARTH = 6371.0
        self.rff_num_features = 128 
        self.rff_scale_kms = [1.0, 3.0, 10.0, 30.0, 100.0]
        self.rff_output_dim = 64 
        num_scales = len(self.rff_scale_kms)
        k_per_scale = self.rff_num_features
        W = []
        for scale_km in self.rff_scale_kms:
            w_scale = torch.randn(3, k_per_scale) * (self.R_EARTH / scale_km)
            W.append(w_scale)
        
        self.register_buffer('rff_W', torch.cat(W, dim=1))  # Shape: [3, K * num_scales]
        self.rff_proj = nn.Linear(2 * k_per_scale * num_scales, self.rff_output_dim)

        self.rbf_num_anchors = num_rbf_centers
        self.rbf_top_k = 16
        self.rbf_output_dim = 64
        lat_anchors = torch.FloatTensor(self.rbf_num_anchors).uniform_(lat_range[0], lat_range[1])
        lon_anchors = torch.FloatTensor(self.rbf_num_anchors).uniform_(lon_range[0], lon_range[1])
        
        lat_anchors_rad = torch.deg2rad(lat_anchors)
        lon_anchors_rad = torch.deg2rad(lon_anchors)
        
        cos_lat_anchors = torch.cos(lat_anchors_rad)
        anchors_3d = torch.stack([cos_lat_anchors * torch.cos(lon_anchors_rad), cos_lat_anchors * torch.sin(lon_anchors_rad), torch.sin(lat_anchors_rad)], dim=-1)  # [num_anchors, 3]
        
        self.register_buffer('rbf_anchors_3d', anchors_3d)
        
        self.rbf_sigmas = nn.Parameter(torch.ones(self.rbf_num_anchors))
        self._init_rbf_sigmas()
        
        self.rbf_proj = nn.Linear(self.rbf_top_k, self.rbf_output_dim)
        gate_input_dim = self.rff_output_dim + self.rbf_output_dim
        
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, max(32, gate_input_dim // 4)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(max(32, gate_input_dim // 4), 2),
            nn.Softmax(dim=-1)
        )
            
        self.fusion_proj = nn.Linear(64, self.embed_dim)

    def _init_rbf_sigmas(self):
        with torch.no_grad():
            anchor_dists = torch.cdist(self.rbf_anchors_3d, self.rbf_anchors_3d, p=2)
            anchor_dists[anchor_dists == 0] = float('inf')
            min_dists, _ = torch.min(anchor_dists, dim=1)
            self.rbf_sigmas.data = min_dists / 3.0

    def _RFF(self, locations):
        original_shape = locations.shape
        locs_flat = locations.view(-1, 2)
        
        lat_rad = torch.deg2rad(locs_flat[:, 0])
        lon_rad = torch.deg2rad(locs_flat[:, 1])

        cos_lat = torch.cos(lat_rad)
        u = torch.stack([cos_lat * torch.cos(lon_rad), cos_lat * torch.sin(lon_rad), torch.sin(lat_rad)], dim=-1)
        uW = u @ self.rff_W
        features = torch.cat([torch.sin(uW), torch.cos(uW)], dim=-1)
        projected_features = self.rff_proj(features)
        
        return projected_features.view(*original_shape[:-1], self.rff_output_dim)

    def _RBF(self, locations):
        original_shape = locations.shape
        locs_flat = locations.view(-1, 2)
        
        lat_rad = torch.deg2rad(locs_flat[:, 0])
        lon_rad = torch.deg2rad(locs_flat[:, 1])
        cos_lat = torch.cos(lat_rad)
        points_3d = torch.stack([cos_lat * torch.cos(lon_rad), cos_lat * torch.sin(lon_rad), torch.sin(lat_rad)], dim=-1)  # [num_points, 3]
        
        dot_products = torch.matmul(points_3d, self.rbf_anchors_3d.T)  # [num_points, num_anchors]
        dot_products = torch.clamp(dot_products, -0.999999, 0.999999)
        spherical_dists = self.R_EARTH * torch.acos(dot_products)  # [num_points, num_anchors]

        sigma_sq = self.rbf_sigmas ** 2  # [num_anchors]
        rbf_responses = torch.exp(-spherical_dists ** 2 / (2 * sigma_sq.unsqueeze(0)))  # [num_points, num_anchors]
        
        top_k_values, top_k_indices = torch.topk(rbf_responses, self.rbf_top_k, dim=-1)
        projected_features = self.rbf_proj(top_k_values)
        
        return projected_features.view(*original_shape[:-1], self.rbf_output_dim)

    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        rff_features = self._RFF(locations)
        rbf_features = self._RBF(locations)
        gate_weights = self.gate_network(torch.cat([rff_features, rbf_features], dim=-1))  # [B, L, 2]
        E_s = rff_features * gate_weights[..., :1] + rbf_features * gate_weights[..., 1:2]
        return self.fusion_proj(E_s)

class Temporal_Encoder(nn.Module):
    def __init__(self, time_embed_dim: int, num_time_freqs: int = 8):
        super().__init__()
        freqs = torch.logspace(0, 3, num_time_freqs)
        self.register_buffer('time_freqs', freqs)

        # 1 (delta_t) + 2*num_time_freqs (sin/cos) + 7 (dow one-hot) + 24 (hour one-hot)
        time_features_dim = 1 + 2 * num_time_freqs + 7 + 24
        self.time_projection = nn.Linear(time_features_dim, time_embed_dim)

        self.decay_gate_weight = nn.Parameter(torch.randn(time_embed_dim))

    def forward(self, time_deltas: torch.Tensor, dow: torch.Tensor = None,
                hour: torch.Tensor = None) -> tuple:
        time_deltas = time_deltas.nan_to_num(0.0)
        time_deltas = torch.clamp(time_deltas, min=0.0, max=1e6)
        log_delta   = torch.log1p(time_deltas)    #这个地方是时间间隔的对数
        #torch.sin(freqs * log_delta), torch.cos(freqs * log_delta) 周期性时频特征

        freqs = self.time_freqs.view(1, 1, -1)
        time_features_list = [log_delta, torch.sin(freqs * log_delta), torch.cos(freqs * log_delta)]
        dow_one_hot = F.one_hot(dow.long() % 7, 7).to(time_deltas.dtype)  # [B, L, 7]
        hour_one_hot = F.one_hot(hour.long() % 24, 24).to(time_deltas.dtype)  # [B, L, 24]
        time_features_list.extend([dow_one_hot, hour_one_hot])
        time_features = torch.cat(time_features_list, dim=-1)
        time_embed = self.time_projection(time_features)  # [..., time_embed_dim]

        decay_gate = torch.sigmoid(torch.sum(time_embed * self.decay_gate_weight, dim=-1, keepdim=True))

        return time_embed, decay_gate


class ST_Enhancement(nn.Module):
    def __init__(self, lorentz_dim: int, geo_embed_dim: int, time_embed_dim: int,
                 semantic_proj_dim: int, euclidean_dim: int, num_rbf_centers: int, 
                 num_users: int, num_heads: int, lat_range: tuple, 
                 lon_range: tuple):
        super().__init__()
        # Geographic and temporal encoding components
        self.geo_embedding = Spatial_Encoder(embed_dim=geo_embed_dim, lat_range=lat_range, lon_range=lon_range, num_rbf_centers=num_rbf_centers,)
        self.time_decay = Temporal_Encoder(time_embed_dim=time_embed_dim)
        # Calculate actual context dimension after concatenation
        actual_context_dim = geo_embed_dim + time_embed_dim
        self.st_modulation = ST_Context_Modulation(lorentz_dim=lorentz_dim, context_dim=actual_context_dim, num_heads=num_heads)
        
    def forward(self, q_semantic: torch.Tensor, geo_seqs: torch.Tensor,
                time_deltas: torch.Tensor, day_of_week: torch.Tensor, hour_of_day: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geo_embed = self.geo_embedding(geo_seqs)  # [B, L, geo_embed_dim]
        time_embed, Delta_t = self.time_decay(time_deltas, day_of_week, hour_of_day)
        E_st = torch.cat([geo_embed, time_embed], dim=-1)
       
        return self.st_modulation(q_semantic, E_st), E_st, Delta_t

class ST_Context_Modulation(nn.Module):
    def __init__(self, lorentz_dim: int, context_dim: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = (lorentz_dim + 1) // num_heads
        self.total_head_dim = self.num_heads * self.head_dim
        self.manifold = Lorentz()
        self.q_proj = nn.Linear(lorentz_dim + 1, self.total_head_dim)
        self.k_proj = nn.Linear(context_dim, self.total_head_dim)
        self.v_proj = nn.Linear(context_dim, self.total_head_dim)

        self.out_proj = nn.Linear(lorentz_dim + 1, lorentz_dim + 1)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query_lorentz: torch.Tensor, E_st: torch.Tensor) -> torch.Tensor:
        B, L, _ = query_lorentz.shape
    
        q = self.q_proj(query_lorentz)
        k_tangent = self.k_proj(E_st)
        v_tangent = self.v_proj(E_st)
        
        q = q.view(B, L, self.num_heads, self.head_dim)
        k_tangent = k_tangent.view(B, L, self.num_heads, self.head_dim)
        v_tangent = v_tangent.view(B, L, self.num_heads, self.head_dim)
        
        q_tangent_full = self.manifold.logmap0(query_lorentz)
        q_tangent = q_tangent_full[..., 1:].view(B, L, self.num_heads, self.head_dim)
        attn_scores = torch.einsum("blhd,blhd->blh", q_tangent, k_tangent) / self.scale
        attn_output_tangent = torch.einsum("blh,blhd->blhd", torch.softmax(attn_scores, dim=-1), v_tangent)
        
        attn_output_tangent_spatial = attn_output_tangent.contiguous().view(B, L, self.total_head_dim)
        zeros = torch.zeros(B, L, 1, device=attn_output_tangent_spatial.device)
        attn_output_lorentz = self.manifold.expmap0(torch.cat([zeros, attn_output_tangent_spatial], dim=-1))
        output = self.manifold.projx(self.out_proj(attn_output_lorentz))
        
        return self.manifold.projx(self.manifold.mobius_add(query_lorentz, output))
