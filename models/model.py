
import torch.nn as nn
from manifolds.lorentz import Lorentz
from manifolds.lorentz_functions import *
from manifolds.gyro_operations import GyroVectorOperations
from modules.STEnhance import ST_Enhancement
from modules.HSFA import MultiLayerHSFA
import math



class HSFA(nn.Module):
    def __init__(self, config):
        super(HSFA, self).__init__()
        self.num_user = config.num_user
        self.num_poi = config.num_poi
        self.num_cat = config.num_cat
        self.num_region = config.num_region
        self.device = config.device

        model_properties = config.model_params
        self.c = 1.0
        self.num_dim = model_properties['num_dim']
        self.hyp_dim = self.num_dim + 1  # dimension for the Lorentz model
        ##########################################################################################
        self.use_target_time = bool(model_properties.get('use_target_time', True))
        self.target_time_hour_slots = int(model_properties.get('target_time_hour_slots', 24))
        self.target_time_dow_slots = int(model_properties.get('target_time_dow_slots', 7))

        if self.num_dim % 2 != 0:
            raise ValueError(f"num_dim must be even to apply Givens rotations, got {self.num_dim}")
        self._target_time_num_pairs = self.num_dim // 2

        # Learn per-timeslot rotation angles (hour-of-day and day-of-week)
        self.tgt_hour_angle = nn.Embedding(self.target_time_hour_slots, self._target_time_num_pairs)
        self.tgt_dow_angle = nn.Embedding(self.target_time_dow_slots, self._target_time_num_pairs)


        # coefficient between hour and dow rotations (scalar, learned; sigmoid(logit)=alpha)
        gamma_init = float(model_properties.get('target_time_alpha_init', 0.7))
        gamma_init = min(max(gamma_init, 1e-4), 1.0 - 1e-4)
        self.target_time_gamma = nn.Parameter(
            torch.tensor(math.log(gamma_init / (1.0 - gamma_init)), dtype=torch.float32)
        )

        ##########################################################################################
        self.num_negs = model_properties.get('num_negs')
        self.dropout = model_properties['dropout']
        self.num_knn = model_properties.get('num_knn')
        self.euclidean_dim = model_properties.get('euclidean_dim')
        self.state_dim = model_properties.get('state_dim')
        self.geo_embed_dim = model_properties.get('geo_embed_dim')
        self.time_embed_dim = model_properties.get('time_embed_dim')
        self.semantic_proj_dim = model_properties.get('semantic_proj_dim')
        self.num_rbf_centers = model_properties.get('num_rbf_centers')
        self.num_heads = model_properties.get('num_heads')
        self.lat_range = tuple(model_properties.get('lat_range'))
        self.lon_range = tuple(model_properties.get('lon_range'))
        self.dt_min = model_properties.get('dt_min')
        self.dt_max = model_properties.get('dt_max')
        # Pre-spatiotemporal sequence encoder (HSFA style)
        self.use_pre_st_hsfa = bool(model_properties.get('use_pre_st_hsfa', True))
        self.num_pre_hsfa_layers = int(
            model_properties.get('num_pre_hsfa_layers', model_properties.get('num_ssm_layers', 1))
        )
        self.pre_hsfa_heads = int(
            model_properties.get('pre_hsfa_heads', self.num_heads if self.num_heads is not None else 4)
        )
        self.pre_hsfa_forward_expansion = int(model_properties.get('pre_hsfa_forward_expansion', 2))
        self.pre_hsfa_alpha = float(model_properties.get('pre_hsfa_alpha', 0.5))
        self.pre_hsfa_num_bands = int(model_properties.get('pre_hsfa_num_bands', 6))
        pre_hsfa_low_band_count = model_properties.get('pre_hsfa_low_band_count', None)
        self.pre_hsfa_low_band_count = (
            int(pre_hsfa_low_band_count) if pre_hsfa_low_band_count is not None else None
        )
        self.pre_hsfa_kernel_size = int(model_properties.get('pre_hsfa_kernel_size', 3))

        self.Lorentz = Lorentz(self.c)
        # Embeddings
        self.user_embed = nn.Embedding(self.num_user, self.hyp_dim)
        self.user_embed.weight.data = self.Lorentz.random_normal((self.num_user, self.hyp_dim))
        self.poi_embed = nn.Embedding(self.num_poi + 1, self.hyp_dim, padding_idx=self.num_poi)
        self.poi_embed.weight.data = self.Lorentz.random_normal((self.num_poi + 1, self.hyp_dim))
        self.cat_embed = nn.Embedding(self.num_cat + 1, self.hyp_dim, padding_idx=self.num_cat)
        self.cat_embed.weight.data = self.Lorentz.random_normal((self.num_cat + 1, self.hyp_dim))
        self.geo_embed = nn.Embedding(self.num_region + 1, self.hyp_dim, padding_idx=self.num_region)
        self.geo_embed.weight.data = self.Lorentz.random_normal((self.num_region + 1, self.hyp_dim))

        self.logit_beta = nn.Parameter(torch.tensor(0.5))  # α
        self.dist_tau = nn.Parameter(torch.tensor(1.0))  # τ
        self.semantic_norm = nn.LayerNorm(self.num_dim)
        # # Core components are now mandatory
        self.spatiotemporal_context= ST_Enhancement(lorentz_dim=self.num_dim, geo_embed_dim=self.geo_embed_dim,
                                                            time_embed_dim=self.time_embed_dim,
                                                            semantic_proj_dim=self.semantic_proj_dim,
                                                            euclidean_dim=self.euclidean_dim,
                                                            num_rbf_centers=self.num_rbf_centers,
                                                            num_users=self.num_user, num_heads=self.num_heads,
                                                            lat_range=self.lat_range, lon_range=self.lon_range)

        # Calculate actual euclidean dimension from spatiotemporal channel
        actual_euclidean_dim = self.geo_embed_dim + self.time_embed_dim
        # Pre-HSFA is requested to run with full tangent coordinates (d+1).
        self.pre_hsfa_model_dim = self.num_dim
        self.pre_hsfa_state_dim = self.state_dim
        self.pre_hsfa_context_dim = actual_euclidean_dim
        hsfa_dt_min = float(self.dt_min) if self.dt_min is not None else 1e-3
        hsfa_dt_max = float(self.dt_max) if self.dt_max is not None else 0.1

        if self.use_pre_st_hsfa and self.num_pre_hsfa_layers > 0:
            self.pre_hsfa = MultiLayerHSFA(
                num_layers=self.num_pre_hsfa_layers,
                model_dim=self.pre_hsfa_model_dim,
                euclidean_dim=self.pre_hsfa_context_dim,
                dt_min=hsfa_dt_min,
                dt_max=hsfa_dt_max,
                state_dim=self.pre_hsfa_state_dim,
                dropout=self.dropout,
                num_heads=self.pre_hsfa_heads,
                forward_expansion=self.pre_hsfa_forward_expansion,
                alpha=self.pre_hsfa_alpha,
                num_bands=self.pre_hsfa_num_bands,
                low_band_count=self.pre_hsfa_low_band_count,
                kernel_size=self.pre_hsfa_kernel_size,
            )
        else:
            self.pre_hsfa = None


        self.poi_decoder = nn.Linear(self.hyp_dim, self.num_poi)
        self.cat_decoder = nn.Linear(self.hyp_dim, self.num_cat)
        self.geo_decoder = nn.Linear(self.hyp_dim, self.num_region)


    def _unpack_inputs(self, inputs):
        """
        Support both legacy inputs (8-tuple) and TS-NPR inputs (10-tuple).

        Legacy (len==8):
            (poi_seqs, cat_seqs, geo_seqs, user_list, locations, time_deltas, dow, hour)

        Time-specific (len==10):
            (..., dow, hour, target_dow, target_hour)
        """
        if len(inputs) == 8:
            poi_seqs, cat_seqs, geo_seqs, user_list, locations, time_deltas, dow, hour = inputs
            target_dow, target_hour = None, None
        elif len(inputs) == 10:
            poi_seqs, cat_seqs, geo_seqs, user_list, locations, time_deltas, dow, hour, target_dow, target_hour = inputs
        else:
            raise ValueError(f"Unexpected inputs length={len(inputs)}. Expected 8 or 10.")

        return poi_seqs, cat_seqs, geo_seqs, user_list, locations, time_deltas, dow, hour, target_dow, target_hour

    def _angles_to_rotary_weights(self, angles: torch.Tensor) -> torch.Tensor:
        """angles: [..., num_pairs] -> weights: [..., num_dim] (interleaved cos/sin)."""
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return torch.stack([cos, sin], dim=-1).reshape(*angles.shape[:-1], self.num_dim)

    def _apply_target_time_rotation(self,
                                    x_lorentz: torch.Tensor,
                                    target_dow: torch.Tensor,
                                    target_hour: torch.Tensor) -> torch.Tensor:
        """
        Apply Time2Rotation-style target-time conditioning via per-sample Givens rotation.

        x_lorentz: [B, L, hyp_dim] (Lorentz embeddings)
        target_dow : [B, L] day-of-week indices (0..6)
        target_hour: [B, L] hour-of-day indices (0..23)

        We compute two angle embeddings (dow/hour), beta them with a learned scalar alpha,
        and rotate the spatial components (dims 1:) while keeping time-axis (dim 0) unchanged.
        """
        if (not self.use_target_time) or (target_dow is None) or (target_hour is None):
            return x_lorentz

        B, L, H = x_lorentz.shape
        if H != self.hyp_dim:
            raise ValueError(f"Expected Lorentz embeddings dim={self.hyp_dim}, got {H}")

        ##################################target#################################################
        target_hour = (target_hour.long() % self.target_time_hour_slots).clamp(min=0)
        target_dow = (target_dow.long() % self.target_time_dow_slots).clamp(min=0)
        hour_angles = self.tgt_hour_angle(target_hour)  # [B,L,num_pairs]
        dow_angles = self.tgt_dow_angle(target_dow)  # [B,L,num_pairs]
        #
        gamma = torch.sigmoid(self.target_time_gamma)  # scalar in (0,1)
        angles = gamma  * hour_angles + (1.0 - gamma ) * dow_angles
        # ##################################target#################################################
        rotary_w = self._angles_to_rotary_weights(angles)  # [B,L,num_dim]
        time_axis = x_lorentz[..., 0:1]  # [B,L,1]
        spatial = x_lorentz[..., 1:]  # [B,L,num_dim]
        spatial_rot = givens_rotations(
            rotary_w.reshape(-1, self.num_dim),
            spatial.reshape(-1, self.num_dim)
        ).reshape(B, L, self.num_dim)

        out = torch.cat([time_axis, spatial_rot], dim=-1)
        return self.Lorentz.projx(out)


    def hyperbolic_embedding(self, user_emb: torch.Tensor, poi_emb: torch.Tensor,
                                   cat_emb: torch.Tensor, geo_emb: torch.Tensor) -> torch.Tensor:
        B, L = poi_emb.shape[:2]
        user_expanded = user_emb.unsqueeze(1).expand(-1, L, -1)
        user_tangent = GyroVectorOperations.log_map_origin(user_expanded)
        poi_tangent = GyroVectorOperations.log_map_origin(poi_emb)
        cat_tangent = GyroVectorOperations.log_map_origin(cat_emb)
        geo_tangent = GyroVectorOperations.log_map_origin(geo_emb)
        combined_tangent = (0.5 * user_tangent + 0.3 * poi_tangent +
                            0.1 * cat_tangent + 0.1 * geo_tangent)
        normalized_spatial = self.semantic_norm(combined_tangent[..., 1:])
        normalized_tangent = torch.zeros_like(combined_tangent)
        normalized_tangent[..., 1:] = normalized_spatial

        return GyroVectorOperations.exp_map_origin(normalized_tangent)


    def forward(self, inputs):

        poi_seqs, cat_seqs, geo_seqs, user_list, locations, time_deltas, dow, hour, target_dow, target_hour = self._unpack_inputs(
            inputs)

        # poi_seqs, cat_seqs, geo_seqs, user_list, locations, time_deltas, dow, hour = self._unpack_inputs(inputs)

        token_mask = (poi_seqs != self.num_poi)
        poi_embeds = self.Lorentz.projx(self.poi_embed(poi_seqs))
        cat_embeds = self.Lorentz.projx(self.cat_embed(cat_seqs))
        geo_embeds = self.Lorentz.projx(self.geo_embed(geo_seqs))
        user_embeds = self.Lorentz.projx(self.user_embed(user_list))
        semantic_input = self.Lorentz.projx(
            self.hyperbolic_embedding(user_embeds, poi_embeds, cat_embeds, geo_embeds))


        semantic_fused, E_st, _ = self.spatiotemporal_context(semantic_input, locations, time_deltas, dow, hour)
        E_b = self.pre_hsfa(semantic_fused, E_st)
        traj_embeds = self.Lorentz.projx(E_b)


    
        traj_embeds = self._apply_target_time_rotation(traj_embeds, target_dow, target_hour)

        tangent_embeds = self.Lorentz.logmap0(traj_embeds)  # shape: (B, L, d+1)
        all_poi_embeds = self.poi_embed.weight[:-1]  # [n, d+1]
        all_cat_embeds = self.cat_embed.weight[:-1]
        all_geo_embeds = self.geo_embed.weight[:-1]
        beta = torch.sigmoid(self.logit_beta)
        output_poi = beta * self.poi_decoder(tangent_embeds) + (1 - beta) * self._calculate_hyperbolic_distance_score(
            traj_embeds, all_poi_embeds)
        output_cat = beta * self.cat_decoder(tangent_embeds) + (1 - beta) * self._calculate_hyperbolic_distance_score(
            traj_embeds, all_cat_embeds)
        output_geo = beta * self.geo_decoder(tangent_embeds) + (1 - beta) * self._calculate_hyperbolic_distance_score(
            traj_embeds, all_geo_embeds)

        return output_poi, output_cat, output_geo


    def _calculate_hyperbolic_distance_score(self, x_lorentz: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, L, H = x_lorentz.shape
        N, H2 = y.shape
        x_lorentz = self.Lorentz.projx(x_lorentz)
        y = self.Lorentz.projx(y)
        uv = self.Lorentz.inner_matmul(x_lorentz.view(-1, H), y.transpose(0, 1))  # [BL, N]
        dist_sq = (-2.0 * self.c - 2.0 * uv.view(B, L, N))
        dist_sq = torch.clamp_min(dist_sq, 1e-8)  # [B,L,N]
        scores = torch.exp(-torch.sqrt(dist_sq) / torch.clamp(self.dist_tau, 1e-3))
        return scores
