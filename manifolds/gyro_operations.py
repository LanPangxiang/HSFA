import torch
import torch.nn as nn
import torch.nn.functional as F
import math as pymath


class GyroVectorOperations:
    @staticmethod
    def lorentz_inner(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        xy = x * y
        xy = xy.clone()
        xy[..., 0] = -xy[..., 0]
        return torch.sum(xy, dim=-1, keepdim=keepdim)
    
    @staticmethod
    def lorentz_distance_squared(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        inner_prod = GyroVectorOperations.lorentz_inner(x, y, keepdim=keepdim)
        return -2.0 - 2.0 * inner_prod
    
    @staticmethod
    def lorentz_distance(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        dist_sq = GyroVectorOperations.lorentz_distance_squared(x, y, keepdim=keepdim)
        return torch.sqrt(torch.clamp_min(dist_sq, 1e-8))
    
    @staticmethod
    def get_origin(shape: tuple, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
        origin = torch.zeros(shape, device=device, dtype=dtype)
        origin[..., 0] = 1.0
        return origin
    
    @staticmethod
    def project_to_lorentz(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        result = torch.zeros_like(x)
        result[..., 1:] = x[..., 1:]
        spatial_norm_sq = torch.sum(x[..., 1:] ** 2, dim=-1, keepdim=True)
        
        result[..., 0:1] = torch.sqrt(spatial_norm_sq + 1.0 + eps)
        
        return result
    
    @staticmethod
    def log_map_origin(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = GyroVectorOperations.project_to_lorentz(x, eps)
        
        spatial = x[..., 1:]
        time_comp = x[..., 0:1]

        # acosh(t) = ln(t + sqrt(t^2 - 1))
        t_clamped = torch.clamp_min(time_comp, 1.0 + eps)
        sqrt_term = torch.sqrt(torch.clamp_min(t_clamped ** 2 - 1.0, eps))
        acosh_t = torch.log(t_clamped + sqrt_term)
        
        spatial_norm = torch.norm(spatial, dim=-1, keepdim=True)
        spatial_norm = torch.clamp_min(spatial_norm, eps)
        
        normalized_spatial = spatial / spatial_norm
        tangent_spatial = normalized_spatial * acosh_t
        
        tangent = torch.zeros_like(x)
        tangent[..., 1:] = tangent_spatial
        
        return tangent
    
    @staticmethod
    def exp_map_origin(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        v_spatial = v[..., 1:]
        v_norm = torch.norm(v_spatial, dim=-1, keepdim=True)

        v_norm = torch.clamp(v_norm, min=eps, max=10.0)
        
        cosh_norm = torch.cosh(v_norm)
        sinh_norm = torch.sinh(v_norm)

        result = torch.zeros_like(v)
        result[..., 0:1] = cosh_norm
        result[..., 1:] = v_spatial * (sinh_norm / v_norm)
        
        return result
