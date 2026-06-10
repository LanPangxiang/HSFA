import torch
from manifolds.utils import acosh, sqrt, clamp


EXP_MAX_NORM = 10.


def inner(u, v, *, keepdim=False, dim=-1):
    r"""
    Minkowski inner product.

    .. math::
        \langle\mathbf{u}, \mathbf{v}\rangle_{\mathcal{L}}:=-u_{0} v_{0}+u_{1} v_{1}+\ldots+u_{d} v_{d}

    Parameters
    ----------
    u : tensor
        vector in ambient space
    v : tensor
        vector in ambient space
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(u, v, keepdim=keepdim, dim=dim)


def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        # return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
        #     dim=dim, keepdim=True
        # )
        return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(
            dim=dim, keepdim=True
        )


def _inner_matmul(x, y, keepdim: bool = False):

    x_time = x[:, 0:1]  # [BL, 1]
    x_space = x[:, 1:]  # [BL, H-1]
    
    y_time = y[0:1, :]  # [1, N]
    y_space = y[1:, :]  # [H-1, N]
    
    # -x_time * y_time + x_space @ y_space
    time_part = -x_time @ y_time  # [BL, 1] @ [1, N] = [BL, N]
    space_part = x_space @ y_space  # [BL, H-1] @ [H-1, N] = [BL, N]
    
    result = time_part + space_part  # [BL, N]
    
    if keepdim:
        return result.unsqueeze(-1)  # [BL, N, 1]
    else:
        return result  # [BL, N]


def inner0(v, *, k, keepdim=False, dim=-1):
    r"""
    Minkowski inner product with zero vector.

    Parameters
    ----------
    v : tensor
        vector in ambient space
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner0(v, k=k, keepdim=keepdim, dim=dim)


def _inner0(v, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    res = -v.narrow(dim, 0, 1)
    if keepdim is False:
        res = res.squeeze(dim)
    return res


def dist(x, y, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid.

    .. math::

        d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})=\sqrt{k} \operatorname{arcosh}\left(-\frac{\langle\mathbf{x}, \mathbf{y}\rangle_{\mathcal{L}}}{k}\right)

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    y : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim)


def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner(x, y, dim=dim, keepdim=keepdim)
    return acosh(d / k)


def dist0(x, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid to zero point.

    .. math::

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and zero point
    """
    return _dist0(x, k=k, keepdim=keepdim, dim=dim)


def _dist0(x, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner0(x, k=k, dim=dim, keepdim=keepdim)
    return acosh(d / k)


def cdist(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor):
    # tmp = torch.ones(x.shape[-1], device=x.device)
    # tmp[0] = -1
    x = x.clone()
    x.narrow(-1, 0, 1).mul_(-1)
    return acosh(-(x @ y.transpose(-1, -2)))


def project(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{k+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k=k, dim=dim)


@torch.jit.script
def _project(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    right_ = x.narrow(dim, 1, dn)
    left_ = torch.sqrt(
        k + (right_ * right_).sum(dim=dim, keepdim=True)
    )
    x = torch.cat((left_, right_), dim=dim)
    return x


def project_polar(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid from polar coordinates.

    ... math::
        \pi((\mathbf{d}, r))=(\sqrt{k} \sinh (r/\sqrt{k}) \mathbf{d}, \cosh (r / \sqrt{k}))

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_polar(x, k=k, dim=dim)


def _project_polar(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    d = x.narrow(dim, 0, dn)
    r = x.narrow(dim, -1, 1)
    res = torch.cat(
        (
            torch.cosh(r / torch.sqrt(k)),
            torch.sqrt(k) * torch.sinh(r / torch.sqrt(k)) * d,
        ),
        dim=dim,
    )
    return res


def project_u(x, v, *, k, dim=-1):
    r"""
    Projection of the vector on the tangent space.

    ... math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, 1}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \mathbf{x} / k

    Parameters
    ----------
    x: tensor
        point on the Hyperboloid
    v: tensor
        vector in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_u(x, v, k=k, dim=dim)


def _project_u(x, v, k: torch.Tensor, dim: int = -1):
    return v.addcmul(_inner(x, v, dim=dim, keepdim=True), x / k)


def project_u0(u):
    narrowed = u.narrow(-1, 0, 1)
    vals = torch.zeros_like(u)
    vals[..., 0:1] = narrowed
    return u - vals


def norm(u, *, keepdim=False, dim=-1):
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Hyperboloid.

    .. math::

        \|\mathbf{v}\|_{\mathcal{L}}=\sqrt{\langle\mathbf{v}, \mathbf{v}\rangle_{\mathcal{L}}}

    Parameters
    ----------
    u : tensor
        tangent vector on Hyperboloid
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    """
    return _norm(u, keepdim=keepdim, dim=dim)


def _norm(u, keepdim: bool = False, dim: int = -1):
    return sqrt(_inner(u, u, keepdim=keepdim))


def expmap(x, u, *, k, dim=-1):
    r"""
    Compute exponential map on the Hyperboloid.

    .. math::

        \exp _{\mathbf{x}}^{k}(\mathbf{v})=\cosh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}


    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    u : tensor
        unit speed vector on Hyperboloid
    k: tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, k=k, dim=dim)


def _expmap(x, u, k: torch.Tensor, dim: int = -1):
    # nomin = (_norm(u, keepdim=True, dim=dim) / torch.sqrt(k)).clamp_max(10.)
    nomin = (_norm(u, keepdim=True, dim=dim))
    u = u / nomin
    nomin = nomin.clamp_max(EXP_MAX_NORM)
    # mask = nomin.lt(EXP_MAX_NORM)
    # if (~mask).any():
    #     nomin_mask = nomin.masked_scatter(mask, torch.ones_like(nomin))
    #     u = u / nomin_mask
    #     nomin = (_norm(u, keepdim=True, dim=dim))
    p = torch.cosh(nomin) * x + torch.sinh(nomin) * u
    return p


def expmap0(u, *, k, dim=-1):
    r"""
    Compute exponential map for Hyperboloid from :math:`0`.

    Parameters
    ----------
    u : tensor
        speed vector on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, k, dim=dim)

'''
def _expmap0(u, k: torch.Tensor, dim: int = -1):
    # 数值稳定：近零与上界保护
    u_norm = _norm(u, keepdim=True, dim=dim)
    # 近零直接返回原点，避免 0/0
    small_mask = u_norm < 1e-8
    if small_mask.any():
        zero_point = torch.zeros_like(u)
        zero_point[..., 0] = torch.sqrt(k)
        # 对小范数位置返回原点，其余位置按稳定分支计算
        u_norm_safe = torch.where(small_mask, torch.ones_like(u_norm), u_norm)
    else:
        u_norm_safe = u_norm

    u_unit = u / u_norm_safe
    u_norm_clamped = u_norm_safe.clamp_max(EXP_MAX_NORM)
    l_v = torch.cosh(u_norm_clamped)
    r_v = torch.sinh(u_norm_clamped) * u_unit
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    if small_mask.any():
        p = torch.where(small_mask.expand_as(p), zero_point, p)
    return p
'''
def _expmap0(u, k: torch.Tensor, dim: int = -1):
    # nomin = (_norm(u, keepdim=True, dim=dim) / torch.sqrt(k)).clamp_max(10.)
    nomin = (_norm(u, keepdim=True, dim=dim))
    u = u / nomin
    nomin = nomin.clamp_max(EXP_MAX_NORM)
    # mask = nomin.lt(EXP_MAX_NORM)
    # if (~mask).any():
    #     nomin_mask = nomin.masked_scatter(mask, torch.ones_like(nomin))
    #     u = u / nomin_mask
    #     nomin = (_norm(u, keepdim=True, dim=dim))
    l_v = torch.cosh(nomin)
    r_v = torch.sinh(nomin) * u
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p




def logmap(x, y, *, k, dim=-1):
    r"""
    Compute logarithmic map for two points :math:`x` and :math:`y` on the manifold.

    .. math::

        \log _{\mathbf{x}}^{k}(\mathbf{y})=d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})
            \frac{\mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}}{\left\|
            \mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}\right\|_{\mathcal{L}}}

    The result of Logarithmic map is a vector such that

    .. math::

        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))


    Parameters
    ----------
    x : tensor
        starting point on Hyperboloid
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, k=k, dim=dim)


def _logmap(x, y, k, dim: int = -1):
    dist_ = _dist(x, y, k=k, dim=dim, keepdim=True)
    nomin = y + 1.0 / k * _inner(x, y, keepdim=True) * x
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom
    # alpha = -1 / k.sqrt() * _inner(x, y, k, keepdim=True)             # 没用到，不确定是不是对的
    # nom = acosh(alpha)
    # denom = (alpha * alpha - 1).sqrt()
    # return nom / denom * (y - alpha * x)


def logmap0(y, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`y` from :math:`0` on the manifold.

    Parameters
    ----------
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k=k, dim=dim)

'''
def _logmap0(y, k, dim: int = -1):
    alpha = -_inner0(y, k, keepdim=True)
    eps = 1e-8
    alpha = torch.clamp_min(alpha, 1.0 + eps)
    zero_point = torch.zeros_like(y)
    zero_point[..., 0] = 1.0
    direction = y - alpha * zero_point
    denom = torch.sqrt(torch.clamp_min(alpha * alpha - 1.0, eps))
    dir_norm = _norm(direction, keepdim=True)
    small = dir_norm < 1e-8
    result = acosh(alpha) / denom * direction
    result = torch.where(small.expand_as(result), torch.zeros_like(result), result)
    return result
'''

def _logmap0(y, k, dim: int = -1):
    # dist_ = _dist0(y, k=k, dim=dim, keepdim=True)
    # nomin_ = 1.0 / k * _inner0(y, k=k, keepdim=True) * torch.sqrt(k)
    # dn = y.size(dim) - 1
    # nomin = torch.cat((nomin_ + y.narrow(dim, 0, 1),
    #                    y.narrow(dim, 1, dn)), dim)
    # denom = _norm(nomin, keepdim=True)
    # return dist_ * nomin / denom
    alpha = -_inner0(y, k, keepdim=True)
    zero_point = torch.zeros(y.shape[-1], device=y.device)
    zero_point[0] = 1
    return acosh(alpha) / torch.sqrt(alpha * alpha - 1) * (y - alpha * zero_point)



def logmap0back(x, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`0` from :math:`x` on the manifold.

    Parameters
    ----------
    x : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0back(x, k=k, dim=dim)


def _logmap0back(x, k, dim: int = -1):
    dist_ = _dist0(x, k=k, dim=dim, keepdim=True)
    nomin_ = 1.0 / k * _inner0(x, k=k, keepdim=True) * x
    dn = nomin_.size(dim) - 1
    nomin = torch.cat(
        (nomin_.narrow(dim, 0, 1) + 1, nomin_.narrow(dim, 1, dn)), dim
    )
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom
    # y = torch.zeros(x.shape[-1], device=x.device)
    # y[0] = k.sqrt()
    # return _logmap(x, y, k, dim)


def egrad2rgrad(x, grad, *, k, dim=-1):
    r"""
    Translate Euclidean gradient to Riemannian gradient on tangent space of :math:`x`.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, k}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \frac{\mathbf{x}}{k}

    Parameters
    ----------
    x : tensor
        point on the Hyperboloid
    grad : tensor
        Euclidean gradient for :math:`x`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in `
    """
    return _egrad2rgrad(x, grad, k=k, dim=dim)


def _egrad2rgrad(x, grad, k, dim: int = -1):
    grad.narrow(-1, 0, 1).mul_(-1)
    grad = grad.addcmul(_inner(x, grad, dim=dim, keepdim=True), x / k)
    return grad


def parallel_transport(x, y, v, *, k, dim=-1):
    r"""
    Perform parallel transport on the Hyperboloid.

    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport(x, y, v, k=k, dim=dim)


def _parallel_transport(x, y, v, k, dim: int = -1):
    # lmap = _logmap(x, y, k=k, dim=dim)
    # nom = _inner(lmap, v, keepdim=True)
    # denom = _dist(x, y, k=k, dim=dim, keepdim=True) ** 2
    # p = v - nom / denom * (lmap + _logmap(y, x, k=k, dim=dim))
    # return p
    nom = _inner(y, v, keepdim=True)
    denom = torch.clamp_min(k - _inner(x, y, keepdim=True), 1e-7)
    # return v + nom / denom * (x + y)
    return v.addcmul(nom / denom, x + y)


def parallel_transport0(y, v, *, k, dim=-1):
    r"""
    Perform parallel transport from zero point.

    Parameters
    ----------
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport0(y, v, k=k, dim=dim)


def _parallel_transport0(y, v, k, dim: int = -1):
    # lmap = _logmap0(y, k=k, dim=dim)
    # nom = _inner(lmap, v, keepdim=True)
    # denom = _dist0(y, k=k, dim=dim, keepdim=True) ** 2
    # p = v - nom / denom * (lmap + _logmap0back(y, k=k, dim=dim))
    # return p
    nom = _inner(y, v, keepdim=True)
    denom = torch.clamp_min(k - _inner0(y, k=k, keepdim=True), 1e-7)
    zero_point = torch.zeros_like(y)
    zero_point[..., 0] = 1
    # return v + nom / denom * (y + zero_point)
    return v.addcmul(nom / denom, y + zero_point)


def parallel_transport0back(x, v, *, k, dim: int = -1):
    r"""
    Perform parallel transport to the zero point.

    Special case parallel transport with last point at zero that
    can be computed more efficiently and numerically stable

    Parameters
    ----------
    x : tensor
        target point
    v : tensor
        vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0back(x, v, k=k, dim=dim)


def _parallel_transport0back(x, v, k, dim: int = -1):
    # lmap = _logmap0back(x, k=k, dim=dim)
    # nom = _inner(lmap, v, keepdim=True)
    # denom = _dist0(x, k=k, dim=dim, keepdim=True) ** 2
    # p = v - nom / denom * (lmap + _logmap0(x, k=k, dim=dim))
    # return p
    nom = _inner0(v, k=k, keepdim=True)
    denom = torch.clamp_min(k - _inner0(x, k=k, keepdim=True), 1e-7)
    zero_point = torch.zeros_like(x)
    zero_point[..., 0] = 1
    # return v + nom / denom * (x + zero_point)
    return v.addcmul(nom / denom, x + zero_point)


def geodesic_unit(t, x, u, *, k):
    r"""
    Compute unit speed geodesic at time :math:`t` starting from :math:`x` with direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{\mathbf{x} \rightarrow \mathbf{u}}^{k}(t)=\cosh \left(\frac{t}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{t}{\sqrt{k}}\right) \mathbf{u}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point
    u : tensor
        unit direction vector
    k : tensor
        manifold negative curvature

    Returns
    -------
    tensor
        the point on geodesic line
    """
    return _geodesic_unit(t, x, u, k=k)


def _geodesic_unit(t, x, u, k):
    return (
        torch.cosh(t) * x
        + torch.sinh(t) * u
    )


def lorentz_to_poincare(x, k, dim=-1):
    r"""
    Diffeomorphism that maps from Hyperboloid to Poincare disk.

    .. math::

        \Pi_{\mathbb{H}^{d, 1} \rightarrow \mathbb{D}^{d, 1}\left(x_{0}, \ldots, x_{d}\right)}=\frac{\left(x_{1}, \ldots, x_{d}\right)}{x_{0}+\sqrt{k}}

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Poincare disk
    """
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + 1)


def project_poincare(x, *, eps=1e-6, dim=-1):
    r"""
    Projection on the Poincare ball.

    .. math::

        \Pi_{\mathbb{R}^{d} \rightarrow \mathbb{D}^{d}}(\mathbf{x}):=\frac{\mathbf{x}}{\max(\|\mathbf{x}\|, 1-\epsilon)}

    Parameters
    ----------
    x: tensor
        point in Rn
    eps: float
        numerical stability constant
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the Poincare ball
    """
    return _project_poincare(x, eps=eps, dim=dim)


@torch.jit.script
def _project_poincare(x, eps: float = 1e-6, dim: int = -1):
    x_norm = torch.norm(x, dim=dim, keepdim=True)
    max_norm = 1.0 - eps
    scale = torch.where(x_norm >= max_norm, max_norm / (x_norm + eps), torch.ones_like(x_norm))
    return x * scale


def poincare_to_lorentz(x, k, dim=-1, eps=1e-6):
    r"""
    Diffeomorphism that maps from Poincare disk to Hyperboloid.

    .. math::

        \Pi_{\mathbb{D}^{d, k} \rightarrow \mathbb{H}^{d d, 1}}\left(x_{1}, \ldots, x_{d}\right)=\frac{\sqrt{k} \left(1+|| \mathbf{x}||_{2}^{2}, 2 x_{1}, \ldots, 2 x_{d}\right)}{1-\|\mathbf{x}\|_{2}^{2}}

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Hyperboloid
    """
    # 确保输入在庞加莱球内
    x = project_poincare(x, eps=eps, dim=dim)
    
    x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
    res = (
        torch.cat((1 + x_norm_square, 2 * x), dim=dim)
        / (1.0 - x_norm_square + eps)
    )
    return res
