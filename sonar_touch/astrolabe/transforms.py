import numpy as np
from coorx import Transform


class SphericalTransform(Transform):
    """Converts Cartesian (x, y, z) to spherical (lon(rad), lat(rad), r)."""
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, dims=(3, 3), **kwargs):
        super().__init__(dims, **kwargs)

    def _map(self, coords):
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        lon = np.arctan2(y, x)
        lat = np.arcsin(z / r)

        ret = np.empty(coords.shape, dtype=coords.dtype)
        ret[..., 0] = lon
        ret[..., 1] = lat
        ret[..., 2] = r
        return ret

    def _imap(self, coords):
        lon, lat, r = coords[..., 0], coords[..., 1], coords[..., 2]
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        ret = np.empty(coords.shape, dtype=coords.dtype)
        ret[..., 0] = x
        ret[..., 1] = y
        ret[..., 2] = z
        return ret

    @property
    def params(self):
        return {}

    def set_params(self, **params):
        return


class MercatorSphericalTransform(Transform):
    """Maps (lon, lat, z) → (x, y, z) using Mercator projection (ignores r)."""
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, dims=(3, 3), **kwargs):
        super().__init__(dims, **kwargs)

    def _map(self, coords):
        lon = coords[..., 0]
        lat = np.clip(coords[..., 1], -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
        x = lon
        y = np.log(np.tan(np.pi / 4 + lat / 2))

        ret = np.empty_like(coords)
        ret[..., 0] = x
        ret[..., 1] = y
        ret[..., 2:] = coords[..., 2:]  # preserve extra axes like brightness
        return ret

    def _imap(self, coords):
        x = coords[..., 0]
        y = coords[..., 1]
        lon = x
        lat = 2 * np.arctan(np.exp(y)) - np.pi / 2

        ret = np.empty_like(coords)
        ret[..., 0] = lon
        ret[..., 1] = lat
        ret[..., 2:] = coords[..., 2:]  # preserve extra axes
        return ret

    @property
    def params(self):
        return {}

    def set_params(self, **params):
        return


class LambertAzimuthalEqualAreaTransform(Transform):
    """Projects (lon, lat, z) → (x, y, z) using Lambert Azimuthal Equal-Area projection."""
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False

    def __init__(self, dims=(3, 3), **kwargs):
        super().__init__(dims, **kwargs)

    def _map(self, coords):
        # Input in radians
        lon = coords[..., 0]
        lat = coords[..., 1]

        # Project relative to north pole (φ₀ = π/2, λ₀ = 0)
        k = np.sqrt(2 / (1 + np.sin(lat)))
        x = k * np.cos(lat) * np.sin(lon)
        y = -k * np.cos(lat) * np.cos(lon)  # y is negative so pole is on top

        ret = np.empty_like(coords)
        ret[..., 0] = x
        ret[..., 1] = y
        ret[..., 2:] = coords[..., 2:]  # preserve z or other axes
        return ret

    def _imap(self, coords):
        x = coords[..., 0]
        y = coords[..., 1]

        rho_sq = x**2 + y**2
        rho = np.sqrt(rho_sq)
        c = 2 * np.arcsin(np.minimum(rho / 2, 1.0))  # clip to avoid domain error

        sin_c = np.sin(c)
        cos_c = np.cos(c)

        # Inverse for center at φ₀ = π/2
        lat = np.arcsin(cos_c)
        lon = np.arctan2(x * sin_c, -y * sin_c)

        ret = np.empty_like(coords)
        ret[..., 0] = lon
        ret[..., 1] = lat
        ret[..., 2:] = coords[..., 2:]  # preserve z or other axes
        return ret

    @property
    def params(self):
        return {}

    def set_params(self, **params):
        return
