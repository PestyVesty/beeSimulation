import numpy as np

def polytunnel_temperature(x_local, y_local, L, W,
                           T_base=22.0, S=1.0,
                           delta_x=3.0, alpha=2,
                           delta_y=1.0):
    """
    Temperature for a tunnel of length L and width W, defined on local coords:
      x_local in [0, L], y_local in [0, W]
    Two open ends: cooler at x=0 and x=L, warmest near midpoint.
    """
    axial   = delta_x * (np.sin(np.pi * x_local / L) ** alpha)
    lateral = delta_y * (1.0 - np.cos(2.0 * np.pi * y_local / W))
    return T_base + S * (axial + lateral)


class TemperatureField:
    """
    Regular grid + bilinear interpolation over a tunnel embedded in world coords.
    World-to-local mapping: x_local = x_world - x0
      - Tunnel spans [x0, x0+L] x [0, W]
      - Outside policy:
          'none'    -> return None outside tunnel
          'ambient' -> return T_base outside tunnel
          'clamp'   -> clamp to nearest edge value
    """
    def __init__(self,
                 x0,              # world x where tunnel starts (e.g., 10.0)
                 L,               # tunnel length in meters (e.g., 100.0)
                 W,               # tunnel width in meters (use model.space_width)
                 Nx=200, Ny=60,
                 T_base=22.0, S=1.0, delta_x=3.0, alpha=2, delta_y=1.0,
                 outside_policy='ambient'):
        self.x0 = float(x0)
        self.L  = float(L)
        self.W  = float(W)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.outside = outside_policy
        self.params = dict(T_base=T_base, S=S, delta_x=delta_x, alpha=alpha, delta_y=delta_y)

        # local grid (0..L, 0..W)
        self.xg = np.linspace(0.0, self.L, self.Nx)
        self.yg = np.linspace(0.0, self.W, self.Ny)
        X, Y = np.meshgrid(self.xg, self.yg)
        self.T = polytunnel_temperature(X, Y, self.L, self.W, **self.params).astype(np.float32)

    def _bilinear(self, x_local, y_local):
        # indices
        ix1 = np.searchsorted(self.xg, x_local, side="right") - 1
        iy1 = np.searchsorted(self.yg, y_local, side="right") - 1
        ix1 = max(0, min(ix1, self.Nx - 2))
        iy1 = max(0, min(iy1, self.Ny - 2))
        ix2, iy2 = ix1 + 1, iy1 + 1

        x1, x2 = self.xg[ix1], self.xg[ix2]
        y1, y2 = self.yg[iy1], self.yg[iy2]
        tx = 0.0 if x2 == x1 else (x_local - x1) / (x2 - x1)
        ty = 0.0 if y2 == y1 else (y_local - y1) / (y2 - y1)

        Q11 = self.T[iy1, ix1]; Q21 = self.T[iy1, ix2]
        Q12 = self.T[iy2, ix1]; Q22 = self.T[iy2, ix2]
        return float((1-tx)*(1-ty)*Q11 + tx*(1-ty)*Q21 + (1-tx)*ty*Q12 + tx*ty*Q22)

    def at_world(self, x_world, y_world):
        # map world -> local
        x_local = x_world - self.x0
        y_local = y_world

        # inside?
        inside_x = (0.0 <= x_local <= self.L)
        inside_y = (0.0 <= y_local <= self.W)

        if inside_x and inside_y:
            return self._bilinear(x_local, y_local)

        # outside handling
        if self.outside == 'none':
            return None
        if self.outside == 'ambient':
            return self.params['T_base']
        
        xc = min(max(x_local, 0.0), self.L)
        yc = min(max(y_local, 0.0), self.W)
        return self._bilinear(xc, yc)