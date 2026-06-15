"""
Python wrapper around the heat_core C++ extension.
Mirrors the notebook interface: pass the same arrays + grad_T function,
get back the same dict the notebook produces.
"""
import numpy as np
import heat_core


def run(
    Nx, Ny, Lx, Ly,
    K_BASE, RHO, CP,
    gamma_k, T0, T_init,
    dt, t_end,
    grad_T_fn,
    stretch=2.0,
    max_iter=50,
    tol=1e-6,
    n_plots=8,
):
    """
    Parameters
    ----------
    Nx, Ny          : grid resolution
    Lx, Ly          : domain size [m]
    K_BASE          : base conductivity array (Ny, Nx) [W/(m·K)]
    RHO, CP         : density (Ny,Nx) [kg/m³] and specific heat (Ny,Nx) [J/(kg·K)]
    gamma_k         : k(T) exponent [1/°C]
    T0              : reference and initial temperature [°C]
    T_init          : uniform initial temperature [°C]
    dt              : time step [s]
    t_end           : simulation end time [s]
    grad_T_fn       : callable f(t) → temperature gradient [°C/m]
    stretch         : log-mesh stretching factor in x
    max_iter, tol   : Picard iteration controls
    n_plots         : number of field snapshots

    Returns
    -------
    dict with keys matching the notebook output:
        time_array, q_array, picard_iters,
        plot_times, T_fields, k_fields,
        corner_T, T_final
    """
    # Build non-uniform x mesh (same as notebook)
    s      = np.linspace(0, 1, Nx + 1)
    xf     = Lx * (np.exp(stretch * s) - 1) / (np.exp(stretch) - 1)
    dx_arr = np.diff(xf).astype(np.float64)
    dy     = Ly / Ny

    # Pre-compute grad_T for every time step (avoids Python callbacks from C++)
    Nt           = int(t_end / dt)
    grad_T_vals  = np.array([grad_T_fn((n + 1) * dt) for n in range(Nt)], dtype=np.float64)

    return heat_core.simulate(
        Nx, Ny,
        dx_arr, dy,
        np.asarray(K_BASE, dtype=np.float64),
        np.asarray(RHO,    dtype=np.float64),
        np.asarray(CP,     dtype=np.float64),
        gamma_k, T0, float(T_init),
        dt, t_end,
        grad_T_vals,
        max_iter, tol, n_plots,
    )
