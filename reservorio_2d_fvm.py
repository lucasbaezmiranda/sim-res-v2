"""
Reservorio 2D — FVM + iteración de Picard
==========================================

Ecuación de difusión de presión (fluido ligeramente compresible en medio poroso):

    φ·cₜ·∂P/∂t = ∇·((k(P)/μ)·∇P)

Dividiendo por φ·cₜ queda la misma forma que el caso térmico:

    ∂P/∂t = ∇·(η(P)·∇P)      con   η(P) = k(P)/(μ·φ·cₜ)

Analogía térmica → reservorio:
    T       →  P       (temperatura → presión)
    α(T)    →  η(P)    (difusividad térmica → difusividad hidráulica)
    q_heat  →  q_darcy (flujo de calor → velocidad Darcy de producción)

Ley de permeabilidad:
    k(P) = k_i·exp(γ·(P − Pᵢ))
    γ < 0 : k cae al bajar P (compactación de poros al deplecionar)
    γ = 0 : permeabilidad constante (problema lineal)

Condiciones de borde:
    - Cara derecha, superior, inferior : no-flow  (∂P/∂n = 0)
    - Cara izquierda:
        mitad superior (j ≥ Ny//2) → no-flow
        mitad inferior (j <  Ny//2) → intervalo productor con caudal PWM
            ON  : extrae a velocidad Darcy q_prod [m/s]  →  P decrece
            OFF : cerrado (no-flow)                       →  P se recupera

Malla cell-centered. Indexación: celda (i,j) → k = j*Nx + i
    i : dirección x  (0=izquierda, Nx-1=derecha)
    j : dirección y  (0=abajo,     Ny-1=arriba )
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
# GEOMETRÍA
# ============================================================
Lx, Ly = 500.0, 500.0   # dimensiones del reservorio [m]
Nx, Ny  = 20, 20         # celdas en cada dirección

dt    = 3600.0           # paso de tiempo [s]  (1 hora)
t_end = 30 * 86400.0     # tiempo total   [s]  (30 días)

n_plots  = 8             # snapshots de presión a guardar
max_iter = 50            # máx iteraciones de Picard por paso de tiempo
tol      = 1e-6          # tolerancia de convergencia de Picard

# ============================================================
# PROPIEDADES DEL RESERVORIO
# ============================================================
Pi  = 20e6        # presión inicial uniforme        [Pa]   (200 bar)
k_i = 9.869e-14   # permeabilidad inicial           [m²]   (≈ 100 mD)
mu  = 1e-3        # viscosidad dinámica             [Pa·s] (agua, 1 cP)
por = 0.20        # porosidad                       [-]
ct  = 1e-9        # compresibilidad total           [Pa⁻¹]

# Difusividad hidráulica inicial: η_i = k_i / (μ·φ·cₜ)   [m²/s]
eta_i = k_i / (mu * por * ct)

# Sensibilidad de permeabilidad a la presión
gamma = -1e-8     # [Pa⁻¹]  (γ=0 → caso lineal)

def eta(P):
    """Difusividad hidráulica η(P) = η_i·exp(γ·(P − Pᵢ))  [m²/s]"""
    return eta_i * np.exp(gamma * (P - Pi))

# ============================================================
# PRODUCCIÓN — caudal PWM en la mitad inferior de la cara izquierda
# ============================================================
q_prod = 1e-6          # velocidad Darcy de extracción [m/s]
                       # Q [m³/(m·día)] = q_prod × intervalo_activo × 86400
T_pwm  = 5 * 86400.0  # período del ciclo PWM   [s]  (5 días)
duty   = 0.6           # fracción del período produciendo  (60 % ON, 40 % OFF)

def q_darcy(t):
    """Velocidad Darcy de extracción [m/s]. Positivo = sale del reservorio."""
    phase = (t % T_pwm) / T_pwm
    return q_prod if phase < duty else 0.0

def phi_bc(t):
    """
    Flujo normalizado para el FVM: φ_bc = η·∂P/∂n = −q_darcy/(φ·cₜ)
    Derivación:
        q_darcy_out = −(k/μ)·∂P/∂n_in
        η·∂P/∂n_in  = −q_darcy_out/(φ·cₜ)
    Se suma al RHS como: b[p] += phi_bc(t)·Δy
    """
    return -q_darcy(t) / (por * ct)

# ============================================================
# MALLA
# ============================================================
dx = Lx / Nx
dy = Ly / Ny
x  = np.linspace(dx/2, Lx - dx/2, Nx)
y  = np.linspace(dy/2, Ly - dy/2, Ny)
X, Y = np.meshgrid(x, y)

def idx(i, j):
    return j * Nx + i

N = Nx * Ny

# ============================================================
# ENSAMBLE — idéntico al caso térmico con α→η y T→P
#
# Media aritmética de η en cada cara:
#   η_face = 0.5·(η(P_P) + η(P_nb))
# Conductancia:
#   Este/Oeste: η_face·Δy/Δx
#   Norte/Sur:  η_face·Δx/Δy
# ============================================================
def ensamblar(P_iter, P_prev, t):
    """
    P_iter : P^k  — iterado actual (evalúa η)
    P_prev : P^n  — solución del paso anterior (término transiente del RHS)
    """
    A = lil_matrix((N, N))
    b = np.zeros(N)

    for j in range(Ny):
        for i in range(Nx):
            p   = idx(i, j)
            P_p = P_iter[p]
            e_p = eta(P_p)

            aP   = dx * dy / dt        # coef. transiente
            b[p] = aP * P_prev[p]

            # ---- Este ----
            if i < Nx - 1:
                ef   = 0.5 * (e_p + eta(P_iter[idx(i+1, j)]))
                coef = ef * dy / dx
                A[p, idx(i+1, j)] -= coef
                aP += coef

            # ---- Oeste ----
            if i > 0:
                ef   = 0.5 * (e_p + eta(P_iter[idx(i-1, j)]))
                coef = ef * dy / dx
                A[p, idx(i-1, j)] -= coef
                aP += coef
            elif j < Ny // 2:
                # Intervalo productor: flujo prescrito
                b[p] += phi_bc(t) * dy
            # else: mitad superior izquierda → no-flow

            # ---- Norte ----
            if j < Ny - 1:
                ef   = 0.5 * (e_p + eta(P_iter[idx(i, j+1)]))
                coef = ef * dx / dy
                A[p, idx(i, j+1)] -= coef
                aP += coef

            # ---- Sur ----
            if j > 0:
                ef   = 0.5 * (e_p + eta(P_iter[idx(i, j-1)]))
                coef = ef * dx / dy
                A[p, idx(i, j-1)] -= coef
                aP += coef

            A[p, p] = aP

    return csr_matrix(A), b

# ============================================================
# CONDICIÓN INICIAL
# ============================================================
P = np.full(N, Pi)

# ============================================================
# BUCLE DE TIEMPO
# ============================================================
Nt         = int(t_end / dt)
plot_every = max(1, Nt // n_plots)
plot_times  = []
plot_fields = []
picard_iters = []

# Índices de monitoreo
corner_idx = {
    'inf-izq (prod)': idx(0,    0   ),
    'inf-der'       : idx(Nx-1, 0   ),
    'sup-izq'       : idx(0,    Ny-1),
    'sup-der'       : idx(Nx-1, Ny-1),
}
corner_P   = {name: [] for name in corner_idx}
time_days  = []
Q_daily    = []    # caudal volumétrico [m²/día] por unidad de profundidad
Np_list    = []    # producción acumulada [m³/m] por unidad de profundidad
Np_cum     = 0.0

intervalo_activo = (Ny // 2) * dy   # longitud del intervalo productor [m]

for n in range(Nt):
    t      = (n + 1) * dt
    P_prev = P.copy()
    P_iter = P.copy()

    for k in range(max_iter):
        A_mat, b_vec = ensamblar(P_iter, P_prev, t)
        P_new = spsolve(A_mat, b_vec)

        res = np.linalg.norm(P_new - P_iter) / (np.linalg.norm(P_new) + 1e-15)
        P_iter = P_new

        if res < tol:
            break

    P = P_new
    picard_iters.append(k + 1)

    # Caudal y producción acumulada
    Q   = q_darcy(t) * intervalo_activo         # [m²/s] por unidad de profundidad
    Np_cum += Q * dt
    time_days.append(t / 86400)
    Q_daily.append(Q * 86400)                   # convertir a m²/día
    Np_list.append(Np_cum)

    for name, ci in corner_idx.items():
        corner_P[name].append(P[ci] / 1e6)      # Pa → MPa

    if (n + 1) % plot_every == 0:
        plot_times.append(t / 86400)
        plot_fields.append(P.reshape(Ny, Nx).copy() / 1e6)   # Pa → MPa

# ============================================================
# FIGURA 1 — campo de presión (snapshots)
# ============================================================
ncols = 4
nrows = (len(plot_times) + ncols - 1) // ncols
fig1, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
axes = np.array(axes).flatten()

vmin = min(f.min() for f in plot_fields)
vmax = Pi / 1e6   # presión inicial = máximo

for ax in axes:
    ax.set_visible(False)

for k_p, (t_p, P_grid) in enumerate(zip(plot_times, plot_fields)):
    ax = axes[k_p]
    ax.set_visible(True)
    c = ax.contourf(X / 1e3, Y / 1e3, P_grid, levels=40, cmap='viridis',
                    vmin=vmin, vmax=vmax)
    fig1.colorbar(c, ax=ax, label='P [MPa]')
    ax.set_title(f't = {t_p:.1f} días')
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.plot([0, 0], [0,             Ly/2/1e3], color='red',  lw=4, label='Productor')
    ax.plot([0, 0], [Ly/2/1e3, Ly/1e3      ], color='gray', lw=3, label='No-flow')
    if k_p == 0:
        ax.legend(fontsize=7)

plt.suptitle(f'Presión [MPa] — Reservorio 2D | k={k_i/9.869e-16:.0f} mD  φ={por}  cₜ={ct:.0e} Pa⁻¹', fontsize=11)
plt.tight_layout()
plt.savefig('reservorio_presion.png', dpi=150, bbox_inches='tight')

# ============================================================
# FIGURA 2 — series de tiempo: presión, caudal, Np
# ============================================================
t_arr = np.array(time_days)

fig2, axes2 = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                            gridspec_kw={'hspace': 0.35})

# Panel 1: presión en esquinas
colors_c = {'inf-izq (prod)': 'tab:red', 'inf-der': 'tab:orange',
            'sup-izq': 'tab:blue', 'sup-der': 'tab:cyan'}
for name, vals in corner_P.items():
    axes2[0].plot(t_arr, vals, label=name, color=colors_c[name])
axes2[0].axhline(Pi/1e6, color='k', lw=0.8, ls='--', label=f'Pᵢ = {Pi/1e6:.0f} MPa')
axes2[0].set_ylabel('P [MPa]')
axes2[0].set_title('Presión en las 4 esquinas del reservorio')
axes2[0].legend(fontsize=8)
axes2[0].grid(True, alpha=0.3)

# Panel 2: caudal de producción
axes2[1].fill_between(t_arr, Q_daily, color='steelblue', alpha=0.8)
axes2[1].set_ylabel('Q [m²/día]')
axes2[1].set_title(
    f'Caudal Darcy — PWM: período={T_pwm/86400:.0f} días, duty={duty*100:.0f}%'
    f'  (Q_total = q_prod × {intervalo_activo:.0f} m)')
axes2[1].grid(True, alpha=0.3)

# Panel 3: producción acumulada
axes2[2].plot(t_arr, Np_list, color='darkgreen', lw=2)
axes2[2].set_ylabel('Np [m³/m]')
axes2[2].set_xlabel('Tiempo [días]')
axes2[2].set_title('Producción acumulada (por metro de profundidad)')
axes2[2].grid(True, alpha=0.3)

plt.savefig('reservorio_series.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# RESUMEN
# ============================================================
print(f"\n{'='*50}")
print(f"η_i  = {eta_i:.4f} m²/s")
print(f"Tiempo de difusión L²/η = {Lx**2/eta_i/86400:.1f} días")
print(f"Intervalo productor     = {intervalo_activo:.0f} m")
print(f"{'='*50}")
print(f"P_min final = {P.min()/1e6:.3f} MPa  |  P_max final = {P.max()/1e6:.3f} MPa")
print(f"Np total    = {Np_cum:.2f} m³/m  ({Np_cum:.2f} m³ por metro de profundidad)")
print(f"Picard iter — max: {max(picard_iters)}, promedio: {np.mean(picard_iters):.1f}")
