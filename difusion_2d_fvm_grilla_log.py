"""
Difusión de calor 2D — FVM + Picard  (propiedades por celda)
=============================================================

Ecuación:  ρ·cp·∂T/∂t = ∇·(k(x,y,T)·∇T)

k varía espacialmente (materiales distintos) Y con la temperatura:
    k(x,y,T) = K_BASE[j,i] · exp(γ · (T − T₀))
    γ < 0 → conduce peor cuando está más caliente

Esto hace que α = k/(ρ·cp) sea efectivamente una matriz que evoluciona
con el campo de temperatura — de ahí el array K_BASE en lugar de un
escalar.

Cara entre celdas P y nb → media ARMÓNICA de k (no aritmética):
    k_face = 2·k_P·k_nb / (k_P + k_nb)
La media armónica es la correcta en FVM para materiales heterogéneos:
surge de la solución exacta 1D de dos resistencias en serie.

Condiciones de borde:
  - Cara derecha, superior, inferior : aisladas  (flujo = 0)
  - Cara izquierda:
      mitad superior (j ≥ Ny//2) → aislada
      mitad inferior (j <  Ny//2) → extracción de calor con PWM
          ON  : flujo q_on [W/m²] saliendo  (q_on > 0 → b[p] -= q_on·Δy)
          OFF : aislada

Malla cell-centered. Indexación: celda (i,j) → k = j*Nx + i
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
# GEOMETRÍA Y TIEMPO
# ============================================================
Lx, Ly  = 1.0, 1.0     # [m]
Nx, Ny  = 30, 30        # celdas

dt      = 2.0           # [s]
t_end   = 800.0         # [s]

n_plots  = 8
max_iter = 50
tol      = 1e-6

# ============================================================
# TEMPERATURA INICIAL / REFERENCIA
# ============================================================
T0 = 100.0              # temperatura inicial uniforme [°C]

# ============================================================
# PROPIEDADES POR CELDA  (arrays Ny × Nx)
#
# K_BASE[j,i] : conductividad de referencia en T = T₀  [W/(m·K)]
# RHO[j,i]    : densidad                               [kg/m³]
# CP[j,i]     : calor específico                       [J/(kg·K)]
#
# Para definir regiones con material distinto, asignar bloques:
#   K_BASE[:, Nx//2:] = 2.0   →  mitad derecha menos conductora
# ============================================================
K_BASE = np.full((Ny, Nx), 10.0)    # material único por defecto
RHO    = np.full((Ny, Nx), 800.0)
CP     = np.full((Ny, Nx), 400.0)

# Ejemplo de dominio heterogéneo (descomentar para activar):
K_BASE[:, Nx//2:] = 1.5              # mitad derecha: peor conductor

# ============================================================
# DEPENDENCIA DE k CON T
# γ < 0 → k decrece con T  (conduce peor cuando está caliente)
# ============================================================
gamma_k = -0.1        # [1/°C]

def k_cell(T, k_b):
    """k(T) = k_b · exp(γ · (T − T₀))"""
    return k_b * np.exp(gamma_k * (T - T0))

def k_harm(k1, k2):
    """Media armónica para la cara entre dos celdas."""
    return 2.0 * k1 * k2 / (k1 + k2 + 1e-30)

# ============================================================
# CONDICIÓN DE BORDE — extracción de calor PWM con amplitud variable
# ============================================================
T_pwm  = 60.0           # período del ciclo PWM  [s]
duty   = 0.5            # fracción del período encendido  (50%)

def q_amplitude(t):
    """
    Amplitud del flujo de calor cuando el PWM está ON  [W/m²].
    Modificar esta función para q variable en el tiempo.

    Ejemplos:
      return 4000.0                                      # constante
      return 4000.0 * (1 - np.exp(-t / 150))            # rampa exponencial
      return 2000.0 + 2000.0 * np.sin(np.pi*t / t_end) # senoidal
      return np.interp(t, [0,200,400,800], [0,5000,2000,4000])  # tabla
    """
    return 4000.0        # constante por defecto

def q_bc(t):
    """Flujo extraído [W/m²]. Combina amplitud variable con lógica PWM."""
    phase = (t % T_pwm) / T_pwm
    return q_amplitude(t) if phase < duty else 0.0

# ============================================================
# MALLA
# x : gradada (exponencial) — celdas finas en x=0, gruesas en x=Lx
# y : uniforme
# ============================================================
stretch = 3.0      # factor de estiramiento; mayor → más concentración cerca de x=0
                   # stretch=0 → uniforme; stretch=3 → celda derecha ≈18× más ancha que izquierda
s   = np.linspace(0, 1, Nx + 1)
xf  = Lx * (np.exp(stretch * s) - 1) / (np.exp(stretch) - 1)  # posiciones de caras en x
xc  = 0.5 * (xf[:-1] + xf[1:])                                 # centros de celda en x
dx_arr = np.diff(xf)                                            # anchos de celda (no uniformes)

dy = Ly / Ny
y  = np.linspace(dy/2, Ly - dy/2, Ny)

X, Y = np.meshgrid(xc, y)

def idx(i, j):
    return j * Nx + i

N = Nx * Ny

# ============================================================
# ENSAMBLE
# Transiente : ρ·cp·dx_i·dy/Δt   (dx_i varía por columna)
# Caras en x : resistencias en serie exactas para malla no uniforme:
#                coef = dy / ( dx_i/(2·k_P) + dx_{nb}/(2·k_nb) )
#              Para malla uniforme esto reduce a k_harm·dy/dx.
# Caras en y : malla uniforme → coef = k_harm(k_P,k_nb)·dx_i/dy
# BC calor   : b[p] -= q_bc(t)·dy   (resta → enfría)
# ============================================================
def ensamblar(T_iter, T_prev, t):
    A = lil_matrix((N, N))
    b = np.zeros(N)

    for j in range(Ny):
        for i in range(Nx):
            p    = idx(i, j)
            T_p  = T_iter[p]
            k_p  = k_cell(T_p, K_BASE[j, i])
            dx_i = dx_arr[i]

            aP   = RHO[j, i] * CP[j, i] * dx_i * dy / dt
            b[p] = aP * T_prev[p]

            # ---- Este ----
            if i < Nx - 1:
                k_e  = k_cell(T_iter[idx(i+1, j)], K_BASE[j, i+1])
                coef = dy / (dx_i/(2*k_p) + dx_arr[i+1]/(2*k_e))
                A[p, idx(i+1, j)] -= coef
                aP += coef

            # ---- Oeste ----
            if i > 0:
                k_w  = k_cell(T_iter[idx(i-1, j)], K_BASE[j, i-1])
                coef = dy / (dx_arr[i-1]/(2*k_w) + dx_i/(2*k_p))
                A[p, idx(i-1, j)] -= coef
                aP += coef
            elif j < Ny // 2:
                b[p] -= q_bc(t) * dy

            # ---- Norte ---- (y uniforme → k_harm · dx_i/dy)
            if j < Ny - 1:
                k_n  = k_cell(T_iter[idx(i, j+1)], K_BASE[j+1, i])
                coef = k_harm(k_p, k_n) * dx_i / dy
                A[p, idx(i, j+1)] -= coef
                aP += coef

            # ---- Sur ----
            if j > 0:
                k_s  = k_cell(T_iter[idx(i, j-1)], K_BASE[j-1, i])
                coef = k_harm(k_p, k_s) * dx_i / dy
                A[p, idx(i, j-1)] -= coef
                aP += coef

            A[p, p] = aP

    return csr_matrix(A), b

# ============================================================
# CONDICIÓN INICIAL
# ============================================================
T = np.full(N, T0)

# ============================================================
# BUCLE DE TIEMPO
# ============================================================
Nt         = int(t_end / dt)
plot_every = max(1, Nt // n_plots)
plot_times  = []
plot_fields = []
k_fields    = []        # campo de k efectiva para visualizar
picard_iters = []

corner_idx = {
    'inf-izq': idx(0,    0   ),
    'inf-der': idx(Nx-1, 0   ),
    'sup-izq': idx(0,    Ny-1),
    'sup-der': idx(Nx-1, Ny-1),
}
corner_T   = {name: [] for name in corner_idx}
time_array = []

for n in range(Nt):
    t      = (n + 1) * dt
    T_prev = T.copy()
    T_iter = T.copy()

    for k in range(max_iter):
        A_mat, b_vec = ensamblar(T_iter, T_prev, t)
        T_new = spsolve(A_mat, b_vec)

        res = np.linalg.norm(T_new - T_iter) / (np.linalg.norm(T_new) + 1e-15)
        T_iter = T_new
        if res < tol:
            break

    T = T_new
    picard_iters.append(k + 1)

    time_array.append(t)
    for name, ci in corner_idx.items():
        corner_T[name].append(T[ci])

    if (n + 1) % plot_every == 0:
        plot_times.append(t)
        plot_fields.append(T.reshape(Ny, Nx).copy())
        # campo de k efectiva: k_cell(T[j,i], K_BASE[j,i])
        T_grid = T.reshape(Ny, Nx)
        k_eff  = np.vectorize(k_cell)(T_grid, K_BASE)
        k_fields.append(k_eff.copy())

# ============================================================
# FIGURA 1 — campo de temperatura (snapshots)
# ============================================================
ncols = 4
nrows = (len(plot_times) + ncols - 1) // ncols
fig1, axes = plt.subplots(nrows, ncols, figsize=(4*ncols + 1, 4*nrows))
axes = np.array(axes).flatten()

vmin = min(f.min() for f in plot_fields)   # mínimo global → fijo para todas
vmax = T0                                   # máximo = condición inicial

for ax in axes:
    ax.set_visible(False)

for k_p, (t_p, T_grid) in enumerate(zip(plot_times, plot_fields)):
    ax = axes[k_p]
    ax.set_visible(True)
    c = ax.contourf(X, Y, T_grid, levels=40, cmap='plasma', vmin=vmin, vmax=vmax)
    ax.set_title(f't = {t_p:.0f} s')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axvline(Lx/2, color='white', lw=1, ls='--', alpha=0.5)
    ax.plot([0, 0], [0,    Ly/2], color='cyan', lw=4,
            label=f'q={q_bc(t_p):.0f} W/m²')
    ax.plot([0, 0], [Ly/2, Ly  ], color='lightgray', lw=3, label='Aislada')
    if k_p == 0:
        ax.legend(fontsize=7)

# Colorbar compartida — misma escala en todos los subplots
fig1.colorbar(c, ax=axes.tolist(), label='T [°C]', shrink=0.6, pad=0.02)

plt.suptitle(f'Temperatura [°C] — k(T)=K_BASE·exp(γ·(T−T₀)),  γ={gamma_k}/°C', fontsize=11)
plt.tight_layout()
plt.savefig('difusion_temperatura.png', dpi=150, bbox_inches='tight')

# ============================================================
# FIGURA 2 — campo de k efectiva (snapshots)
# Muestra la "matriz de α" evolucionando: k baja donde T bajó más
# ============================================================
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
axes2 = np.array(axes2).flatten()

kmin = min(f.min() for f in k_fields)
kmax = K_BASE.max()   # máximo = valor inicial (T=T0, factor=1)

for ax in axes2:
    ax.set_visible(False)

for k_p, (t_p, k_eff) in enumerate(zip(plot_times, k_fields)):
    ax = axes2[k_p]
    ax.set_visible(True)
    c = ax.contourf(X, Y, k_eff, levels=40, cmap='coolwarm', vmin=kmin, vmax=kmax)
    fig2.colorbar(c, ax=ax, label='k [W/(m·K)]')
    ax.set_title(f't = {t_p:.0f} s')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axvline(Lx/2, color='white', lw=1, ls='--', alpha=0.7)

plt.suptitle('Conductividad efectiva k(T) [W/(m·K)] — evoluciona con T', fontsize=11)
plt.tight_layout()
plt.savefig('difusion_k_efectiva.png', dpi=150, bbox_inches='tight')

# ============================================================
# FIGURA 3 — series temporales: T en esquinas + PWM
# ============================================================
t_arr = np.array(time_array)
fig3, (ax_T, ax_pwm) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

colors = {'inf-izq': 'tab:red', 'inf-der': 'tab:orange',
          'sup-izq': 'tab:blue', 'sup-der': 'tab:cyan'}
for name, vals in corner_T.items():
    ax_T.plot(t_arr, vals, label=name, color=colors[name])
ax_T.axhline(T0, color='k', lw=0.8, ls='--', label=f'T₀ = {T0:.0f} °C')
ax_T.set_ylabel('T [°C]')
ax_T.set_title('Temperatura en las 4 esquinas')
ax_T.legend(fontsize=8)
ax_T.grid(True, alpha=0.3)

pwm_vals = np.array([q_bc(t) for t in t_arr])
ax_pwm.fill_between(t_arr, pwm_vals, color='steelblue', alpha=0.7)
ax_pwm.set_ylabel('q [W/m²]')
ax_pwm.set_xlabel('Tiempo [s]')
ax_pwm.set_title(f'PWM — período={T_pwm} s, duty={duty*100:.0f}%')
ax_pwm.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('difusion_esquinas.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"T_max = {T.max():.2f} °C  |  T_min = {T.min():.2f} °C")
print(f"k efectiva mín = {k_fields[-1].min():.3f}  máx = {k_fields[-1].max():.3f}  W/(m·K)")
print(f"Picard iter/paso — max: {max(picard_iters)}, promedio: {np.mean(picard_iters):.1f}")
print(f"\nMalla en x (stretch={stretch}):")
print(f"  dx mínimo (cara izq) = {dx_arr[0]*1e3:.2f} mm")
print(f"  dx máximo (cara der) = {dx_arr[-1]*1e3:.1f} mm")
print(f"  razón dx_max/dx_min  = {dx_arr[-1]/dx_arr[0]:.1f}×")
