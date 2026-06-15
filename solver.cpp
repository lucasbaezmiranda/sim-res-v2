#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

using SpMat   = Eigen::SparseMatrix<double>;
using VecXd   = Eigen::VectorXd;
using Triplet = Eigen::Triplet<double>;

// ---------------------------------------------------------------------------
// Physics helpers
// ---------------------------------------------------------------------------

static inline double k_cell(double T, double k_b, double gamma_k, double T0) {
    double arg = gamma_k * (T - T0);
    if (arg >  50.0) arg =  50.0;
    if (arg < -50.0) arg = -50.0;
    return k_b * std::exp(arg);
}

static inline double k_harm(double k1, double k2) {
    return 2.0 * k1 * k2 / (k1 + k2 + 1e-30);
}

// ---------------------------------------------------------------------------
// FVM assembly  (mirrors ensamblar() in the Python notebook)
// ---------------------------------------------------------------------------

static void assemble(
    const VecXd&              T_iter,
    const VecXd&              T_prev,
    double                    grad_T_val,
    int Nx, int Ny,
    const std::vector<double>& dx,
    double dy,
    const std::vector<double>& K_flat,
    const std::vector<double>& RHO_flat,
    const std::vector<double>& CP_flat,
    double gamma_k, double T0, double dt,
    SpMat& A, VecXd& b
) {
    const int N        = Nx * Ny;
    const int n_active = Ny / 2;   // j < Ny//2 → active BC face

    std::vector<Triplet> trips;
    trips.reserve(5 * N);
    b.setZero(N);

    // k_prom on the active face (i=0, j < Ny/2), from current Picard iterate
    double k_prom = 0.0;
    for (int j = 0; j < n_active; ++j)
        k_prom += k_cell(T_iter[j * Nx], K_flat[j * Nx], gamma_k, T0);
    k_prom /= n_active;
    const double q_bc = grad_T_val * k_prom;   // [°C/m] · [W/(m·K)] = [W/m²]

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const int    p    = j * Nx + i;
            const double T_p  = T_iter[p];
            const double k_p  = k_cell(T_p, K_flat[p], gamma_k, T0);
            const double dx_i = dx[i];

            double aP = RHO_flat[p] * CP_flat[p] * dx_i * dy / dt;
            b[p]      = aP * T_prev[p];

            // East
            if (i < Nx - 1) {
                const int    pe   = j * Nx + (i + 1);
                const double k_e  = k_cell(T_iter[pe], K_flat[pe], gamma_k, T0);
                const double coef = dy / (dx_i / (2.0 * k_p) + dx[i+1] / (2.0 * k_e));
                trips.push_back({p, pe, -coef});
                aP += coef;
            }

            // West
            if (i > 0) {
                const int    pw   = j * Nx + (i - 1);
                const double k_w  = k_cell(T_iter[pw], K_flat[pw], gamma_k, T0);
                const double coef = dy / (dx[i-1] / (2.0 * k_w) + dx_i / (2.0 * k_p));
                trips.push_back({p, pw, -coef});
                aP += coef;
            } else if (j < n_active) {
                b[p] -= q_bc * dy;   // heat extraction BC
            }
            // else: insulated (no-op)

            // North
            if (j < Ny - 1) {
                const int    pn   = (j + 1) * Nx + i;
                const double k_n  = k_cell(T_iter[pn], K_flat[pn], gamma_k, T0);
                const double coef = k_harm(k_p, k_n) * dx_i / dy;
                trips.push_back({p, pn, -coef});
                aP += coef;
            }

            // South
            if (j > 0) {
                const int    ps   = (j - 1) * Nx + i;
                const double k_s  = k_cell(T_iter[ps], K_flat[ps], gamma_k, T0);
                const double coef = k_harm(k_p, k_s) * dx_i / dy;
                trips.push_back({p, ps, -coef});
                aP += coef;
            }

            trips.push_back({p, p, aP});
        }
    }

    A.setFromTriplets(trips.begin(), trips.end());
}

// ---------------------------------------------------------------------------
// Main simulation entry point
// ---------------------------------------------------------------------------

py::dict simulate(
    int Nx, int Ny,
    py::array_t<double, py::array::c_style | py::array::forcecast> dx_np,
    double dy,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_BASE_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> RHO_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> CP_np,
    double gamma_k,
    double T0,
    double T_init,
    double dt,
    double t_end,
    py::array_t<double, py::array::c_style | py::array::forcecast> grad_T_np,
    int    max_iter,
    double tol,
    int    n_plots
) {
    auto dx_r     = dx_np.unchecked<1>();
    auto K_r      = K_BASE_np.unchecked<2>();
    auto RHO_r    = RHO_np.unchecked<2>();
    auto CP_r     = CP_np.unchecked<2>();
    auto gradT_r  = grad_T_np.unchecked<1>();

    const int N          = Nx * Ny;
    const int Nt         = static_cast<int>(t_end / dt);
    const int plot_every = std::max(1, Nt / n_plots);

    // Flatten 2-D numpy arrays to contiguous C++ vectors
    std::vector<double> dx(Nx), K_flat(N), RHO_flat(N), CP_flat(N);
    for (int i = 0; i < Nx; ++i) dx[i] = dx_r(i);
    for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i) {
            const int p = j * Nx + i;
            K_flat[p]   = K_r(j, i);
            RHO_flat[p] = RHO_r(j, i);
            CP_flat[p]  = CP_r(j, i);
        }

    // Initial temperature
    VecXd T = VecXd::Constant(N, T_init);

    // Output containers
    std::vector<double> time_arr, q_arr;
    std::vector<int>    picard_arr;
    std::vector<double> c_inf_izq, c_inf_der, c_sup_izq, c_sup_der;
    std::vector<double> plot_times;
    std::vector<std::vector<double>> T_snaps, k_snaps;

    const int ci_ii = 0 + 0 * Nx;              // idx(0,    0   )
    const int ci_id = (Nx-1) + 0 * Nx;         // idx(Nx-1, 0   )
    const int ci_si = 0 + (Ny-1) * Nx;         // idx(0,    Ny-1)
    const int ci_sd = (Nx-1) + (Ny-1) * Nx;   // idx(Nx-1, Ny-1)

    SpMat A(N, N);
    Eigen::SparseLU<SpMat> lu;

    for (int n = 0; n < Nt; ++n) {
        const double t          = (n + 1) * dt;
        const double grad_T_val = gradT_r(n);

        VecXd T_prev = T;
        VecXd T_iter = T;
        VecXd T_new  = T;

        int picard_k = 0;
        for (int k = 0; k < max_iter; ++k) {
            picard_k = k;

            VecXd b;
            assemble(T_iter, T_prev, grad_T_val,
                     Nx, Ny, dx, dy, K_flat, RHO_flat, CP_flat,
                     gamma_k, T0, dt, A, b);

            lu.analyzePattern(A);
            lu.factorize(A);
            if (lu.info() != Eigen::Success)
                throw std::runtime_error(
                    "SparseLU factorization failed at t=" + std::to_string(t));

            T_new = lu.solve(b);

            const double res = (T_new - T_iter).norm() / (T_new.norm() + 1e-15);
            T_iter = T_new;
            if (res < tol) break;
        }

        T = T_new;
        time_arr.push_back(t);
        picard_arr.push_back(picard_k + 1);

        // Actual q from converged solution
        const int n_active = Ny / 2;
        double k_prom = 0.0;
        for (int j = 0; j < n_active; ++j)
            k_prom += k_cell(T[j * Nx], K_flat[j * Nx], gamma_k, T0);
        k_prom /= n_active;
        q_arr.push_back(grad_T_val * k_prom);

        // Corner temperatures
        c_inf_izq.push_back(T[ci_ii]);
        c_inf_der.push_back(T[ci_id]);
        c_sup_izq.push_back(T[ci_si]);
        c_sup_der.push_back(T[ci_sd]);

        // Field snapshots
        if ((n + 1) % plot_every == 0) {
            plot_times.push_back(t);
            T_snaps.emplace_back(T.data(), T.data() + N);

            std::vector<double> kf(N);
            for (int p = 0; p < N; ++p)
                kf[p] = k_cell(T[p], K_flat[p], gamma_k, T0);
            k_snaps.push_back(std::move(kf));
        }
    }

    // ---- Pack results into numpy arrays ------------------------------------

    const int n_snap = static_cast<int>(T_snaps.size());

    // T_fields shape (n_snap, Ny, Nx)
    py::array_t<double> T_fields_np({n_snap, Ny, Nx});
    auto tf = T_fields_np.mutable_unchecked<3>();
    for (int s = 0; s < n_snap; ++s)
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i)
                tf(s, j, i) = T_snaps[s][j * Nx + i];

    // k_fields shape (n_snap, Ny, Nx)
    py::array_t<double> k_fields_np({n_snap, Ny, Nx});
    auto kf = k_fields_np.mutable_unchecked<3>();
    for (int s = 0; s < n_snap; ++s)
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i)
                kf(s, j, i) = k_snaps[s][j * Nx + i];

    auto to_np = [](const std::vector<double>& v) {
        py::array_t<double> a(v.size());
        std::copy(v.begin(), v.end(), a.mutable_data());
        return a;
    };

    py::dict corner;
    corner["inf-izq"] = to_np(c_inf_izq);
    corner["inf-der"] = to_np(c_inf_der);
    corner["sup-izq"] = to_np(c_sup_izq);
    corner["sup-der"] = to_np(c_sup_der);

    py::dict result;
    result["time_array"]   = to_np(time_arr);
    result["q_array"]      = to_np(q_arr);
    result["picard_iters"] = py::array_t<int>(picard_arr.size(), picard_arr.data());
    result["plot_times"]   = to_np(plot_times);
    result["T_fields"]     = T_fields_np;
    result["k_fields"]     = k_fields_np;
    result["corner_T"]     = corner;
    result["T_final"]      = to_np(std::vector<double>(T.data(), T.data() + N));
    return result;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(heat_core, m) {
    m.doc() = "FVM 2D heat diffusion solver — C++ / Eigen backend";
    m.def("simulate", &simulate,
        py::arg("Nx"),
        py::arg("Ny"),
        py::arg("dx_arr"),
        py::arg("dy"),
        py::arg("K_BASE"),
        py::arg("RHO"),
        py::arg("CP"),
        py::arg("gamma_k"),
        py::arg("T0"),
        py::arg("T_init"),
        py::arg("dt"),
        py::arg("t_end"),
        py::arg("grad_T_vals"),   // pre-computed array, shape (Nt,)
        py::arg("max_iter") = 50,
        py::arg("tol")      = 1e-6,
        py::arg("n_plots")  = 8,
        R"doc(
Run the full FVM simulation.

Parameters
----------
Nx, Ny        : grid size
dx_arr        : cell widths in x, shape (Nx,)
dy            : uniform cell height in y
K_BASE        : base thermal conductivity, shape (Ny, Nx)
RHO, CP       : density and specific heat, shape (Ny, Nx)
gamma_k       : k(T) exponential coefficient [1/°C]
T0            : reference temperature [°C]
T_init        : uniform initial temperature [°C]
dt            : time step [s]
t_end         : end time [s]
grad_T_vals   : imposed temperature gradient at each step [°C/m], shape (Nt,)
max_iter      : max Picard iterations per step
tol           : Picard convergence tolerance
n_plots       : number of field snapshots to store

Returns
-------
dict with keys: time_array, q_array, picard_iters, plot_times,
                T_fields (n_snap,Ny,Nx), k_fields (n_snap,Ny,Nx),
                corner_T (dict of 4 arrays), T_final (N,)
        )doc"
    );
}
