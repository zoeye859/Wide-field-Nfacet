from scipy.optimize import leastsq, brent
from scipy.linalg import solve_triangular
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=16)

def trap(vec, dx):
    # Perform trapezoidal integration
    return dx * (np.sum(vec) - 0.5 * (vec[0] + vec[-1]))

def func_to_min(h, x0, M, W):
    N = len(h)
    nu = (np.arange(M, dtype=float) + 0.5) / (2 * M)
    x = x0 * np.arange(N+1, dtype=float)/N
    C = calc_gridder_as_C(h, x0, nu, W)
    dnu = nu[1] - nu[0]
    dx = x[1] - x[0]
    h_ext = np.concatenate(([1.0], h))
    loss = np.zeros((len(h_ext), 2, M), dtype=float)
    for n, x_val in enumerate(x):
        one_app = 0
        for r in range(0, W):
            l = r - (W/2) + 1
            one_app += h_ext[n] * C[r, :] * np.exp(2j * np.pi * (l - nu) * x_val)
        loss[n, 0, :] = 1.0 - np.real(one_app)
        loss[n, 1, :] = np.imag(one_app)
        if n in [0, N]:
            loss[n, :, :] /= np.sqrt(2)
    loss = loss.reshape(2 * M * (N + 1))
    return loss

def optimal_grid(W, x0, N, M, h_initial=None):
    if h_initial is None:
        h_initial = np.ones(N, dtype=float)
    return leastsq(func_to_min, h_initial, args=(x0, M, W), full_output=True)

def make_evaluation_grids(W, M, N):
    """Generate vectors nu and x on which the gridder and gridding correction functions need to be evaluated.
        W is the number of integer gridpoints in total 
        M determines the sampling of the nu grid, dnu = 1/(2*M)
        N determines the sampling of the x grid, dx = 1/(2*N)
    """
    nu = (np.arange(W * M, dtype=float) + 0.5) / (2 * M)
    x = np.arange(N+1, dtype=float)/(2 * N)
    return nu, x

def calc_gridder_as_C_old(h, x0, nu, R):
    # Calculate gridding function C(l,nu) from gridding correction h(x) evaluated at (n+1) x_0/N where n is in range 0,...,N-1. 
    #  We assume that h(0)=1 for normalization, and use the trapezium rule for integration.
    # The gridding function is calculated for l=-R+1 to R at points nu
    M = len(nu)
    C = np.zeros((W, M), dtype=float)
    K = np.zeros((W, W))
    N = len(h)
    x = x0 * np.arange(0, N+1, dtype=float)/N
    dx = x0 / N
    h_ext = np.concatenate(([1.0], h))
    
    for rp in range(0, W):
        lp = rp - (W/2) + 1
        for r in range(W):
            l = r -  (W/2) + 1
            K[rp, r] = trap((h_ext**2) * np.cos(2 * np.pi * (lp - l) * x), dx)
    #Kinv = np.linalg.pinv(K)
    
    rhs = np.zeros(W, dtype=float)
    for m, nu_val in enumerate(nu):
        for rp in range(0, W):
            lp = rp - (W/2) + 1
            rhs[rp] = trap(h_ext * np.cos(2 * np.pi * (lp - nu_val) * x), dx)
        C[:,m] = np.linalg.lstsq(K, rhs, 1e-14)[0]
        # C[:,m] = np.dot(Kinv, rhs)
    return C

def calc_gridder_as_C(h, x0, nu, W):
    # Calculate gridding function C(l,nu) from gridding correction h(x) evaluated at (n+1) x_0/N where n is in range 0,...,N-1. 
    #  We assume that h(0)=1 for normalization, and use the trapezium rule for integration.
    # The gridding function is calculated for l=(1-W)/2 to (W-1)/2 at points nu
    factor = 1./np.sqrt(2.)
    M = len(nu)
    C = np.zeros((W, M), dtype=float)
    N = len(h)
    B = np.zeros((2 * N + 2, W))
    x = x0 * np.arange(0, N+1, dtype=float)/N
    dx = x0 / N
    h_ext = np.concatenate(([1.0], h))
    
    rhs = np.r_[np.ones(N + 1, dtype=float), np.zeros(N + 1, dtype=float)]
    rhs[0] = rhs[0]*factor
    rhs[N] = rhs[N]*factor

    for m, nu_val in enumerate(nu):
        for r in range(W):
            k = r - (W/2) + 1
            B[:N+1, r] = h_ext * np.cos(2 * np.pi * (k - nu_val) * x)
            B[N+1:, r] = h_ext * np.sin(2 * np.pi * (k - nu_val) * x)
        B[0, :] = B[0, :]*factor
        B[N, :] = B[N, :]*factor
        B[N+1, :] = B[N+1, :]*factor
        B[2*N+1, :] = B[2*N+1, :]*factor
#        q, r = np.linalg.qr(B)
#        C[:,m] = solve_triangular(r,np.dot(q.transpose(),rhs))
        C[:,m] = np.linalg.lstsq(B, rhs)[0]
    return C

def calc_gridder_as_C_odd(h, x0, nu, W):
    # Calculate gridding function C(l,nu) from gridding correction h(x) evaluated at (n+1) x_0/N where n is in range 0,...,N-1. 
    #  We assume that h(0)=1 for normalization, and use the trapezium rule for integration.
    # The gridding function is calculated for l=(1-W)/2 to (W-1)/2 at points nu
    factor = 1./np.sqrt(2.)
    M = len(nu)
    C = np.zeros((W, M), dtype=float)
    N = len(h)
    B = np.zeros((2 * N + 2, W))
    x = x0 * np.arange(0, N+1, dtype=float)/N
    dx = x0 / N
    h_ext = np.concatenate(([1.0], h))
    
    rhs = np.r_[np.ones(N + 1, dtype=float), np.zeros(N + 1, dtype=float)]
    rhs[0] = rhs[0]*factor
    rhs[N] = rhs[N]*factor

    for m, nu_val in enumerate(nu):
        for r in range(W):
            if nu_val > 0.5:
                k = r - (W//2) + 1
            else:
                k = r - (W//2) 
            B[:N+1, r] = h_ext * np.cos(2 * np.pi * (k - nu_val) * x)
            B[N+1:, r] = h_ext * np.sin(2 * np.pi * (k - nu_val) * x)
        B[0, :] = B[0, :]*factor
        B[N, :] = B[N, :]*factor
        B[N+1, :] = B[N+1, :]*factor
        B[2*N+1, :] = B[2*N+1, :]*factor
#        q, r = np.linalg.qr(B)
#        C[:,m] = solve_triangular(r,np.dot(q.transpose(),rhs))
        C[:,m] = np.linalg.lstsq(B, rhs)[0]
    return C


def calc_gridder(h, x0, nu, W):
    # Calculate gridder function on a grid nu which should have been generated using make_evaluation_grids
    #  The array h is the result of an optimization process for the gridding correction function evaluated
    #  on a relatively coarse grid extending from 0 to x0
    C = calc_gridder_as_C(h, x0, nu, W)
    gridder = np.zeros(M*W, dtype=float)
    for m in range(M):
        for rp in range(0, W):
            lp = rp - (W/2) + 1
            indx = int(m - 2*lp*M)
            if indx >= 0:
                gridder[indx] = C[rp, m]
            else:
                gridder[-indx-1] = C[rp, m]
    return gridder

def gridder_to_C(gridder, W):
    """Reformat gridder evaluated on the nu grid returned by make_evaluation_grids into the sampled C function 
    which has an index for the closest gridpoint and an index for the fractional distance from that gridpoint
    """
    M = len(gridder) // W
    C = np.zeros((W, M), dtype=float)
    for r in range(0, W):
        l = r - (W/2) + 1
        indx = (np.arange(M) - 2 * M * l).astype(int)
        # Use symmetry to deal with negative indices
        indx[indx<0] = -indx[indx<0] - 1
        C[r, :] = gridder[indx]
    return C
    
def gridder_to_grid_correction(gridder, nu, x, W):
    """Calculate the optimal grid correction function from the gridding function. The vectors x and nu should
    have been constructed using make_evaluation_grids"""
    M = len(nu) // W
    N = len(x) - 1
    dnu = nu[1] - nu[0]
    C = gridder_to_C(gridder, W)
    c = np.zeros(x.shape, dtype=float)
    d = np.zeros(x.shape, dtype=float)
    for n, x_val in enumerate(x):
        for rp in range(0, W):
            lp = rp - (W/2) + 1
            for r in range(0, W):
                l = r - (W/2) + 1
                d[n] += np.sum(C[rp, :] * C[r, :] * np.cos(2 * np.pi * (lp - l) * x_val)) * dnu
            c[n] += np.sum(C[rp, :] * np.cos(2 * np.pi * (lp - nu[:M]) * x_val)) * dnu
    return c/d

def calc_map_error(gridder, grid_correction, nu, x, W):
    M = len(nu) // W
    N = len(x) - 1
    dnu = nu[1] - nu[0]
    C = gridder_to_C(gridder, W)
    loss = np.zeros((len(x), 2, M), dtype=float)
    for n, x_val in enumerate(x):
        one_app = 0
        for r in range(0, W):
            l = r - (W/2) + 1
            one_app += grid_correction[n] * C[r, :] * np.exp(2j * np.pi * (l - nu[:M]) * x_val)
        loss[n, 0, :] = 1.0 - np.real(one_app)
        loss[n, 1, :] = np.imag(one_app)
    map_error = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        map_error[i] = 2 * np.sum((loss[i, :, :].flatten())**2) * dnu
    return map_error
