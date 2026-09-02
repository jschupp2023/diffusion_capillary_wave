from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.sparse import eye as speye
import math

import time
import scipy.linalg as la
from scipy.signal import savgol_filter

import scipy.io as sio
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

import os, h5py

def load_Q_dataset(power: str, labels: list[str], base_path: str = "/home/jonas/ucsd_thesis/DHM_new_1Dcenter"
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    """
    Load Q datasets for given power and labels.

    Parameters
    ----------
    power : str
        Example: '0p10'
    labels : list of str
        Example: ['a','b','c',...]
    base_path : str
        Root directory

    Returns
    -------
    t : ndarray
    x : ndarray
    Q : dict
        Dictionary of Q matrices keyed by label
    nx : int
    """
    
    base = os.path.join(base_path, power)

    def h5read(fname, dset):
        with h5py.File(fname, "r") as f:
            return np.array(f[dset])

    # Use first label to load t, x
    first_file = os.path.join(base, f"Q_1D_{power}vpp_{labels[0]}.h5")
    t = h5read(first_file, "/t")
    x = h5read(first_file, "/x")
    nx = len(x)

    # Load all Q
    Q = {}
    for lbl in labels:
        fname = os.path.join(base, f"Q_1D_{power}vpp_{lbl}.h5")
        Q[lbl] = h5read(fname, "/Q_1D")

    return Q, t, x, nx

def preprocess_Q(Q_dict: dict[str, np.ndarray], t: np.ndarray, labels: list[str], split_size: int = 5760) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess Q data:
    - split into segments
    - concatenate across realizations
    - compute mean field

    Parameters
    ----------
    Q_dict : dict
        {label: Q matrix (nx, nt)}
    t : ndarray
        time vector
    labels : list
        list of labels to include
    split_size : int

    Returns
    -------
    Q_split : dict
        {label: (nx, num_segs, split_size)}
    Qstate_all : ndarray
        (nx, total_segs, split_size)
    X_mean : ndarray
        (nx, split_size)
    tt : ndarray
        truncated time vector
    """

    nt = len(t)
    num_segs = nt // split_size
    tt = t[:split_size]

    def split_Q(Qi):
        Qi = Qi[:, :num_segs * split_size]
        nx = Qi.shape[0]
        return Qi.reshape(nx, num_segs, split_size)

    # Split all
    Q_split = {lbl: split_Q(Q_dict[lbl]) for lbl in labels}

    # Concatenate across labels (segments axis = 1)
    Qstate_all = np.concatenate(
        [Q_split[lbl] for lbl in labels],
        axis=1
    )

    # Mean over segments
    X_mean = np.mean(Qstate_all, axis=1)

    return Q_split, Qstate_all, X_mean, tt


def page_norm(A: np.ndarray) -> float:
    """
    Frobenius norm over all entries and pages.
    Equivalent to MATLAB: norm(A(:))
    """
    return np.linalg.norm(A.ravel())

def page_cov(x: np.ndarray, transpose_pages: bool = False) -> np.ndarray:
    """
    Compute covariance matrix pagewise.

    Parameters
    ----------
    x : ndarray, shape (nObs, nVar, nPages) or (nVar, nObs, nPages)
    transpose_pages : bool
        If True, swap first two axes before computing covariance.

    Returns
    -------
    C : ndarray, shape (nVar, nVar, nPages)
        Pagewise covariance matrices.
    """
    if transpose_pages:
        x = np.transpose(x, (1, 0, 2))

    nObs, nVar, nPages = x.shape
    C = np.zeros((nVar, nVar, nPages))

    for i in range(nPages):
        Xi = x[:, :, i]                       # (nObs, nVar)
        Xi = Xi - Xi.mean(axis=0, keepdims=True)

        if nObs > 1:
            C[:, :, i] = (Xi.T @ Xi) / (nObs - 1)
        else:
            C[:, :, i] = np.zeros((nVar, nVar))

    return C

def central_finite_differences(x: np.ndarray, h: float, ord: int = 2, axis: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute central finite differences of x along a given axis.

    Parameters
    ----------
    x : ndarray
        Input array.
    h : float
        Time step.
    ord : int, optional
        Accuracy: 2, 4, 6, 8 (default=2).
    axis : int, optional
        Axis along which to compute derivative (default=1, 0-indexed in Python).

    Returns
    -------
    y : ndarray
        Interior points of x where derivative is defined.
    dxdt : ndarray
        Central finite difference derivative along axis.
    ind : ndarray
        Valid indices along the axis used.
    """

    # -----------------------------
    # Coefficients for central differences
    # -----------------------------
    if ord == 2:
        indx = np.array([-1, 0, 1])
        coef = np.array([-0.5, 0.0, 0.5]) / h
    elif ord == 4:
        indx = np.array([-2, -1, 0, 1, 2])
        coef = np.array([1/12, -2/3, 0, 2/3, -1/12]) / h
    elif ord == 6:
        indx = np.array([-3, -2, -1, 0, 1, 2, 3])
        coef = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]) / h
    elif ord == 8:
        indx = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        coef = np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]) / h
    else:
        raise ValueError("acc can only be 2, 4, 6, or 8!")

    # -----------------------------
    # Determine interior indices
    # -----------------------------
    p = len(coef)
    n = x.shape[axis]
    start = (p - 1) // 2
    end = n - (p - 1) // 2
    ind = np.arange(start, end)  # valid indices

    # -----------------------------
    # Allocate output lists
    # -----------------------------
    dxdt_list = []
    y_list = []

    # -----------------------------
    # Iterate over valid indices
    # -----------------------------
    for ii in ind:
        # Slice array around current index along given axis
        slicer = [slice(None)] * x.ndim
        values = []
        for k, offset in enumerate(indx):
            slicer[axis] = ii + offset
            values.append(coef[k] * x[tuple(slicer)])

        dX = sum(values)

        # Current value
        slicer[axis] = ii
        Xii = x[tuple(slicer)]

        dxdt_list.append(dX)
        y_list.append(Xii)

    # Stack along the axis
    dxdt = np.stack(dxdt_list, axis=axis)
    y = np.stack(y_list, axis=axis)

    return y, dxdt, ind




def regularizer(r: int, lam: np.ndarray, isbilinear: bool, isBu: bool) -> np.ndarray:
    """
    Construct regularizer diagonal matrix.

    Parameters
    ----------
    r : int
        Reduced dimension
    lam : array-like (length 2 or 3)
        Regularization parameters [lambda_A, lambda_B, (lambda_N)]
    isbilinear : bool
        Whether bilinear term is included
    isBu : bool
        Whether the input term is included

    Returns
    -------
    Reg : (d, d) ndarray
        Diagonal regularization matrix
    """

    m = 1

    if isbilinear:
        if isBu:
            d = r + m + r*m
        else:
            d = r + r*m
    else:
        if isBu:
            d = r + m
        else:
            d = r

    Reg_vec = np.zeros(d)

    # A block
    Reg_vec[:r] = lam[0]

    if isBu:
        # B block
        Reg_vec[r] = lam[1]
        
        if isbilinear:
            Reg_vec[r + 1:] = lam[2]
    else:
        if isbilinear:
            Reg_vec[r:] = lam[1]

    return np.diag(Reg_vec)


def infer_drift(E_train: np.ndarray, u_train: np.ndarray, h: float, isbilinear: bool, isBu: bool, regs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Infer the drift operators for the reduced-order model (OpInf).

    Parameters
    ----------
    E_train : ndarray, shape (r, s)
        Expected value of the reduced states across samples.
    h : float
        Time step.
    isbilinear : bool
        Whether the system has bilinear terms.
    s : int
        Number of time snapshots.
    reg : float or None
        Regularization parameter. If None, no regularization is applied.

    Returns
    -------
    Ehat : ndarray, shape (r, r)
        Mass matrix (identity in our examples).
    Ahat : ndarray, shape (r, r)
        Linear reduced operator.
    Nhat : ndarray, shape (r, ?)
        Bilinear reduced operator (zeros if isbilinear=False).
    """

    r = E_train.shape[0]
    
    # m = len(u_train)
    m = 1
    u = u_train

    # Central finite difference (accuracy 2)
    # Returns Er (states), Er_dot (time derivatives), ind (indices used)
    Er, Er_dot, ind = central_finite_differences(E_train, h, ord=6, axis=1)
    
    K = len(ind)
    
    if isbilinear:
        col_dim = r + m + r*m
    else:
        col_dim = r + m

    # build data matrix
    if isbilinear:
        # u: (m, K)
        # Er: (r, K)

        kron_block = np.zeros((r * m, K))

        for jj in range(K):
            # kron_block[:, jj] = np.kron(u[jj], Er[:, jj])
            # kron_block[:, jj] = np.kron(u[:, jj], Er[:, jj])
            kron_block[:, jj] = u[jj] * Er[:, jj]

        if isBu:
            D = np.vstack([Er[:, :K], u[:K], kron_block])
        else:
            D = np.vstack([Er[:, :K], kron_block])
        
    else: # not bilinear
        if isBu:    
            D = np.vstack([
                Er[:, :K], u[:K],
            ])
        else:
            D = np.vstack([Er[:, :K]])
        
    rhs = Er_dot[:, :K]
        
    Dt = D.T
    Rt = rhs.T

    # Solve least-squares problem (unconstrained)
    if regs is not None:
        Gamma = regularizer(r, regs, isbilinear, isBu)
        D_modified = Dt.T @ Dt + Gamma.T @ Gamma
        rhs_modified = D @ Rt          # note: rhs.T to match MATLAB dimensions
        ops = np.linalg.solve(D_modified, rhs_modified)
        ops = ops.T
    else:
        # Non-regularized least-squares
        # ops = rhs / D  --> MATLAB left-division
        ops = np.linalg.lstsq(Dt, Rt, rcond=None)[0].T
    
    
    # Extract operators
    Ahat = ops[:, :r]
    
    if isBu:
        Bhat = ops[:, r:r + m]
        
        if isbilinear:
            Nhat = ops[:, r+m:]      # m must be defined or passed if using inputs
        else:
            Nhat = np.zeros([r, r])
    else:
        Bhat = np.zeros([r, 1])
        if isbilinear:
            Nhat = ops[:, r:]      # m must be defined or passed if using inputs
        else:
            Nhat = np.zeros([r, r])

    # Mass matrix
    Ehat = speye(r).toarray()   # dense identity matrix

    return Ehat, Ahat, Bhat, Nhat



def infer_diffusion(C_train: np.ndarray, u_train: np.ndarray, h: float, Ahat: np.ndarray, Nhat: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer diffusion operator from covariance dynamics.
    """
    
    # Central finite differences along the 3rd axis (time)
    Cr_dot = savgol_filter(C_train, window_length=7, polyorder=3, deriv=1, delta=h, axis=2)
    Cr = C_train
    ind = Cr.shape[2]
    T = ind

    n = Ahat.shape[0]
    Hhat = np.zeros((n, n))
    
    # Build unconstrained Hhat
    for jj in range(T):
        u_j = u_train[:, jj] if u_train.ndim > 1 else u_train[jj]

        Psi_hat = Ahat + Nhat * u_j
        Clyap = Psi_hat @ Cr[:, :, jj]
        incre = Cr_dot[:, :, jj] - (Clyap + Clyap.T)

        Hhat += 0.5 * (incre + incre.T)      # symmetrize each step before accumulation
        
        # Hhat += incre
    
    Hhat /= (T + reg)  # average over time steps

    # Symmetrize before projection
    # Hhat = 0.5 * (Hhat + Hhat.T)
    
    # Project onto PSD cone
    eigvals, eigvecs = np.linalg.eigh(Hhat)
    Hhat_psd = eigvecs @ np.diag(np.maximum(eigvals, 0)) @ eigvecs.T    # clamp negative eigenvalues to zero (nearst PSD matrix)

    # Eigen-decomposition
    eigvals_psd, eigvecs_psd = np.linalg.eigh(Hhat_psd)

    # sort descending
    idx = np.argsort(eigvals_psd)[::-1]
    eigvals_psd = eigvals_psd[idx]
    eigvecs_psd = eigvecs_psd[:, idx]

    HS = np.diag(eigvals_psd)
    HU = eigvecs_psd

    # truncation threshold
    tol = np.max(eigvals_psd) / 1e3

    valid = np.where(eigvals_psd >= tol)[0]
    d = valid[-1] + 1 if len(valid) > 0 else 0

    # avoid log-scale issues
    eps = 1e-16
    
    # print(d)

    # reconstruct diffusion factor
    if d > 0:
        Mhat = HU[:, :d] @ np.sqrt(HS[:d, :d])
    else:
        Mhat = np.zeros((n, 0))
    
    Khat = np.eye(d)

    return Mhat, Khat


def compute_model(f: Callable, xr0_test: np.ndarray, u_test: np.ndarray, batch_size: int, L_test: int, noise_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of FOM/ROM states with input u_test.

    Parameters
    ----------
    f : function
        Step function: f(x0, u, L) -> Euler-Maruyama simulation
    xr0_test : ndarray
        Initial condition (r x 1)
    u_test : ndarray
        Input signal (m x s)
    batch_size : int
        Number of trajectories per batch
    L_test : int
        Total number of noise samples

    Returns
    -------
    E_test : ndarray
        Mean of the states (r x s)
    C_test : ndarray
        Covariance of the states (r x r x s)
    """

    r = xr0_test.shape[0]
    # m, s = u_test.shape
    s = len(u_test)

    # Determine number of batches
    num_batches = math.ceil(L_test / batch_size)
    if L_test <= batch_size:
        num_batches = 1
        batch_size = L_test

    # Initialize accumulators
    E_test = np.zeros((r, s))
    C_test = np.zeros((r, r, s))

    for batch in range(num_batches):
        # Last batch might be smaller
        Nb = batch_size
        if batch == num_batches - 1:    # last batch
            Nb = L_test - (batch) * batch_size  

        # Compute statistics for one batch
        E_temp, C_temp, Xr_temp = estimate(f, xr0_test, u_test, Nb, noise_dim)

        # Accumulate mean
        E_test += (Nb / L_test) * E_temp

        # Accumulate second moments
        for k in range(s):
            # outer product of mean vector
            C_test[:, :, k] += (Nb / L_test) * (C_temp[:, :, k] + np.outer(E_temp[:, k], E_temp[:, k]))
            
    # Subtract mean outer product to get covariance
    for k in range(s):
        C_test[:, :, k] -= np.outer(E_test[:, k], E_test[:, k])
        
    Xr_test = Xr_temp

    return E_test, C_test, Xr_test


def estimate(f: Callable, xr0: np.ndarray, u: np.ndarray, L: int, noise_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Empirical mean and covariance from stochastic simulations.
    """

    # Run stochastic simulations
    # Xr shape: (r, L, s)
    Xr = stepSDE(f, xr0, u, L, noise_dim)

    # Mean (over noise samples)
    E_empirical = np.mean(Xr, axis=1)   # shape: (r, s)

    # Empirical covariance
    C_empirical = page_cov(Xr, transpose_pages=True)  # shape (r, r, s)

    return E_empirical, C_empirical, Xr


def stepSDE(f: Callable, x0: np.ndarray, u: np.ndarray, L: int, noise_dim: int) -> np.ndarray:
    """
    Run stochastic Euler-Maruyama steps over time with input u.

    Parameters
    ----------
    f : function
        EulerMaruyamaStep-like function:
        f(x, u_t, L) -> (r, L) samples
    x0 : ndarray
        Initial condition (r,)
    u : ndarray
        Input signal (m, s) or (s,) depending on model
    L : int
        Number of noise realizations

    Returns
    -------
    Xr : ndarray
        Shape (r, L, s)
    """

    s = len(u)
    r = x0.shape[0]

    Xr = np.zeros((r, L, s))
    
    # Generate fresh noise for this batch: (noise_dim, L, s)
    noise_allsample = np.random.randn(noise_dim, L, s)
    
    # Handle IC
    if x0.ndim == 1 or x0.shape[1] == 1:
        x0_batch = np.tile(x0.reshape(r, 1), (1, L))  # (r, L)
    else:
        x0_batch = x0        

    # First time step
    Xr[:, :, 0] = f(x0_batch, u[:, 0] if u.ndim > 1 else u[0], L, 0, noise_allsample[:, :, 0])
    # Xr[:, :, 0] = f(x0, u[:, 0] if u.ndim > 1 else u[0], L, 0)

    # Time stepping
    for ii in range(1, s):
        u_prev = u[:, ii-1] if u.ndim > 1 else u[ii-1]  # scalar
        Xr[:, :, ii] = f(Xr[:, :, ii-1], u_prev, L, ii, noise_allsample[:, :, ii])
        # Xr[:, :, ii] = f(Xr[:, :, ii-1], u_prev, L)

    return Xr


from sympy import symbols, factorial, Matrix, Rational
def finite_diff_coeffs(x_vals: list[int], x0: int, derivative_order: int) -> np.ndarray:
    """
    x_vals: list of grid points (e.g., [0, 1, 2, 3, 4, 5, 6])
    x0: the point at which the derivative is to be approximated (e.g., 1 for coeffs1)
    derivative_order: which derivative (1 for first derivative)
    """
    m = len(x_vals)
    # k outer loop, x inner loop
    A = Matrix([[Rational((x - x0)**k, factorial(k)) for x in x_vals] for k in range(m)])
    b = Matrix([1 if k == derivative_order else 0 for k in range(m)])
    coeffs = A.LUsolve(b)
    return np.array(coeffs, dtype=np.float64)

def ctr_FD_tensor(X: np.ndarray, h: float, order: int, axis: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    High-order central FD along given axis (vectorized).
    """

    half = order // 2
    offsets = np.arange(-half, half + 1)

    # central coefficients (scaled properly)
    coeffs = finite_diff_coeffs(offsets * h, 0, 1)
    coeffs = np.array(coeffs, dtype=float)

    # valid indices
    n = X.shape[axis]
    ind = np.arange(half, n - half)

    # slice central region
    slices = [slice(None)] * X.ndim
    slices[axis] = ind
    Xc = X[tuple(slices)]

    dX = np.zeros_like(Xc)

    for k, offset in enumerate(offsets):
        slices[axis] = ind + offset
        dX += coeffs[k] * X[tuple(slices)]

    return Xc, dX, ind

def ctr_FD(f: np.ndarray, h: float, order: int) -> np.ndarray:
    N = len(f)
    df = np.empty_like(f)

    x_vals = list(range(order+1))
    
    if order == 8:
        # Central difference (i = 4 to N-5)
        for i in range(4, N-4):
            df[i] = (3*f[i-4] - 32*f[i-3] + 168*f[i-2] - 672*f[i-1]
                    + 672*f[i+1] - 168*f[i+2] + 32*f[i+3] - 3*f[i+4]) / (840*h)

        scale = 840
        coeffs0 = np.round(finite_diff_coeffs(x_vals, x0=0, derivative_order=1) * scale).astype(int).squeeze()
        coeffs1 = np.round(finite_diff_coeffs(x_vals, x0=1, derivative_order=1) * scale).astype(int).squeeze()
        coeffs2 = np.round(finite_diff_coeffs(x_vals, x0=2, derivative_order=1) * scale).astype(int).squeeze()
        coeffs3 = np.round(finite_diff_coeffs(x_vals, x0=3, derivative_order=1) * scale).astype(int).squeeze()

        # Forward-biased stencils (last 4 points)
        df[0] = f[:9] @ coeffs0
        df[1] = f[:9] @ coeffs1
        df[2] = f[:9] @ coeffs2
        df[3] = f[:9] @ coeffs3

        # Backward-biased stencils (last 4 points)
        df[-4] = f[-9:] @ -coeffs3[::-1]
        df[-3] = f[-9:] @ -coeffs2[::-1]
        df[-2] = f[-9:] @ -coeffs1[::-1]
        df[-1] = f[-9:] @ -coeffs0[::-1]
        
    if order == 10:
        for i in range(5, N-5):
            df[i] = (-2*f[i-5] + 25*f[i-4] - 150*f[i-3] + 600*f[i-2] - 2100*f[i-1]
                    + 2100*f[i+1] - 600*f[i+2] + 150*f[i+3] - 25*f[i+4] + 2*f[i+5]) / (2520*h)
        
        scale = 2520
        coeffs0 = np.round(finite_diff_coeffs(x_vals, x0=0, derivative_order=1) * scale).astype(int).squeeze()
        coeffs1 = np.round(finite_diff_coeffs(x_vals, x0=1, derivative_order=1) * scale).astype(int).squeeze()
        coeffs2 = np.round(finite_diff_coeffs(x_vals, x0=2, derivative_order=1) * scale).astype(int).squeeze()
        coeffs3 = np.round(finite_diff_coeffs(x_vals, x0=3, derivative_order=1) * scale).astype(int).squeeze()
        coeffs4 = np.round(finite_diff_coeffs(x_vals, x0=4, derivative_order=1) * scale).astype(int).squeeze()

        # Forward-biased stencils (last 5 points)
        df[0] = f[:11] @ coeffs0
        df[1] = f[:11] @ coeffs1
        df[2] = f[:11] @ coeffs2
        df[3] = f[:11] @ coeffs3
        df[4] = f[:11] @ coeffs4

        # Backward-biased stencils (last 5 points)
        df[-5] = f[-11:] @ -coeffs4[::-1]
        df[-4] = f[-11:] @ -coeffs3[::-1]
        df[-3] = f[-11:] @ -coeffs2[::-1]
        df[-2] = f[-11:] @ -coeffs1[::-1]
        df[-1] = f[-11:] @ -coeffs0[::-1]

    if order == 12:
        for i in range(6, N-6):
            df[i] = (5*f[i-6] - 72*f[i-5] + 495*f[i-4] - 2200*f[i-3] + 7425*f[i-2] - 23760*f[i-1]
                    + 23760*f[i+1] - 7425*f[i+2] + 2200*f[i+3] - 495*f[i+4] + 72*f[i+5] - 5*f[i+6]) / (27720*h)
        
        scale = 27720
        coeffs0 = np.round(finite_diff_coeffs(x_vals, x0=0, derivative_order=1) * scale).astype(int).squeeze()
        coeffs1 = np.round(finite_diff_coeffs(x_vals, x0=1, derivative_order=1) * scale).astype(int).squeeze()
        coeffs2 = np.round(finite_diff_coeffs(x_vals, x0=2, derivative_order=1) * scale).astype(int).squeeze()
        coeffs3 = np.round(finite_diff_coeffs(x_vals, x0=3, derivative_order=1) * scale).astype(int).squeeze()
        coeffs4 = np.round(finite_diff_coeffs(x_vals, x0=4, derivative_order=1) * scale).astype(int).squeeze()
        coeffs5 = np.round(finite_diff_coeffs(x_vals, x0=5, derivative_order=1) * scale).astype(int).squeeze()

        # Forward-biased stencils (last 5 points)
        df[0] = f[:13] @ coeffs0
        df[1] = f[:13] @ coeffs1
        df[2] = f[:13] @ coeffs2
        df[3] = f[:13] @ coeffs3
        df[4] = f[:13] @ coeffs4
        df[5] = f[:13] @ coeffs5

        # Backward-biased stencils (last 5 points)
        df[-6] = f[-13:] @ -coeffs5[::-1]
        df[-5] = f[-13:] @ -coeffs4[::-1]
        df[-4] = f[-13:] @ -coeffs3[::-1]
        df[-3] = f[-13:] @ -coeffs2[::-1]
        df[-2] = f[-13:] @ -coeffs1[::-1]
        df[-1] = f[-13:] @ -coeffs0[::-1]
        
    return df


def save_result(E_error: np.ndarray, EROM: list, EFOM: np.ndarray, input_name: str, r_opt: int, sigma: float, H_reg_opt: float, isBu: bool) -> str:
    """
    Find optimal EROM entry and save results.

    Parameters
    ----------
    E_error    : ndarray (n_outer, n_inner)
                Matrix of errors
    EROM       : list of lists
                Cell array of ROM predictions, same indexing as E_error
    EFOM       : ndarray
                Full-order model expectation
    input_name : str
                Case name (used in filename)
    r_opt      : int
                Optimal ROM rank
    sigma      : float
                Noise level
    H_reg_opt  : float
                Diffusion regularization parameter
    isBu       : bool
                Whether B (input) term is included

    Returns
    -------
    filename : str
        Path to saved .mat file
    """

    # Remove zeros
    E_error_nonzero = E_error[E_error != 0]
    if len(E_error_nonzero) == 0:
        raise ValueError('E_error contains only zeros.')

    # Find minimum (first occurrence)
    min_val = E_error_nonzero.min()
    flat_idx = np.argmax(E_error == min_val)          # first occurrence
    i_min, j_min = np.unravel_index(flat_idx, E_error.shape)

    # Extract optimal ROM
    EROM_opt = EROM[i_min][j_min]

    # ROM form string
    ROM_form = 'ABN' if isBu else 'AN'

    # Create directory if needed
    save_dir = f'/disk/hyk049/WT_RomFit/{input_name}'
    os.makedirs(save_dir, exist_ok=True)

    # Create filename
    filename = (f'{save_dir}/{input_name}vpp_r={r_opt}_{ROM_form}'
                f'_sigma{sigma:.1f}_Hreg{int(H_reg_opt)}.mat')

    # Save results
    sio.savemat(filename, {'EFOM': EFOM, 'EROM_opt': EROM_opt})

    print(f'Saved to: {filename}')
    print(f'Optimal indices: i_min={i_min}, j_min={j_min} | E_error={min_val:.4f}')

    return filename


def analyze_rom_results(
    E_error: np.ndarray, C_error: np.ndarray,
    Eopinfs: list, Copinfs: list,
    E_train: np.ndarray, C_train: np.ndarray,
    tt: np.ndarray, N_reg: np.ndarray,
    H_reg_opt: float, sigma: float, L: int,
    figPlot: bool = True,
) -> dict:
    """
    Analyze ROM results: find optimal models and generate plots.

    Returns
    -------
    dict with optimal indices and models
    """

    # Find optimal indices
    E_err_flat = E_error.copy()
    E_err_flat[E_err_flat == 0] = np.inf

    C_err_flat = C_error.copy()
    C_err_flat[C_err_flat == 0] = np.inf

    i_Emin, j_Emin = np.unravel_index(E_err_flat.argmin(), E_err_flat.shape)
    i_Cmin, j_Cmin = np.unravel_index(C_err_flat.argmin(), C_err_flat.shape)

    Eopinf_opt = Eopinfs[i_Emin][j_Emin]
    # Copinf_opt = Copinfs[i_Cmin][j_Cmin]
    
    Copinf_opt = Copinfs[i_Emin][j_Emin]

    # (kept for clarity even if same)
    Eopinf_min = Eopinf_opt
    Copinf_min = Copinf_opt

    print(f"\nBest E_error: N_reg={N_reg[i_Emin]:.2e} | E_err={E_error[i_Emin,j_Emin]:.5f}")
    print(f"Best C_error: N_reg={N_reg[i_Cmin]:.2e} | C_err={C_error[i_Cmin,j_Cmin]:.5f}")
    
    print(f"\nC_error under best E_error: N_reg={N_reg[i_Emin]:.2e} | C_err={C_error[i_Emin,j_Emin]:.5f}")

    if figPlot:
        mpl.rcParams.update({'font.size': 12})
        
        colors = {
            "exp": 'C0',   # experiment
            "rom": 'C1',     # rom
            "err": "black",
            "ref": "black" 
        }

        s = E_train.shape[1]

        # --- Time-resolved errors ---
        C_err_min = np.array([
            la.norm(C_train[:, :, k] - Copinf_min[:, :, k], 'fro') /
            la.norm(C_train[:, :, k], 'fro')
            for k in range(s)
        ])

        E_err_min = np.array([
            la.norm(E_train[:, k] - Eopinf_min[:, k]) /
            la.norm(E_train[:, k])
            for k in range(s)
        ])

        # Weak errors plot
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='w')

        ax.semilogy(tt, E_err_min, ls='--', lw=1.5, color=colors["exp"],
            # label=f'Expectation error (avg={E_err_min.mean():.3f})')
            label=f'mean error')
        
        ax.semilogy(tt, C_err_min, ls='-', lw=1.5, color='black',
            # label=f'Covariance error (avg={C_err_min.mean():.3f})')
            label=f'covariance error'
            )

        print(f"Maximum mean error: {np.max(E_err_min):.5f}")
        print(f"Maximum covariance error: {np.max(C_err_min):.5f}")
        
        # ax.axhline(0.20, color = 'r', ls='--', lw=1.5, label='20%')

        ax.set_ylabel('$\\varepsilon_\\text{cov}$', fontsize=35)
        ax.set_xlabel('Time [s]', fontsize=25)
        ax.legend(fontsize=25, framealpha = 0.3)
        ax.grid(True)
        ax.set_xlim([tt[0], tt[-1]])
        ax.set_xticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
        ax.set_ylim([3e-4, 5e-1])
        ax.tick_params(axis='both', which='major', labelsize=25)
        
        plt.tight_layout()
        plt.show()


    return {
        "Eopinf_opt": Eopinf_opt,
        "Copinf_opt": Copinf_opt,
        "i_Emin": i_Emin,
        "j_Emin": j_Emin,
        "i_Cmin": i_Cmin,
        "j_Cmin": j_Cmin
    }


def run_grid_search(data: StochasticOpInfTrainingData, grid: RegularizationGrid,
                        isbilinear: bool, isBu: bool, sigma: float) -> dict:
    """
    Run regularization grid search for stochastic OpInf.

    Parameters
    ----------
    data       : StochasticOpInfTrainingData   training set (E, C, Q, u, Vr, h)
    grid       : RegularizationGrid            N_reg/H_reg/B_loop/get_lambda/seed
    isbilinear : bool                          bilinear ROM flag
    isBu       : bool                          include B term flag
    sigma      : float                         noise scaling

    Returns
    -------
    results : dict with keys:
        Eopinfs, Copinfs, EROM, CROM, ROMs_all,
        E_error, C_error
    """

    E_train, C_train, Q_train, u_train, Vr, h = (
        data.E_train, data.C_train, data.Q_train, data.u_train, data.Vr, data.h)
    N_reg, H_reg_opt, B_loop, get_lambda, seed_test = (
        grid.N_reg, grid.H_reg, grid.B_loop, grid.get_lambda, grid.seed)
    n_outer, L, s = grid.n_outer, data.L, data.s

    # Storage
    n_inner  = len(B_loop)
    Eopinfs  = [[None]*n_inner for _ in range(n_outer)]
    Copinfs  = [[None]*n_inner for _ in range(n_outer)]
    EROM     = [[None]*n_inner for _ in range(n_outer)]
    CROM     = [[None]*n_inner for _ in range(n_outer)]
    ROMs_all = [[None]*n_inner for _ in range(n_outer)]
    E_error  = np.zeros((n_outer, n_inner))
    C_error  = np.zeros((n_outer, n_inner))

    Xr_opinfs = [[None]*n_inner for _ in range(n_outer)]  # store all trajectories

    # Actual experimental ICs
    xr0_exp = Q_train[:, :, 0]   # r x L

    for ii in range(n_outer):
        t0 = time.time()
        # print(f"\n{'='*80}")
        # print(f"{ii+1}th N_reg = {N_reg[ii]:.2e}")
        # print(f"{'='*80}")

        for jj in B_loop:
            lam = get_lambda(ii, jj)
            # print(f"\n  lambda = {lam}")

            # Drift OpInf
            Ehatr, Ahatr, Bhatr, Nhatr = infer_drift(
                E_train, u_train, h, isbilinear, isBu, lam)

            # Diffusion OpInf
            Mhatr, Khatr = infer_diffusion(
                C_train, u_train, h, Ahatr, Nhatr, H_reg_opt)

            # Save operators
            ROMs_all[ii][jj] = dict(Ehat=Ehatr, Ahat=Ahatr, Bhat=Bhatr,
                                    Nhat=Nhatr, Mhat=Mhatr, Khat=Khatr,
                                    lam=lam, N_reg=N_reg[ii], H_reg=H_reg_opt)

            # ROM step function
            np.random.seed(seed_test)

            # from numba import jit
            # @jit(nopython=True)
            def fhatr(x0, u, L_batch, k, noise_k,
                    _E=Ehatr, _A=Ahatr, _B=Bhatr, _N=Nhatr, _M=Mhatr):
                lhs   = _E - h*_A - h*_N*u
                noise = np.sqrt(h) * _M * sigma @ noise_k
                hBu   = (h * _B * u).reshape(-1, 1)
                rhs   = x0 + hBu + noise
                return np.linalg.solve(lhs, rhs)

            # Simulate ROM
            Eopinf, Copinf, Xr_opinf = compute_model(
                fhatr, xr0_exp, u_train, L, L, Mhatr.shape[1])

            # Print error table
            # _print_error_table(E_train, C_train, Eopinf, Copinf, s, la)

            # Store results
            Eopinfs[ii][jj] = Eopinf
            Copinfs[ii][jj] = Copinf
            Xr_opinfs[ii][jj] = Xr_opinf
            EROM[ii][jj]    = Vr @ Eopinf
            CROM[ii][jj]    = Copinf

            E_error[ii,jj] = la.norm(E_train - Eopinf) / la.norm(E_train)
            C_error[ii,jj] = (page_norm(C_train - Copinf) /
                            page_norm(C_train))

        # print(f"\n  Elapsed: {time.time()-t0:.2f}s")

    return dict(Eopinfs=Eopinfs, Copinfs=Copinfs,
                EROM=EROM,       CROM=CROM,
                ROMs_all=ROMs_all,
                E_error=E_error,  C_error=C_error,
                Xr_opinfs=Xr_opinfs)


class QDataset:
    """Loaded/preprocessed Q (1D wave height) dataset for one power level."""

    def __init__(self, Q: dict[str, np.ndarray], t: np.ndarray, x: np.ndarray, nx: int,
                    power: str, labels: list[str], base_path: str) -> None:
        self.Q = Q
        self.t = t
        self.x = x
        self.nx = nx
        self.power = power
        self.labels = labels
        self.base_path = base_path

        self.Q_split: dict[str, np.ndarray] | None = None
        self.Qstate_all: np.ndarray | None = None
        self.X_mean: np.ndarray | None = None
        self.tt: np.ndarray | None = None

    @classmethod
    def load(cls, power: str, labels: list[str], base_path: str = "/home/jonas/ucsd_thesis/DHM_new_1Dcenter") -> "QDataset":
        Q, t, x, nx = load_Q_dataset(power, labels, base_path)
        return cls(Q, t, x, nx, power, labels, base_path)

    def preprocess(self, split_size: int = 5760) -> "QDataset":
        self.Q_split, self.Qstate_all, self.X_mean, self.tt = preprocess_Q(
            self.Q, self.t, self.labels, split_size)
        return self


@dataclass
class StochasticOpInfTrainingData:
    """Training data for a StochasticOpInfROM fit."""

    E_train: np.ndarray   # (r, s)     mean
    C_train: np.ndarray   # (r, r, s)  covariance
    Q_train: np.ndarray   # (r, L, s)  projected snapshots
    u_train: np.ndarray   # (s,)       input signal
    Vr: np.ndarray        # (nx, r)    POD basis
    h: float              # time step

    @property
    def L(self) -> int:
        return self.Q_train.shape[1]

    @property
    def s(self) -> int:
        return self.Q_train.shape[2]


@dataclass
class RegularizationGrid:
    """Regularization search grid for StochasticOpInfROM.grid_search."""

    N_reg: np.ndarray
    H_reg: float
    B_loop: np.ndarray
    get_lambda: Callable
    seed: int

    @property
    def n_outer(self) -> int:
        return len(self.N_reg)


class StochasticOpInfROM:
    """Stochastic operator inference bilinear ROM: dx = [Ax + Bu + Nxu] dt + M dW_t."""

    def __init__(self, isbilinear: bool = True, isBu: bool = True) -> None:
        self.isbilinear = isbilinear
        self.isBu = isBu

        self.Ehat: np.ndarray | None = None
        self.Ahat: np.ndarray | None = None
        self.Bhat: np.ndarray | None = None
        self.Nhat: np.ndarray | None = None
        self.Mhat: np.ndarray | None = None
        self.Khat: np.ndarray | None = None

    def fit_drift(self, E_train: np.ndarray, u_train: np.ndarray, h: float,
                    regs: np.ndarray | None = None) -> "StochasticOpInfROM":
        self.Ehat, self.Ahat, self.Bhat, self.Nhat = infer_drift(
            E_train, u_train, h, self.isbilinear, self.isBu, regs)
        return self

    def fit_diffusion(self, C_train: np.ndarray, u_train: np.ndarray, h: float, reg: float,
                        psd: bool = True) -> "StochasticOpInfROM":
        self.Mhat, self.Khat = infer_diffusion(
            C_train, u_train, h, self.Ahat, self.Nhat, reg)
        return self

    def _step(self, x0: np.ndarray, u: np.ndarray, L: int, k: int, noise_k: np.ndarray,
                h: float, sigma: float) -> np.ndarray:
        lhs = self.Ehat - h * self.Ahat - h * self.Nhat * u
        noise = np.sqrt(h) * self.Mhat * sigma @ noise_k
        hBu = (h * self.Bhat * u).reshape(-1, 1)
        rhs = x0 + hBu + noise
        return np.linalg.solve(lhs, rhs)

    def simulate(self, xr0: np.ndarray, u_test: np.ndarray, h: float, sigma: float,
                    batch_size: int, L: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)

        def f(x0, u, L_batch, k, noise_k):
            return self._step(x0, u, L_batch, k, noise_k, h, sigma)

        return compute_model(f, xr0, u_test, batch_size, L, self.Mhat.shape[1])

    def grid_search(self, data: StochasticOpInfTrainingData,
                            grid: RegularizationGrid, sigma: float) -> dict:
        return run_grid_search(data, grid, self.isbilinear, self.isBu, sigma)
