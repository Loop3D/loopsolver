"""
Unit tests for ADMM batch mode.
"""

import numpy as np
from scipy.sparse import csr_matrix

from loopsolver import admm_solve


def _build_problem(seed=42, n_data=30, n_vars=15, n_ineq=200):
    rng = np.random.RandomState(seed)
    A = csr_matrix(rng.randn(n_data, n_vars))
    x_true = rng.randn(n_vars)
    b = A @ x_true + 0.1 * rng.randn(n_data)

    Q = csr_matrix(rng.randn(n_ineq, n_vars))
    Q_x_true = Q @ x_true
    bounds = np.column_stack(
        [
            Q_x_true - 0.5,
            Q_x_true + 0.5,
        ]
    )

    x0 = np.zeros(n_vars)
    return A, b, Q, bounds, x0


def test_admm_batch_size_runs():
    A, b, Q, bounds, x0 = _build_problem()

    result = admm_solve(
        A,
        b,
        Q,
        bounds,
        x0,
        admm_weight=0.1,
        nmajor=10,
        batch_size=50,
        random_seed=123,
        linsys_solver_kwargs={"atol": 1e-6, "btol": 1e-6},
    )

    assert result.shape == (A.shape[1],)
    assert np.all(np.isfinite(result))


def test_admm_batch_fraction_runs():
    A, b, Q, bounds, x0 = _build_problem(seed=7)

    result = admm_solve(
        A,
        b,
        Q,
        bounds,
        x0,
        admm_weight=0.1,
        nmajor=10,
        batch_fraction=0.25,
        random_seed=456,
        linsys_solver_kwargs={"atol": 1e-6, "btol": 1e-6},
    )

    assert result.shape == (A.shape[1],)
    assert np.all(np.isfinite(result))


def test_admm_batch_mode_matches_full_shape():
    A, b, Q, bounds, x0 = _build_problem(seed=99)

    full_result = admm_solve(
        A,
        b,
        Q,
        bounds,
        x0,
        admm_weight=0.1,
        nmajor=5,
        linsys_solver_kwargs={"atol": 1e-6, "btol": 1e-6},
    )

    batch_result = admm_solve(
        A,
        b,
        Q,
        bounds,
        x0,
        admm_weight=0.1,
        nmajor=5,
        batch_size=40,
        random_seed=789,
        linsys_solver_kwargs={"atol": 1e-6, "btol": 1e-6},
    )

    assert full_result.shape == batch_result.shape
    assert np.all(np.isfinite(full_result))
    assert np.all(np.isfinite(batch_result))
