import numpy as np
from loopsolver.admm_method import ADMM
from dataclasses import dataclass
from scipy.sparse.linalg import lsmr
from scipy.sparse import vstack


@dataclass
class Config:
    verbose: bool = False


def solve(
    A,
    b,
    Q,
    bounds,
    x0,
    rho_ADMM,
    nelements,
    rmin=0,
    nminor=100,
    nmajor=200,
):
    n_ie = bounds.shape[0]
    qx_val = np.zeros((Q.shape[0], 1))
    model = np.zeros(nelements)
    model[:] = x0[:]
    # initialise the admm method, sets up the u and v matrices as 0s
    admm_method = ADMM(n_ie)
    b0 = np.zeros(b.shape)
    b0[:] = b[:]
    # the b vector used for the lsqr soln is the size of A + Q
    b = np.zeros(A.shape[0] + Q.shape[0])
    A_size = A.shape[0]
    xmin = bounds[:, [0]]
    xmax = bounds[:, [1]]
    x0_ADMM = np.zeros(Q.shape[0])
    # scale the Q matrix by the admm f
    Q *= rho_ADMM
    for _i in range(nmajor):
        # current model value
        Mx = vstack([A, Q]) @ model  # np.dot(A, model)

        qx_val[:, 0] = Mx[A_size:,] / rho_ADMM
        x0_ADMM = admm_method.admm_method_iterate_admm_array(xmin, xmax, qx_val)
        # print(x0_ADMM, qx_val.shape)
        # raise Exception
        b[:A_size] = b0[:A_size] - Mx[:A_size]
        b[A_size:] = -rho_ADMM * (qx_val[:, 0] - x0_ADMM)
        cost_data1 = np.linalg.norm(b[:A_size])
        cost_data2 = np.linalg.norm(b0[A_size:])
        model_norm = np.linalg.norm(model)
        if Config.verbose:
            cost_data = -1.0
            cost_data_model = 0.0
            if cost_data2 > 0:
                cost_data = cost_data1 / cost_data2
            if model_norm > 0:
                cost_data_model = cost_data1 / model_norm
            cost_admm1 = np.linalg.norm(qx_val - admm_method.z)
            cost_admm2 = np.linalg.norm(admm_method.z)
            cost_admm = -1.0
            if cost_admm2 > 0:
                cost_admm = cost_admm1 / cost_admm2
            print("----------------------------------------")
            print(f"it = {_i}")
            print("cost_data = ", cost_data)
            print("cost_data_model = ", cost_data_model)
            print("cost_admm = ", cost_admm)
            print("----------------------------------------")
        x = lsmr(vstack([A, Q]), b, maxiter=nminor)
        model += x[0]
    return model
