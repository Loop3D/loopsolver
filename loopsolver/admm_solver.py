import numpy as np
from loopsolver.admm_method import ADMM
from dataclasses import dataclass
from scipy.sparse.linalg import lsmr
from scipy.sparse import vstack, csr_matrix
import tqdm

@dataclass
class Config:
    verbose: bool = False
    progress: bool = True


progressbar = lambda x: x

try:
    import tqdm

    if Config.progress:
        progessbar = tqdm.tqdm
    else:
        progressbar = lambda x: x
except ImportError:
    Config.progress = False
    progressbar = lambda x: x


def admm_solve(
    A: csr_matrix,
    b: np.ndarray,
    Q: csr_matrix,
    bounds: np.ndarray,
    x0: np.ndarray,
    admm_weight: float = 0.1,
    nmajor=200,
    linsys_solver_kwargs={},
    linsys_solver="lsmr",
    batch_size: int = None,
    batch_fraction: float = None,
    random_seed: int = None,
):
    if A.shape[1] != x0.shape[0]:
        raise ValueError("Number of columns in interpolation matrix does not match x0")
    if A.shape[1] != Q.shape[1]:
        raise ValueError(
            "Number of columns in interpolation matrix and inequality matrix are different "
        )
    if Q.shape[0] != bounds.shape[0]:
        raise ValueError("Number of rows in inequality matrix and bounds are different")
    if bounds.shape[1] == 2:
        bounds = np.hstack([bounds, np.ones((bounds.shape[0], 1))])
    if bounds.shape[1] != 3:
        raise ValueError("Bounds must have two columns")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Number of rows in interpolation matrix and b are different")
    n_ie = bounds.shape[0]
    
    # Setup batch mode
    use_batch_mode = False
    n_batch = n_ie
    rng = None
    
    if batch_size is not None and batch_fraction is not None:
        raise ValueError("Cannot specify both batch_size and batch_fraction")
    
    if batch_size is not None:
        if batch_size < 1 or batch_size > n_ie:
            raise ValueError(f"batch_size must be between 1 and {n_ie}")
        n_batch = batch_size
        use_batch_mode = True
    elif batch_fraction is not None:
        if batch_fraction <= 0 or batch_fraction > 1:
            raise ValueError("batch_fraction must be between 0 and 1")
        n_batch = max(1, int(n_ie * batch_fraction))
        use_batch_mode = True
    
    if use_batch_mode and random_seed is not None:
        rng = np.random.RandomState(random_seed)
    elif use_batch_mode:
        rng = np.random.RandomState()
    
    qx_val = np.zeros((Q.shape[0], 1))
    model = np.zeros(A.shape[1])
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
    Q *= admm_weight
    matrix = vstack([A, Q])
    for k in linsys_solver_kwargs:
        if not hasattr(linsys_solver_kwargs[k], '__len__') or len(linsys_solver_kwargs[k]) != nmajor:
            linsys_solver_kwargs[k] = [linsys_solver_kwargs[k]] * nmajor
    for _i in tqdm.tqdm(range(nmajor)):
        # Sample batch of inequality constraints if batch mode is enabled
        if use_batch_mode:
            batch_idx = rng.choice(n_ie, size=n_batch, replace=False)
            batch_idx = np.sort(batch_idx)  # Sort for consistent sparse matrix operations
            Q_batch = Q[batch_idx, :]
            xmin_batch = xmin[batch_idx, :]
            xmax_batch = xmax[batch_idx, :]
            matrix = vstack([A, Q_batch])
            b = np.zeros(A.shape[0] + n_batch)
            
            # Create a temporary ADMM object for the batch if needed
            # This maintains z and u variables only for the sampled constraints
            admm_batch = ADMM(n_batch)
            admm_batch.z = admm_method.z[batch_idx].copy()
            admm_batch.u = admm_method.u[batch_idx].copy()
        else:
            batch_idx = None
            
        # current model value
        Mx = matrix @ model  # np.dot(A, model)
        b[:A_size] = b0[:A_size] - Mx[:A_size]

        if Q.shape[0] > 0:
            if use_batch_mode:
                qx_val_batch = np.zeros((n_batch, 1))
                qx_val_batch[:, 0] = Mx[A_size:] / admm_weight
                x0_ADMM_batch = admm_batch.admm_method_iterate_admm_array(xmin_batch, xmax_batch, qx_val_batch)
                b[A_size:] = -admm_weight * (qx_val_batch[:, 0] - x0_ADMM_batch)
                
                # Update the main ADMM state with the batch results
                admm_method.z[batch_idx] = admm_batch.z
                admm_method.u[batch_idx] = admm_batch.u
            else:
                qx_val[:, 0] = Mx[A_size:,] / admm_weight
                x0_ADMM = admm_method.admm_method_iterate_admm_array(xmin, xmax, qx_val)
                # print(x0_ADMM, qx_val.shape)
                # raise Exception
                b[A_size:] = -admm_weight * (qx_val[:, 0] - x0_ADMM)
        # cost_data1 = np.linalg.norm(b[:A_size])
        # cost_data2 = np.linalg.norm(b0[A_size:])
        # model_norm = np.linalg.norm(model)
        if Config.verbose:
            if use_batch_mode:
                # In batch mode, compute metrics on the full constraint set
                Qx_full = Q @ model
                cost_admm1 = np.linalg.norm(Qx_full / admm_weight - admm_method.z)
                cost_admm2 = np.linalg.norm(admm_method.z)
                print("----------------------------------------")
                print(f"it = {_i} (batch mode: {n_batch}/{n_ie} constraints)")
                print(f"cost_admm = {cost_admm1 / cost_admm2 if cost_admm2 > 0 else -1.0}")
                print("----------------------------------------")
            else:
                cost_data = -1.0
                cost_data_model = 0.0
                # if cost_data2 > 0:
                #     cost_data = cost_data1 / cost_data2
                # if model_norm > 0:
                #     cost_data_model = cost_data1 / model_norm
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
        linsys_kwargs = {k:v[_i] for k,v in linsys_solver_kwargs.items()}
        x = lsmr(matrix, b, **linsys_kwargs)
        model += x[0]
    return model
