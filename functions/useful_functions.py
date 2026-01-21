import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import *

from numba import njit, prange, get_num_threads

################################ basic angular conversions and maths ####################################

def radtoarcmin(angle_rad):
    """
    This function converts an an angle expressed in radians
    into arcmins.
    """
    
    angle_arcmin = angle_rad * 60 * 180 / np.pi
    
    return angle_arcmin


def arcmintorad(angle_arcmin):
    """
    This function converts an an angle expressed in arcmins
    into radians.
    """
    
    angle_rad = angle_arcmin / (60 * 180 / np.pi)
    
    return angle_rad

def delta_func(a,b):
    if a == b:
        x = 1
    else:
        x = 0

    return x

def cos_law_side(b,c,A):

    number = b**2 + c**2 - 2*b*c*np.cos(A)

    if np.any(number < 0):
        print("warning! number = ", number)
    
    return np.sqrt(b**2 + c**2 - 2*b*c*np.cos(A))

def cos_law_angle(b, c, a):
    b = np.asarray(b)
    c = np.asarray(c)
    a = np.asarray(a)
    
    denominator = 2 * b * c
    
    if np.any(denominator == 0):
        raise ValueError("Invalid input: some values of b or c are zero, leading to division by zero.")
    
    cos_angle = (b**2 + c**2 - a**2) / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.arccos(cos_angle)

def sin2(x):
    return np.sin(2*x)

def cos2(x):
    return np.cos(2*x)


################################ JIT-compiled versions for fast Monte Carlo integration ####################################

@njit
def sin2_jit(x):
    """JIT-compiled sin(2x)."""
    return np.sin(2*x)

@njit
def cos2_jit(x):
    """JIT-compiled cos(2x)."""
    return np.cos(2*x)

@njit
def cos_law_side_jit(b, c, A):
    """JIT-compiled law of cosines to find side a given sides b, c and angle A."""
    value = b**2 + c**2 - 2*b*c*np.cos(A)
    # Clip small negative values to avoid NaNs from roundoff.
    value = np.maximum(value, 0.0)
    return np.sqrt(value)

@njit
def cos_law_angle_jit(b, c, a):
    """JIT-compiled law of cosines to find angle A given sides a, b, c."""
    denom = 2 * b * c
    cos_angle = (b**2 + c**2 - a**2) / denom
    # Guard degenerate triangles to avoid NaNs.
    cos_angle = np.where(denom == 0, 1.0, cos_angle)
    # Clip to [-1, 1] to handle numerical precision issues.
    cos_angle = np.minimum(np.maximum(cos_angle, -1.0), 1.0)
    return np.arccos(cos_angle)

@njit
def interp_jit(x, xp, fp):
    """
    JIT-compiled linear interpolation (similar to np.interp but works in nopython mode).
    x: values at which to interpolate (1D array)
    xp: x-coordinates of data points (1D array, must be increasing)
    fp: y-coordinates of data points (1D array)
    Returns: interpolated values at x
    """
    n = len(x)
    result = np.empty(n)
    nxp = len(xp)

    for i in range(n):
        xi = x[i]
        # Handle out of bounds
        if xi <= xp[0]:
            result[i] = fp[0]
        elif xi >= xp[nxp-1]:
            result[i] = fp[nxp-1]
        else:
            # Binary search for the interval
            lo = 0
            hi = nxp - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if xp[mid] <= xi:
                    lo = mid
                else:
                    hi = mid
            # Linear interpolation
            t = (xi - xp[lo]) / (xp[hi] - xp[lo])
            result[i] = fp[lo] + t * (fp[hi] - fp[lo])

    return result


@njit
def interp_index_weight_jit(x, xp):
    """
    Pre-compute indices and weights for linear interpolation on a fixed grid.
    Returns:
        idx: lower grid indices for each x
        t: interpolation weights in [0, 1]
    """
    n = len(x)
    idx = np.empty(n, dtype=np.int64)
    t = np.empty(n)
    nxp = len(xp)

    for i in range(n):
        xi = x[i]
        if xi <= xp[0]:
            idx[i] = 0
            t[i] = 0.0
        elif xi >= xp[nxp - 1]:
            idx[i] = nxp - 2
            t[i] = 1.0
        else:
            lo = 0
            hi = nxp - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if xp[mid] <= xi:
                    lo = mid
                else:
                    hi = mid
            idx[i] = lo
            denom = xp[lo + 1] - xp[lo]
            if denom == 0:
                t[i] = 0.0
            else:
                t[i] = (xi - xp[lo]) / denom

    return idx, t


@njit
def interp_index_weight_uniform_jit(x, x_min, inv_dx, n):
    """
    Pre-compute indices and weights for linear interpolation on a uniform grid.
    """
    m = len(x)
    idx = np.empty(m, dtype=np.int64)
    t = np.empty(m)
    dx = 1.0 / inv_dx

    for i in range(m):
        xi = x[i]
        if xi <= x_min:
            idx[i] = 0
            t[i] = 0.0
        else:
            rel = (xi - x_min) * inv_dx
            j = int(rel)
            if j >= n - 1:
                idx[i] = n - 2
                t[i] = 1.0
            else:
                idx[i] = j
                t[i] = (xi - (x_min + j * dx)) * inv_dx

    return idx, t


@njit
def interp_eval_jit(idx, t, fp):
    """
    Evaluate linear interpolation given pre-computed indices and weights.
    """
    n = len(idx)
    result = np.empty(n)
    for i in range(n):
        j = idx[i]
        result[i] = fp[j] + t[i] * (fp[j + 1] - fp[j])
    return result


_GRID_CACHE = {}


def spline_to_grid(spline_func, r_min, r_max, n_points=1000, use_cache=True):
    """
    Convert a spline function to a grid for fast JIT-compatible interpolation.

    Parameters:
        spline_func: A callable (e.g., CubicSpline) that evaluates the correlation function
        r_min: Minimum radius
        r_max: Maximum radius
        n_points: Number of grid points

    Returns:
        r_grid: 1D array of r values
        f_grid: 1D array of function values at r_grid
    """
    cache_key = None
    if use_cache:
        cache_key = (id(spline_func), float(r_min), float(r_max), int(n_points))
        cached = _GRID_CACHE.get(cache_key)
        if cached is not None:
            return cached

    r_grid = np.linspace(r_min, r_max, n_points)
    f_grid = spline_func(r_grid)
    if use_cache:
        _GRID_CACHE[cache_key] = (r_grid, f_grid)
    return r_grid, f_grid


def annuli_intersection_area(i1, o1, i2, o2):
    """
    Computes the intersection area between two concentric annuli.
    
    Parameters:
        i1, o1 : float - Inner and outer radii of first annulus
        i2, o2 : float - Inner and outer radii of second annulus

    Returns:
        float - Area of intersection (0 if no intersection)
    """
    # Compute the overlapping radial range
    r_inner = max(i1, i2)
    r_outer = min(o1, o2)

    if r_outer <= r_inner:
        return 0.0  # No overlap

    # Area of the overlapping annulus
    area = np.pi * (r_outer**2 - r_inner**2)
    
    return area

####################################### printing ############################################

def roundsf(x, sig_figs=1):
    if x == 0:
        return 0  # Avoid log10 issues
    power = math.floor(math.log10(abs(x)))  # Find order of magnitude
    factor = 10 ** power  # Get the scaling factor
    return round(x / factor) * factor  # Round and scale back

############################# file saving and dictionary reading #############################

def save_pickle(data, filename, descriptor):
    """Helper function to save data to a pickle file, creating folders if needed."""
    try:
        dirpath = os.path.dirname(filename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"Successfully saved {descriptor} at {filename}")
    except Exception as ex:
        print(f"Error during pickling {descriptor}: {ex}")

def add_dict(*objects):
    """Automatically adds multiple objects to global_dict with their variable names as keys."""
    frame = inspect.currentframe().f_back
    for name, value in frame.f_locals.items():
        if any(value is obj for obj in objects):  # Use identity check
            global_dict[name] = value  # Add it to global_dict

def get_item(*names):
    """Retrieves items from global_dict and defines them as global variables."""
    frame = inspect.currentframe().f_back  # Get caller's frame
    caller_globals = frame.f_globals  # Access the caller's global namespace
    
    for name in names:
        if name in global_dict:
            caller_globals[name] = global_dict[name]  # Define the variable globally
        else:
            raise KeyError(f"'{name}' not found in global_dict")

def load_correlations(filename="correlations"):
    """Loads a pickled dictionary from 'filename' and adds its contents to global_dict."""
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)  # Load the dictionary from the pickle file
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")
    except pickle.UnpicklingError:
        raise ValueError(f"File '{filename}' is not a valid pickle file.")

    if isinstance(data, dict):
        global_dict.update(data)  # Merge the loaded dictionary into global_dict
    else:
        raise ValueError("The pickled file does not contain a dictionary.")

def load_file(filename):
    """Loads a pickled dictionary"""
    with open(filename, "rb") as f:
        data = pickle.load(f)  # Load the dictionary from the pickle file

    return(data)

############################### integrals, antiderivatives and solvers ######################################

def radial_integration(correlation_function, Theta_start, Theta_end):

    integrand = lambda x: x*correlation_function(x)
    
    integral, err = quad(integrand, Theta_start, Theta_end)

    return integral
    
def compute_antiderivative(function, thetamax_dist = Thetamax_dist):

    Thetamin_rad = arcmintorad(Thetamin_arcmin) 
    thetamax_rad = arcmintorad(thetamax_dist)
    
    Thetas = np.logspace(np.log10(Thetamin_rad), np.log10(thetamax_rad), theta_resolution)
    Thetas = np.insert(Thetas, 0, 0.0)  # Prepend 0 to the array
    
    antiderivative_list = [0]
    
    for i in range(theta_resolution):
        
        antiderivative = radial_integration(function, 0, Thetas[i + 1])
        antiderivative_list.append(antiderivative)

    return CubicSpline(Thetas, antiderivative_list)

def find_maximum(f, a, b):
    result = minimize_scalar(lambda x: -f(x), bounds=(a, b), method='bounded')
    if result.success:
        x_max = result.x
        f_max = f(x_max)
        return x_max, f_max
    else:
        raise RuntimeError("Failed to find maximum.")

################################# Quasi-Monte Carlo Integration (Sobol) #######################################

@njit(parallel=True)
def _qmc_integrate_parallel_1d(func, all_points, rescale, num_randomizations):
    """
    Parallelized QMC integrator for single-output function.
    Processes multiple randomized Sobol sequences in parallel across cores.

    Parameters:
        func: JIT-compiled integrand function
        all_points: Pre-generated points, shape (num_randomizations, dim, num_samples)
        rescale: Scaling factor for numerical stability
        num_randomizations: Number of independent randomized sequences

    Returns:
        results: Array of integral values from each randomization, shape (num_randomizations,)
    """
    results = np.empty(num_randomizations)

    # Parallelize across randomizations - each core processes one randomization
    for r in prange(num_randomizations):
        points = all_points[r]  # Shape: (dim, num_samples)
        num_samples = points.shape[1]

        # Evaluate function at all points
        values = func(points)

        # Compute mean with rescaling for stability
        total = 0.0
        for i in range(num_samples):
            total += values[i] * rescale

        mean_f = total / num_samples
        results[r] = mean_f / rescale

    return results


@njit(parallel=True)
def _qmc_integrate_parallel_2d(func, all_points, rescale, num_randomizations):
    """
    Parallelized QMC integrator for multi-output function.
    Processes multiple randomized Sobol sequences in parallel across cores.

    Parameters:
        func: JIT-compiled integrand function returning (n_outputs, n_samples)
        all_points: Pre-generated points, shape (num_randomizations, dim, num_samples)
        rescale: Scaling factors for each output, shape (n_outputs,)
        num_randomizations: Number of independent randomized sequences

    Returns:
        results: Array of integral values, shape (num_randomizations, n_outputs)
    """
    n_outputs = rescale.shape[0]
    results = np.empty((num_randomizations, n_outputs))

    # Parallelize across randomizations - each core processes one randomization
    for r in prange(num_randomizations):
        points = all_points[r]  # Shape: (dim, num_samples)
        num_samples = points.shape[1]

        # Evaluate function at all points
        values = func(points)

        # Compute mean for each output
        for o in range(n_outputs):
            total = 0.0
            rescale_o = rescale[o]
            for i in range(num_samples):
                total += values[o, i] * rescale_o
            mean_f = total / num_samples
            results[r, o] = mean_f / rescale_o

    return results


def quasi_monte_carlo_integrate(
    func,
    bounds,
    num_samples=nsamp,
    num_randomizations=None,
    seed=None,
    max_batch_mem_mb=1024,
):
    """
    Quasi-Monte Carlo integration using scrambled Sobol sequences with parallel execution.

    This uses pre-generated Sobol points for better convergence than pseudo-random MC.
    For smooth integrands, QMC error scales as ~1/N compared to ~1/sqrt(N) for MC.
    All randomizations are processed in parallel across available CPU cores.

    Parameters:
        func: JIT-compiled integrand function
              - For 1D output: returns array of shape (n_samples,)
              - For multi-output: returns array of shape (n_outputs, n_samples)
        bounds: List of tuples [(a1, b1), (a2, b2), ...] defining integration domain
        num_samples: Number of QMC points to use (per randomization)
        num_randomizations: Number of randomized Sobol sequences for error estimation
                          If None, defaults to:
                            1. LOSCOV_NUM_RANDOMIZATIONS env var if set
                            2. max(4, current Numba thread count) otherwise
                          Minimum 4 ensures valid error estimates even in parallel job mode
                          Higher values give better error estimates but cost more
        seed: Random seed for reproducibility (affects scrambling)
        max_batch_mem_mb: Upper bound for batch point storage in MB. Lowering this
                          reduces peak memory at the cost of parallelism.

    Returns:
        integral: Mean integral value (or list for multi-output)
        error: Standard error estimated from variance across randomizations

    Notes:
        - QMC works best for smooth integrands in dimensions d < 15
        - Error estimation uses variance across multiple scrambled sequences
        - Total function evaluations = num_samples * num_randomizations
        - Parallelization: randomizations processed across CPU cores using Numba prange
        - For 16-core CPU, num_randomizations=16 gives optimal core utilization
    """
    from scipy.stats import qmc

    dim = len(bounds)
    num_samples = int(num_samples)
    bounds_arr = np.array(bounds, dtype=np.float64)

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if num_randomizations is None:
        # Check for environment variable override first
        env_nrand = os.getenv("LOSCOV_NUM_RANDOMIZATIONS")
        if env_nrand:
            try:
                num_randomizations = int(env_nrand)
            except ValueError:
                num_randomizations = max(4, int(get_num_threads()))
        else:
            # Default to available threads, but require minimum 4 for reasonable error estimation
            # This ensures error computation works even when NUMBA_NUM_THREADS=1 (parallel job mode)
            num_randomizations = max(4, int(get_num_threads()))

    if num_randomizations <= 0:
        raise ValueError("num_randomizations must be positive.")

    # Determine if multi-output by testing with small sample
    rng = np.random.default_rng(seed)
    test_points = np.array([rng.uniform(low=a, high=b, size=10) for a, b in bounds])
    test_output = func(test_points)

    multi_output = test_output.ndim == 2
    if multi_output:
        n_outputs = test_output.shape[0]
    else:
        n_outputs = 1

    # Compute rescale factors for numerical stability
    n_subsample = min(100, num_samples)
    subsample_points = np.array([rng.uniform(low=a, high=b, size=n_subsample) for a, b in bounds])
    subsample_output = func(subsample_points)

    if multi_output:
        rescale = np.empty(n_outputs, dtype=np.float64)
        for j in range(n_outputs):
            typical_scale = np.median(np.abs(subsample_output[j]))
            rescale[j] = 1.0 if typical_scale == 0 else 1.0 / typical_scale
    else:
        typical_scale = np.median(np.abs(subsample_output))
        rescale = np.array([1.0 if typical_scale == 0 else 1.0 / typical_scale], dtype=np.float64)

    results = np.empty((num_randomizations, n_outputs)) if multi_output else np.empty(num_randomizations)

    # Batch randomizations to limit peak memory.
    batch_randomizations = min(int(get_num_threads()), num_randomizations)
    if max_batch_mem_mb is not None:
        bytes_per_randomization = (dim + n_outputs) * int(num_samples) * 8
        max_batch_by_mem = max(1, int((max_batch_mem_mb * 1024 * 1024) // bytes_per_randomization))
        batch_randomizations = min(batch_randomizations, max_batch_by_mem)

    # Pre-generate and process Sobol sequences in batches to cap memory.
    # Batch shape: (batch_randomizations, dim, num_samples)
    use_base2 = (num_samples & (num_samples - 1)) == 0
    base2_m = int(np.log2(num_samples)) if use_base2 else 0
    for start in range(0, num_randomizations, batch_randomizations):
        batch = min(batch_randomizations, num_randomizations - start)
        all_points = np.empty((batch, dim, num_samples), dtype=np.float64)

        for i in range(batch):
            # Generate scrambled Sobol sequence (each randomization gets different scramble)
            seq_seed = None if seed is None else seed + start + i
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seq_seed)
            if use_base2:
                qmc_points = sampler.random_base2(base2_m)
            else:
                qmc_points = sampler.random(n=num_samples)  # Shape: (num_samples, dim)

            # Scale points to integration bounds and transpose for JIT function format
            for d in range(dim):
                low, high = bounds_arr[d]
                all_points[i, d, :] = low + qmc_points[:, d] * (high - low)

        # Compute integrals in parallel across the batch
        if multi_output:
            batch_results = _qmc_integrate_parallel_2d(func, all_points, rescale, batch)
            results[start:start + batch, :] = batch_results
        else:
            batch_results = _qmc_integrate_parallel_1d(func, all_points, rescale[0], batch)
            results[start:start + batch] = batch_results

    # Compute total volume
    total_volume = 1.0
    for low, high in bounds:
        total_volume *= (high - low)

    # Estimate integral and error from randomizations
    if multi_output:
        # results shape: (num_randomizations, n_outputs)
        mean_integrals = np.mean(results, axis=0) * total_volume
        if num_randomizations > 1:
            std_integrals = np.std(results, axis=0, ddof=1) * total_volume / np.sqrt(num_randomizations)
        else:
            std_integrals = np.zeros(n_outputs)
        return mean_integrals.tolist(), std_integrals.tolist()
    else:
        # results shape: (num_randomizations,)
        mean_integral = np.mean(results) * total_volume
        if num_randomizations > 1:
            std_integral = np.std(results, ddof=1) * total_volume / np.sqrt(num_randomizations)
        else:
            std_integral = 0.0
        return float(mean_integral), float(std_integral)


def test_err(error, signal, name):

    if signal != 0:
        if np.abs(error/signal) > total_error_threshold:
            print(f"Warning! Total error for {name} is {round(np.abs(error/signal),1)}") 
