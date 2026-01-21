import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import *

from numba import njit, prange

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

###################################### Monte Carlo Integrator #######################################

def monte_carlo_integrate(funcs, bounds, num_samples=nsamp, num_batches = num_batches):
    """
    Monte Carlo integration over a given domain with error estimation.

    Parameters:
    - funcs (callable or list of callables): The function to integrate. NB it needs to accept an array as input
                       for integration over multiple dimensions.
                       A function can also return a 2D array (n_outputs, n_samples) to compute multiple
                       integrals with shared sample points.
    - bounds (list of tuples): Integration bounds [(a1, b1), (a2, b2), ...].
                                For 1D, use [(a, b)].
    - num_samples (int): Total number of random samples to be used in the integration
    - num_batches: The number of batches into which our samples are split, to reduce the memory burden

    Returns:
    - tuple: (float, float) Estimated value of the integral and its error (or a list of tuples if multiple functions inputted)
    """

    #our function is defined to handle multiple integrands evaluated with the same sample of points.
    #if a single function is provided, we redefine it as a single-item list
    if not isinstance(funcs, (list, tuple)):
        funcs = [funcs]
        single_function = True
    else:
        single_function = False

    nsamp_use = int(num_samples) #the total number of samples in the integration
    batch_size = nsamp_use // num_batches #the number of samples per batch (to reduce the memory burden)
    dim = len(bounds) #the dimensions of the integral
    volumes = [b - a for a, b in bounds] #the volume over which each variable in the integral is evaluated
    total_volume = np.prod(volumes) #the total volume spanned by the integration bounds
    rng = np.random.default_rng() #defining a random number generator

    #compute rescale values once at the start for each function (for numerical stability)
    n_subsample = 100
    subsamples = np.array([rng.uniform(low=a, high=b, size=n_subsample) for a, b in bounds])

    # Detect if function returns multiple outputs (2D array)
    test_output = funcs[0](subsamples)
    if test_output.ndim == 2:
        # Function returns (n_outputs, n_samples) - handle as multi-output
        n_outputs = test_output.shape[0]
        multi_output = True
    else:
        n_outputs = 1
        multi_output = False

    # Compute rescale values for each output of each function
    rescale_vals = []
    for func in funcs:
        f_subsample = func(subsamples)
        if multi_output:
            # Compute separate rescale value for each output
            func_rescales = []
            for j in range(n_outputs):
                typical_scale = np.median(np.abs(f_subsample[j]))
                if typical_scale == 0:
                    func_rescales.append(1.0)
                else:
                    func_rescales.append(1.0 / typical_scale)
            rescale_vals.append(func_rescales)
        else:
            typical_scale = np.median(np.abs(f_subsample))
            if typical_scale == 0:
                rescale_vals.append(1.0)
            else:
                rescale_vals.append(1.0 / typical_scale)

    # Initialize batch storage
    if multi_output:
        # For multi-output: batch_sums[func_idx][output_idx] = list of batch sums
        batch_sums = [[[] for _ in range(n_outputs)] for _ in funcs]
        batch_sumsq = [[[] for _ in range(n_outputs)] for _ in funcs]
    else:
        batch_sums = [[] for _ in funcs]
        batch_sumsq = [[] for _ in funcs]

    batch_ns = []

    #proceed batch by batch
    for _ in range(num_batches):

        #draw a random sample of N points on the integration domain (n=batch_size)
        samples = np.array([rng.uniform(low=a, high=b, size=batch_size) for a, b in bounds])
        batch_ns.append(batch_size)

        #proceed one function at a time
        for i, func in enumerate(funcs):
            values = func(samples)

            if multi_output:
                # values has shape (n_outputs, batch_size)
                for j in range(n_outputs):
                    scaled_values = values[j] * rescale_vals[i][j]
                    batch_sums[i][j].append(np.sum(scaled_values))
                    batch_sumsq[i][j].append(np.sum(scaled_values**2))
            else:
                scaled_values = values * rescale_vals[i]
                batch_sums[i].append(np.sum(scaled_values))
                batch_sumsq[i].append(np.sum(scaled_values**2))

    final_integrals = []
    errors = []

    for i in range(len(funcs)):
        if multi_output:
            func_integrals = []
            func_errors = []
            for j in range(n_outputs):
                N = int(np.sum(batch_ns))
                total_sum = float(np.sum(batch_sums[i][j]))
                total_sumsq = float(np.sum(batch_sumsq[i][j]))

                mean_f = total_sum / N

                if N > 1:
                    var_f = (total_sumsq - N * mean_f**2) / (N - 1)
                    var_f = max(var_f, 0.0)
                else:
                    var_f = 0.0

                mean_integral = total_volume * mean_f / rescale_vals[i][j]
                std_error = total_volume * np.sqrt(var_f / N) / rescale_vals[i][j]

                func_integrals.append(mean_integral)
                func_errors.append(std_error)

            final_integrals.append(func_integrals)
            errors.append(func_errors)
        else:
            N = int(np.sum(batch_ns))
            total_sum = float(np.sum(batch_sums[i]))
            total_sumsq = float(np.sum(batch_sumsq[i]))

            mean_f = total_sum / N

            if N > 1:
                var_f = (total_sumsq - N * mean_f**2) / (N - 1)
                var_f = max(var_f, 0.0)
            else:
                var_f = 0.0

            mean_integral = total_volume * mean_f / rescale_vals[i]
            std_error = total_volume * np.sqrt(var_f / N) / rescale_vals[i]

            final_integrals.append(mean_integral)
            errors.append(std_error)

    if single_function:
        return final_integrals[0], errors[0]
    else:
        return final_integrals, errors


@njit(parallel=True)
def _monte_carlo_integrate_jit_1d(func, bounds, num_samples, num_batches, rescale, seed):
    """
    JIT-native Monte Carlo integrator for a single-output function.
    """
    if seed >= 0:
        np.random.seed(seed)

    dim = bounds.shape[0]
    batch_size = num_samples // num_batches

    total_volume = 1.0
    for d in range(dim):
        total_volume *= (bounds[d, 1] - bounds[d, 0])

    N = batch_size * num_batches
    if N == 0:
        return 0.0, 0.0

    batch_sum = np.zeros(num_batches)
    batch_sumsq = np.zeros(num_batches)

    for b in prange(num_batches):
        samples = np.empty((dim, batch_size))
        for d in range(dim):
            low = bounds[d, 0]
            width = bounds[d, 1] - bounds[d, 0]
            for i in range(batch_size):
                samples[d, i] = low + width * np.random.random()

        values = func(samples)

        sum_b = 0.0
        sumsq_b = 0.0
        for i in range(batch_size):
            v = values[i] * rescale
            sum_b += v
            sumsq_b += v * v

        batch_sum[b] = sum_b
        batch_sumsq[b] = sumsq_b

    total_sum = 0.0
    total_sumsq = 0.0
    for b in range(num_batches):
        total_sum += batch_sum[b]
        total_sumsq += batch_sumsq[b]

    mean_f = total_sum / N
    if N > 1:
        var_f = (total_sumsq - N * mean_f * mean_f) / (N - 1)
        if var_f < 0.0:
            var_f = 0.0
    else:
        var_f = 0.0

    mean_integral = total_volume * mean_f / rescale
    std_error = total_volume * np.sqrt(var_f / N) / rescale

    return mean_integral, std_error


@njit(parallel=True)
def _monte_carlo_integrate_jit_2d(func, bounds, num_samples, num_batches, rescale, seed):
    """
    JIT-native Monte Carlo integrator for a multi-output function.
    """
    if seed >= 0:
        np.random.seed(seed)

    dim = bounds.shape[0]
    batch_size = num_samples // num_batches
    n_outputs = rescale.shape[0]

    total_volume = 1.0
    for d in range(dim):
        total_volume *= (bounds[d, 1] - bounds[d, 0])

    N = batch_size * num_batches
    if N == 0:
        return np.zeros(n_outputs), np.zeros(n_outputs)

    batch_sum = np.zeros((num_batches, n_outputs))
    batch_sumsq = np.zeros((num_batches, n_outputs))

    for b in prange(num_batches):
        samples = np.empty((dim, batch_size))
        for d in range(dim):
            low = bounds[d, 0]
            width = bounds[d, 1] - bounds[d, 0]
            for i in range(batch_size):
                samples[d, i] = low + width * np.random.random()

        values = func(samples)

        for o in range(n_outputs):
            r = rescale[o]
            sum_b = 0.0
            sumsq_b = 0.0
            for i in range(batch_size):
                v = values[o, i] * r
                sum_b += v
                sumsq_b += v * v
            batch_sum[b, o] = sum_b
            batch_sumsq[b, o] = sumsq_b

    total_sum = np.zeros(n_outputs)
    total_sumsq = np.zeros(n_outputs)
    for b in range(num_batches):
        for o in range(n_outputs):
            total_sum[o] += batch_sum[b, o]
            total_sumsq[o] += batch_sumsq[b, o]

    mean_f = total_sum / N
    var_f = np.zeros(n_outputs)
    for o in range(n_outputs):
        if N > 1:
            v = (total_sumsq[o] - N * mean_f[o] * mean_f[o]) / (N - 1)
            if v < 0.0:
                v = 0.0
        else:
            v = 0.0
        var_f[o] = v

    mean_integral = total_volume * mean_f / rescale
    std_error = total_volume * np.sqrt(var_f / N) / rescale

    return mean_integral, std_error


def monte_carlo_integrate_jit(func, bounds, num_samples=nsamp, num_batches=num_batches, seed=None):
    """
    Python wrapper for the JIT-native Monte Carlo integrator.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")

    batch_size = int(num_samples) // int(num_batches)
    if batch_size <= 0:
        raise ValueError("num_samples must be >= num_batches.")

    bounds_arr = np.array(bounds, dtype=np.float64)

    rng = np.random.default_rng()
    n_subsample = 100
    subsamples = np.array([rng.uniform(low=a, high=b, size=n_subsample) for a, b in bounds])

    test_output = func(subsamples)
    if test_output.ndim == 2:
        n_outputs = test_output.shape[0]
        rescale = np.empty(n_outputs, dtype=np.float64)
        for j in range(n_outputs):
            typical_scale = np.median(np.abs(test_output[j]))
            rescale[j] = 1.0 if typical_scale == 0 else 1.0 / typical_scale
    else:
        n_outputs = 1
        typical_scale = np.median(np.abs(test_output))
        rescale = np.array([1.0 if typical_scale == 0 else 1.0 / typical_scale], dtype=np.float64)

    jit_seed = -1 if seed is None else int(seed)

    if n_outputs == 1:
        integrals, errors = _monte_carlo_integrate_jit_1d(
            func, bounds_arr, int(num_samples), int(num_batches), rescale[0], jit_seed
        )
        return float(integrals), float(errors)

    integrals, errors = _monte_carlo_integrate_jit_2d(
        func, bounds_arr, int(num_samples), int(num_batches), rescale, jit_seed
    )
    return integrals.tolist(), errors.tolist()

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


def quasi_monte_carlo_integrate(func, bounds, num_samples=nsamp, num_randomizations=16, seed=None):
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
                          Default 16 to fully utilize typical multi-core CPUs
                          Higher values give better error estimates but cost more
        seed: Random seed for reproducibility (affects scrambling)

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
    bounds_arr = np.array(bounds, dtype=np.float64)

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

    # Pre-generate all randomized Sobol sequences
    # Shape: (num_randomizations, dim, num_samples)
    all_points = np.empty((num_randomizations, dim, num_samples), dtype=np.float64)

    for i in range(num_randomizations):
        # Generate scrambled Sobol sequence (each randomization gets different scramble)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=None if seed is None else seed + i)
        qmc_points = sampler.random(n=num_samples)  # Shape: (num_samples, dim)

        # Scale points to integration bounds and transpose for JIT function format
        for d in range(dim):
            low, high = bounds_arr[d]
            all_points[i, d, :] = low + qmc_points[:, d] * (high - low)

    # Compute integrals in parallel across all randomizations
    if multi_output:
        results = _qmc_integrate_parallel_2d(func, all_points, rescale, num_randomizations)
    else:
        results = _qmc_integrate_parallel_1d(func, all_points, rescale[0], num_randomizations)

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
