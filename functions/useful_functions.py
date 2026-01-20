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


def spline_to_grid(spline_func, r_min, r_max, n_points=1000):
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
    r_grid = np.linspace(r_min, r_max, n_points)
    f_grid = spline_func(r_grid)
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

def test_err(error, signal, name):

    if signal != 0:
        if np.abs(error/signal) > total_error_threshold:
            print(f"Warning! Total error for {name} is {round(np.abs(error/signal),1)}") 
