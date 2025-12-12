import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import *

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

    #define lists of lists, with as many lists as we have functions
    batch_sums = [ [] for _ in funcs ]     # will store the sums of f from each batch
    batch_sumsq = [ [] for _ in funcs ]    # will store the sums of f**2 from each batch
    
    batch_ns = []                          # will store the number of samples per batch 
    #as currently written, every element in batch_ns will simply be equal to batch_size

    #proceed batch by batch
    for _ in range(num_batches):

        #draw a random sample of N points on the integration domain (n=batch_size)
        samples = np.array([rng.uniform(low=a, high=b, size=batch_size) for a, b in bounds])

        #we also want a subsample to define our rescaling
        n_subsample = 100
        subsamples = np.array([rng.uniform(low=a, high=b, size=n_subsample) for a, b in bounds])

        #proceed one function at a time
        for i, func in enumerate(funcs):
            
            f_subsample = func(subsamples)
            typical_scale = np.median(np.abs(f_subsample))
            print('typical_scale = ', typical_scale)
            
            if typical_scale == 0:
                rescale_val = 1.0  # function is zero, no rescaling needed
            else:
                rescale_val = 1.0 / typical_scale
            
            values = func(samples) * rescale_val  #evaluate the function at each of the N sampled points 
            
            batch_sums[i].append(np.sum(values)) #store the sum of each of the n function outputs
            batch_sumsq[i].append(np.sum(values**2)) #store the sum of the square of these outputs
        batch_ns.append(values.size)   #store n (should always be batch_size with the current structure)

    final_integrals = [] #will store the final estimated integrals for each function
    errors = [] #will store the errors in those estimates

    for i in range(len(funcs)):
        N = int(np.sum(batch_ns))                        # total number of samples (might be less than nsamp, depending on the batching)
        total_sum = float(np.sum(batch_sums[i]))         # sum of f over all samples
        total_sumsq = float(np.sum(batch_sumsq[i]))      # sum of f^2 over all samples

        mean_f = total_sum / N #the mean of the function across all points

        if N > 1:
            # sample variance
            var_f = (total_sumsq - N * mean_f**2) / (N - 1) #the sample variance
            var_f = max(var_f, 0.0) #in case of numerical errors leading to negative values
        else:
            var_f = 0.0 #no variance if we've just evaluated at a single point

        mean_integral = total_volume * mean_f #the monte carlo integral estimate for this function
        std_error = total_volume * np.sqrt(var_f / N) #the error in the monte carlo integral for this function

        final_integrals.append(mean_integral/rescale_val)
        errors.append(std_error/rescale_val)

    if single_function:
        return final_integrals[0], errors[0]
        
    else:
        return final_integrals, errors

def test_err(error, signal, name):

    if signal != 0:
        if np.abs(error/signal) > total_error_threshold:
            print(f"Warning! Total error for {name} is {round(np.abs(error/signal),1)}") 