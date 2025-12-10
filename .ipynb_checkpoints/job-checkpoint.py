import sys

##############################################################################################################################
################################################ 1. PRELIMINARIES ############################################################
##############################################################################################################################

from config import *                                #all constants, defined in the config.py file

from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file

load_correlations(filename=f'correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}')
angular_distributions = load_file(f"data/{suffix}/angular_distributions")
redshift_distributions = load_file(f"data/{suffix}/redshift_distributions")

add_dict(angular_distributions)
add_dict(redshift_distributions)

from functions.angular_distributions import *

from functions.covariance.LLLL import *
from functions.covariance.LELE import *
from functions.covariance.LLLE import *
from functions.covariance.LPLP import *
from functions.covariance.LLLP import *
from functions.covariance.LELP import *

##############################################################################################################################
######################################### 5. GENERATING COVARIANCE MATRICES ##################################################
##############################################################################################################################
            
def compute_covariance_piece(args):
    """Computes a specific covariance component."""
    b1, b2, cov_matrix, cov_type = args

    func = globals().get(f"generate_{cov_type}_{cov_matrix}")
    if func is None:
        print(f"Error: Covariance function {cov_type}_{cov_matrix} not found.")
        return (b1, b2, cov_matrix, cov_type, None)

    # Dynamically build argument list
    func_args = []

    # Conditionally add b1 and b2 if they were given
    if b1 is not None:
        func_args.append(b1)
    if b2 is not None:
        func_args.append(b2)

    result = func(*func_args)

    return (b1, b2, cov_matrix, cov_type, result)

# Read command-line arguments
args = sys.argv[1:]

if len(args) == 2:
    b1 = b2 = None
    cov_matrix, cov_type = args
elif len(args) == 3:
    b1 = int(args[0])
    b2 = None
    cov_matrix, cov_type = args[1:]
elif len(args) == 4:
    b1 = int(args[0])
    b2 = int(args[1])
    cov_matrix, cov_type = args[2:]
else:
    # print("arguments = ", args)
    raise ValueError("Expected either 2, 3, or 4 arguments: [b1] [b2] cov_matrix cov_type")

# Compute covariance piece
result = compute_covariance_piece((b1, b2, cov_matrix, cov_type))

matrices_folder = f'data/{suffix}/covariance'

# Save the result immediately
if result is not None:
    b1, b2, cov_matrix, cov_type, data = result

    if data is not None:
        output_dir = f"{matrices_folder}/{cov_matrix}"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/{cov_type}"
        if b1 is not None:
            filename += f"_{b1}"
        if b2 is not None:
            filename_transp = f"{output_dir}/{cov_type}" + f"_{b2}" + f"_{b1}"
            filename += f"_{b2}"

        if cov_type == 'ncov':
            filename_s = f"{output_dir}/scov"
            
            if b1 is not None:
                filename_s += f"_{b1}"
            if b2 is not None:
                filename_transp_s = f"{output_dir}/scov" + f"_{b2}" + f"_{b1}"
                filename_s += f"_{b2}"

        if cov_type == 'ccov':
            save_pickle(data, filename, f"{cov_type} for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")
        
        elif cov_type == 'ncov':
            save_pickle(data[0], filename, f"{cov_type} for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")
            save_pickle(data[1], filename_s, f"scov for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")
        
        if b1 is not None:
            if b2 is not None:
                if b1 != b2 and cov_matrix != 'LELP':

                    if cov_type == 'ccov':
                        save_pickle(np.transpose(data), filename_transp, f"{cov_type} for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")
                    
                    elif cov_type == 'ncov':
                        save_pickle(np.transpose(data[0]), filename_transp, f"{cov_type} for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")
                        save_pickle(np.transpose(data[1]), filename_transp_s, f"scov for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")
                    

# print(f"Finished task: {b1}, {b2}, {cov_matrix}, {cov_type}")