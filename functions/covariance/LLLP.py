import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

from numba import njit

get_item('LLp','LLx','LP', 'angular_distributions', 'redshift_distributions', 'L0', 'E0')


################################################## JIT-compiled integrands ##############################################################

@njit
def _ccov_integrand_LLLP(params, r_grid, LLp_grid, LLx_grid, LP_D_grid):
    """
    JIT-compiled cosmic covariance integrand for LLLP.
    Computes p, x components with shared geometry calculation.
    """
    psi_j, psi_kd, r_j, r_kd, r_k = params

    # Geometry (computed once for both components)
    y_kj = r_j * np.sin(psi_j)
    x_kj = r_j * np.cos(psi_j) - r_k

    r_kj = np.sqrt(y_kj**2 + x_kj**2)
    psi_kj = np.arctan2(y_kj, x_kj)

    r_jd = cos_law_side_jit(r_kd, r_kj, (psi_kd - psi_kj))
    psi_jd = cos_law_angle_jit(r_kd, r_jd, r_kj) + psi_kd

    # Pre-compute trig functions (used multiple times)
    c2_j = cos2_jit(psi_j)
    s2_j = sin2_jit(psi_j)
    c2_kd = cos2_jit(psi_kd)
    s2_kd = sin2_jit(psi_kd)
    c2_jd_j = cos2_jit(psi_jd - psi_j)

    # Interpolate correlation function values (fast grid lookup)
    idx_rk, t_rk = interp_index_weight_jit(r_k, r_grid)
    idx_rjd, t_rjd = interp_index_weight_jit(r_jd, r_grid)

    LLp_rk = interp_eval_jit(idx_rk, t_rk, LLp_grid)
    LLx_rk = interp_eval_jit(idx_rk, t_rk, LLx_grid)
    LP_rjd = interp_eval_jit(idx_rjd, t_rjd, LP_D_grid)

    # Jacobian
    jacobian = 2 * np.pi * r_k * r_j * r_kd

    # Compute both components
    f_p = LP_rjd * c2_jd_j * (LLp_rk * c2_j * c2_kd + LLx_rk * s2_j * s2_kd)
    f_x = LP_rjd * c2_jd_j * (LLx_rk * c2_j * s2_kd - LLp_rk * s2_j * c2_kd)

    n = len(r_k)
    result = np.empty((2, n))
    result[0] = f_p * jacobian
    result[1] = f_x * jacobian
    return result


@njit
def _ncov_integrand_LLLP(params, r_grid, LP_D_grid):
    """
    JIT-compiled noise covariance integrand for LLLP.
    """
    r_i, r_d, psi_d = params

    # Geometry
    y_id = r_d * np.sin(psi_d)
    x_id = r_d * np.cos(psi_d) - r_i

    r_id = np.sqrt(y_id**2 + x_id**2)
    psi_id = np.arctan2(y_id, x_id)

    # Pre-compute trig functions
    c2_d = cos2_jit(psi_d)
    s2_d = sin2_jit(psi_d)
    c2_id = cos2_jit(psi_id)
    s2_id = sin2_jit(psi_id)

    # Interpolate correlation function value
    idx_rid, t_rid = interp_index_weight_jit(r_id, r_grid)
    LP_rid = interp_eval_jit(idx_rid, t_rid, LP_D_grid)

    # Jacobian
    jacobian = 2 * np.pi * r_i * r_d

    # Compute both components
    f_p = LP_rid * c2_d * c2_id
    f_x = LP_rid * s2_d * s2_id

    n = len(r_i)
    result = np.empty((2, n))
    result[0] = f_p * jacobian
    result[1] = f_x * jacobian
    return result


################################################## LLLP cosmic covariance ##############################################################

def generate_ccov_LLLP(D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - LOS shear cross LOS shear - galaxy position correlation functions.

    D             : the galaxy redshift bin D (0 to Nbinz_P)
    """

    # Pre-compute grids for fast JIT-compiled interpolation (done once for all sign combinations)
    n_grid_points = 2000
    r_grid_max = min(3 * r2_max, Thetamax)
    r_grid, LLp_grid = spline_to_grid(LLp, 0, r_grid_max, n_points=n_grid_points)
    _, LLx_grid = spline_to_grid(LLx, 0, r_grid_max, n_points=n_grid_points)
    _, LP_D_grid = spline_to_grid(LP[D], 0, r_grid_max, n_points=n_grid_points)

    def generate_matrices(sign):

        angular_distribution1 = angular_distributions[f'LL_{sign}']
        angular_distribution2 = angular_distributions['LP'][D]

        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)

        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for Lp
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for Lp (in rad)

        # Initialise the blocks
        ccov_p = np.zeros((Nbin1, Nbin2))
        ccov_x = np.zeros((Nbin1, Nbin2))

        err_p = np.zeros((Nbin1, Nbin2))
        err_x = np.zeros((Nbin1, Nbin2))

        # Wrapper for JIT-compiled integrand
        @njit
        def integrand_all(params):
            """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
            return _ccov_integrand_LLLP(params, r_grid, LLp_grid, LLx_grid, LP_D_grid)

        def integral_bins(alpha, beta):
            """Compute both component integrals with shared samples."""
            ranges = [(0, 2*np.pi), (0, 2*np.pi),
                      (rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, r2_max)]

            integrals, errs = monte_carlo_integrate_jit(integrand_all, ranges, Csamp)

            # normalisation of differential elements
            norm = 2/(Omegatot * Omegas1[alpha] * Omegas2[beta])
            integrals = [i * norm for i in integrals]
            errs = [e * norm for e in errs]
            return integrals, errs

        for alpha in range(Nbin1):
            for beta in range(Nbin2):

                integrals, errs = integral_bins(alpha, beta)
                ccov_p[alpha, beta], ccov_x[alpha, beta] = integrals
                err_p[alpha, beta], err_x[alpha, beta] = errs
    
                test_err(err_p[alpha, beta], ccov_p[alpha, beta], f'LLLP ccov plus redshifts {D} angular bins {alpha, beta}')
                test_err(err_x[alpha, beta], ccov_x[alpha, beta], f'LLLP ccov times redshifts {D} angular bins {alpha, beta}')

        err = np.sqrt(err_p**2+err_x**2)
        
        return ccov_p, ccov_x, err

    #plus

    ccov_p, ccov_x, errp = generate_matrices('plus')
    
    ccovp = ccov_p + ccov_x
    
    #minus

    ccov_p, ccov_x, errm = generate_matrices('minus')
    
    ccovm = ccov_p - ccov_x
    
    ccov = np.block([[ccovm],
                     [ccovp]])
    
    err = np.block([[errm],
                     [errp]])
    
    return ccov, err

################################################## LeLe noise/sparsity covariance #############################################################

def generate_ncov_LLLP(D):
    """
    Computes the contribution of noise and sparsity variance in the
    covariance matrix of the LOS shear - galaxy position correlation functions.

    D             : the galaxy redshift bin D (0 to 4)
    """

    # Pre-compute grids for fast JIT-compiled interpolation (done once for all sign combinations)
    n_grid_points = 2000
    r_grid_max = min(3 * r2_max, Thetamax)
    r_grid, LP_D_grid = spline_to_grid(LP[D], 0, r_grid_max, n_points=n_grid_points)

    def generate_matrices(sign):

        angular_distribution1 = angular_distributions[f'LL_{sign}']
        angular_distribution2 = angular_distributions['LP'][D]

        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL (sign)
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)

        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LP
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LP (in rad)

        # Initialise the blocks
        ncov_p = np.zeros((Nbin1, Nbin2))
        ncov_x = np.zeros((Nbin1, Nbin2))

        nerr_p = np.zeros((Nbin1, Nbin2))
        nerr_x = np.zeros((Nbin1, Nbin2))

        scov_p = np.zeros((Nbin1, Nbin2))
        scov_x = np.zeros((Nbin1, Nbin2))

        serr_p = np.zeros((Nbin1, Nbin2))
        serr_x = np.zeros((Nbin1, Nbin2))

        # Wrapper for JIT-compiled integrand
        @njit
        def integrand_all(params):
            """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
            return _ncov_integrand_LLLP(params, r_grid, LP_D_grid)

        def integral_bins(alpha, beta):
            """Compute both component integrals with shared samples."""
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]

            integrals, errs = monte_carlo_integrate_jit(integrand_all, ranges, Nsamp)

            # normalisation of differential elements
            norm = 1/(Omegas1[alpha] * Omegas2[beta])
            integrals = [i * norm for i in integrals]
            errs = [e * norm for e in errs]
            return integrals, errs

        for alpha in range(Nbin1):
            for beta in range(Nbin2):

                integrals, errs = integral_bins(alpha, beta)
                int_p, int_x = integrals
                err_p, err_x = errs

                ncov_p[alpha, beta] = (sigma_L**2/Nlens) * int_p
                nerr_p[alpha, beta] = (sigma_L**2/Nlens) * err_p

                scov_p[alpha, beta] = (L0/Nlens) * int_p
                serr_p[alpha, beta] = (L0/Nlens) * err_p

                ncov_x[alpha, beta] = (sigma_L**2/Nlens) * int_x
                nerr_x[alpha, beta] = (sigma_L**2/Nlens) * err_x

                scov_x[alpha, beta] = (L0/Nlens) * int_x
                serr_x[alpha, beta] = (L0/Nlens) * err_x
    
                test_err(nerr_p[alpha, beta], ncov_p[alpha, beta], f'LLLP ncov plus redshift {D} angular bins {alpha, beta}')
                test_err(nerr_x[alpha, beta], ncov_x[alpha, beta], f'LLLP ncov times redshift {D} angular bins {alpha, beta}')
    
                test_err(serr_p[alpha, beta], scov_p[alpha, beta], f'LLLP scov plus redshift {D} angular bins {alpha, beta}')
                test_err(serr_x[alpha, beta], scov_x[alpha, beta], f'LLLP scov times redshift {D} angular bins {alpha, beta}')

        nerr = np.sqrt(nerr_p**2 + nerr_x**2)
        serr = np.sqrt(serr_p**2 + serr_x**2)
        
        return ncov_p, ncov_x, scov_p, scov_x, nerr, serr

    #plus

    ncov_p, ncov_x, scov_p, scov_x, nerrp, serrp = generate_matrices('plus')
    
    ncovp = ncov_p + ncov_x
    scovp = scov_p + scov_x

    #minus

    ncov_p, ncov_x, scov_p, scov_x, nerrm, serrm = generate_matrices('minus')
    
    ncovm = ncov_p - ncov_x
    scovm = scov_p - scov_x
    
    ncov = np.block([[ncovm],
                     [ncovp]])
    
    nerr = np.block([[nerrm],
                     [nerrp]])
    
    scov = np.block([[scovm],
                     [scovp]])
    
    serr = np.block([[serrm],
                     [serrp]])
    
    return [ncov, scov], [nerr, serr]
